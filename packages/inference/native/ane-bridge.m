// Experimental Apple Neural Engine bridge (C ABI for Bun FFI).
//
// One handle = one program for the ANE from caller-supplied MIL text, through
// the private AppleNeuralEngine.framework in-memory model API (the mechanism
// oMLX ships, Apache-2.0: jundot/omlx qwen35_ane.mm), loaded once. Private API:
// it may break on a macOS update, so every entry point fails soft.
//
// Compiling and loading are separate steps. mbane_program_compile compiles a
// program into the system ANE compile cache (aned, /Library/Caches/com.apple.aned)
// and loads nothing; mbane_program_create only loads a program from that cache
// (about 4 ms) and refuses one the cache lacks rather than compile it, so a
// process that serves never compiles. The cache keys a program by its content
// (MIL, weights and options: the model's hexStringIdentifier) and by the
// code-signing identifier of the process that compiled it: a program one
// executable compiled is absent for another, so the executable that serves is
// the one that compiles (measured on M4 Pro, macOS 27: two builds of one tool
// under different names do not see each other's programs; a rebuild under the
// same name does). A load compiles implicitly when the program's source is
// staged in its directory (measured: 130 ms and the program then cached), so
// only mbane_program_compile stages it, and removes it once compiled.
//
// Inputs and the output are IOSurfaces (one per MIL input, in the program's
// symbol order, which sorts input names alphabetically). Callers write inputs
// through mbane_input(), read the output through mbane_output() after
// mbane_wait(); the GPU may also write them through zero-copy Metal wraps.
// Evaluation runs on a serial GCD queue so the caller (and the GPU) keep
// working while the ANE computes. Built without ARC: the handle owns its
// model and request.
//
// A loaded model holds one direct-path client of the ANE driver until it is
// unloaded, and the driver serves about a thousand across the whole machine
// (M4 Pro, macOS 27: loads blocked with 1020 held, however they were split
// between processes). Past that, the client open inside the load fails and the
// framework retries it every 15 s for as long as the clients stay held, so
// program creation blocks here and in every other ANE user on the machine.
// Layers that run one at a time therefore share one handle when their programs
// are identical (`share`): the handle counts its holders and the last
// mbane_free unloads it. Compiling holds no client (measured: the driver's
// direct-path client count is unchanged across a compile).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <stdint.h>
#include <string.h>

#define MBANE_MAX_INPUTS 8

typedef struct {
  int holders;  // mbane_program_create results not yet passed to mbane_free
  id model;
  id request;
  int inputCount;
  IOSurfaceRef inputs[MBANE_MAX_INPUTS], output;
  int inputLocked[MBANE_MAX_INPUTS], outputLocked;
  dispatch_semaphore_t done;
  int pending;
  char error[512];
} mbane_program;

static dispatch_queue_t queue(void) {
  static dispatch_queue_t q;  // process lifetime
  static dispatch_once_t once;
  dispatch_once(&once, ^{ q = dispatch_queue_create("mlx-bun.ane", DISPATCH_QUEUE_SERIAL); });
  return q;
}

static void put_error(char *dst, int len, NSString *prefix, NSError *error) {
  if (!dst || len <= 0) return;
  NSString *text = error ? [NSString stringWithFormat:@"%@: %@", prefix, error.localizedDescription] : prefix;
  strlcpy(dst, text.UTF8String ?: "ane error", (size_t)len);
}

static int framework_loaded(void) {
  static int available = -1;
  if (available < 0) {
    available = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) &&
      NSClassFromString(@"_ANEInMemoryModelDescriptor") && NSClassFromString(@"_ANEInMemoryModel") &&
      NSClassFromString(@"_ANERequest") && NSClassFromString(@"_ANEIOSurfaceObject");
  }
  return available;
}

static IOSurfaceRef make_surface(size_t n) {
  size_t a = n < 65536 ? 65536 : (n + 65535) & ~(size_t)65535;
  return IOSurfaceCreate((CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(a), (id)kIOSurfaceHeight : @1, (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(a), (id)kIOSurfaceAllocSize : @(a), (id)kIOSurfacePixelFormat : @0 });
}

/** The in-memory model for mil and its weight files (autoreleased; nothing compiled or loaded), or nil.
 *  text and files, when given, receive the compiler's inputs. */
static id describe(const char *mil, int weightCount, const char *const *weightNames, const void *const *weightData,
                   const uint64_t *weightBytes, NSData **text, NSDictionary **files) {
  NSData *source = [NSData dataWithBytes:mil length:strlen(mil)];
  NSMutableDictionary *weights = [NSMutableDictionary dictionary];
  NSMutableDictionary *named = [NSMutableDictionary dictionary];
  for (int i = 0; i < weightCount; i++) {
    NSData *data = [NSData dataWithBytes:weightData[i] length:(NSUInteger)weightBytes[i]];
    NSString *name = [NSString stringWithUTF8String:weightNames[i]];
    weights[[@"@model_path/weights/" stringByAppendingString:name]] = @{@"offset" : @0, @"data" : data};
    named[name] = data;
  }
  if (text) *text = source;
  if (files) *files = named;
  id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(NSClassFromString(@"_ANEInMemoryModelDescriptor"),
    @selector(modelWithMILText:weights:optionsPlist:), source, weights, nil);
  return desc ? ((id (*)(Class, SEL, id))objc_msgSend)(NSClassFromString(@"_ANEInMemoryModel"),
    @selector(inMemoryModelWithDescriptor:), desc) : nil;
}

static NSString *identifier(id model) { return ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier)); }

/** The framework's own check: the program is in the system compile cache for this process's code identity. */
static BOOL compiled(id model) {
  return [model respondsToSelector:@selector(compiledModelExists)] &&
    ((BOOL (*)(id, SEL))objc_msgSend)(model, @selector(compiledModelExists));
}

void mbane_free(void *handle);

// mil: program text. Weight files (BLOBFILE "@model_path/weights/<name>") are
// passed as names + byte ranges, already in the blob format. inputBytes: one
// size per MIL input in symbol order. share: a program that never runs at the
// same time as this one, whose buffers must each be at least as large. When
// the MIL and weights are identical to share's, the result is share itself
// with one more holder (nothing is loaded); otherwise a new program loads on
// share's buffers. Each result is released by one mbane_free. Loads only: a
// program absent from the compile cache fails with *uncompiled set to 1.
void *mbane_program_create(const char *mil, int weightCount, const char *const *weightNames, const void *const *weightData,
                           const uint64_t *weightBytes, int inputCount, const uint64_t *inputBytes, uint64_t outputBytes,
                           void *share, int *uncompiled, char *err, int errlen) {
  @autoreleasepool {
    if (uncompiled) *uncompiled = 0;
    if (!framework_loaded()) { put_error(err, errlen, @"AppleNeuralEngine.framework unavailable", nil); return NULL; }
    if (inputCount < 1 || inputCount > MBANE_MAX_INPUTS) { put_error(err, errlen, @"unsupported input count", nil); return NULL; }
    mbane_program *other = share;
    if (other) {
      int fits = other->inputCount >= inputCount && IOSurfaceGetAllocSize(other->output) >= outputBytes;
      for (int i = 0; fits && i < inputCount; i++) fits = IOSurfaceGetAllocSize(other->inputs[i]) >= inputBytes[i];
      if (!fits) { put_error(err, errlen, @"shared IOSurfaces do not fit", nil); return NULL; }
    }
    id model = describe(mil, weightCount, weightNames, weightData, weightBytes, NULL, NULL);
    if (!model) { put_error(err, errlen, @"ANE in-memory model creation failed", nil); return NULL; }
    // The identifier hashes the MIL, the weights and the options.
    if (other && [identifier(model) isEqualToString:identifier(other->model)]) {
      other->holders++;
      return other;
    }
    if (!compiled(model)) {
      if (uncompiled) *uncompiled = 1;
      put_error(err, errlen, @"not in the system Neural Engine compile cache", nil);
      return NULL;
    }
    mbane_program *h = calloc(1, sizeof(mbane_program));
    if (!h) return NULL;
    h->holders = 1;
    h->inputCount = inputCount;
    NSError *error = nil;
    if (!((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &error)) {
      put_error(err, errlen, @"ANE load failed", error); mbane_free(h); return NULL;
    }
    h->model = [model retain];
    Class surf = NSClassFromString(@"_ANEIOSurfaceObject");
    NSMutableArray *ins = [NSMutableArray array], *idx = [NSMutableArray array];
    for (int i = 0; i < inputCount; i++) {
      h->inputs[i] = other ? (IOSurfaceRef)CFRetain(other->inputs[i]) : make_surface((size_t)inputBytes[i]);
      if (!h->inputs[i]) { put_error(err, errlen, @"IOSurface allocation failed", nil); mbane_free(h); return NULL; }
      [ins addObject:((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surf, @selector(objectWithIOSurface:), h->inputs[i])];
      [idx addObject:@(i)];
    }
    h->output = other ? (IOSurfaceRef)CFRetain(other->output) : make_surface((size_t)outputBytes);
    if (!h->output) { put_error(err, errlen, @"IOSurface allocation failed", nil); mbane_free(h); return NULL; }
    id yo = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surf, @selector(objectWithIOSurface:), h->output);
    h->request = [((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(NSClassFromString(@"_ANERequest"),
      @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
      ins, idx, @[ yo ], @[ @0 ], nil, nil, @0) retain];
    if (!h->request) { put_error(err, errlen, @"ANE request creation failed", nil); mbane_free(h); return NULL; }
    h->done = dispatch_semaphore_create(0);
    return h;
  }
}

/** Compiles the program (mil and weights as for mbane_program_create) into the
 *  system compile cache unless it is there already, staging its source in a
 *  temporary directory that is removed afterwards. Loads nothing. Returns 1
 *  when it compiled now, 2 when the cache already held it, 0 on failure (err). */
int mbane_program_compile(const char *mil, int weightCount, const char *const *weightNames, const void *const *weightData,
                          const uint64_t *weightBytes, char *err, int errlen) {
  @autoreleasepool {
    if (!framework_loaded()) { put_error(err, errlen, @"AppleNeuralEngine.framework unavailable", nil); return 0; }
    NSData *text = nil;
    NSDictionary *files = nil;
    id model = describe(mil, weightCount, weightNames, weightData, weightBytes, &text, &files);
    if (!model) { put_error(err, errlen, @"ANE in-memory model creation failed", nil); return 0; }
    if (compiled(model)) return 2;
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier(model)];
    NSString *wdir = [dir stringByAppendingPathComponent:@"weights"];
    [fm removeItemAtPath:dir error:nil];
    [fm createDirectoryAtPath:wdir withIntermediateDirectories:YES attributes:nil error:nil];
    [text writeToFile:[dir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *name in files) [files[name] writeToFile:[wdir stringByAppendingPathComponent:name] atomically:YES];
    NSError *error = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
    [fm removeItemAtPath:dir error:nil];
    if (!ok) { put_error(err, errlen, @"ANE compile failed", error); return 0; }
    return 1;
  }
}

/** 1 when the system compile cache holds the program for this process's code
 *  identity (the framework's compiledModelExists), 0 when not, -1 on failure (err). */
int mbane_program_cached(const char *mil, int weightCount, const char *const *weightNames, const void *const *weightData,
                         const uint64_t *weightBytes, char *err, int errlen) {
  @autoreleasepool {
    if (!framework_loaded()) { put_error(err, errlen, @"AppleNeuralEngine.framework unavailable", nil); return -1; }
    id model = describe(mil, weightCount, weightNames, weightData, weightBytes, NULL, NULL);
    if (!model) { put_error(err, errlen, @"ANE in-memory model creation failed", nil); return -1; }
    return compiled(model) ? 1 : 0;
  }
}

/** Removes the program from the system compile cache (the framework's
 *  purgeCompiledModel). Returns 1 when the cache no longer holds it, 0 when it
 *  still does, -1 on failure (err). */
int mbane_program_purge(const char *mil, int weightCount, const char *const *weightNames, const void *const *weightData,
                        const uint64_t *weightBytes, char *err, int errlen) {
  @autoreleasepool {
    if (!framework_loaded()) { put_error(err, errlen, @"AppleNeuralEngine.framework unavailable", nil); return -1; }
    id model = describe(mil, weightCount, weightNames, weightData, weightBytes, NULL, NULL);
    if (!model) { put_error(err, errlen, @"ANE in-memory model creation failed", nil); return -1; }
    ((void (*)(id, SEL))objc_msgSend)(model, @selector(purgeCompiledModel));
    return compiled(model) ? 0 : 1;
  }
}

/** Base address of input i, locked for host/GPU writes until the next evaluation. */
void *mbane_input(void *handle, int i) {
  mbane_program *h = handle;
  if (i < 0 || i >= h->inputCount) return NULL;
  if (!h->inputLocked[i]) { IOSurfaceLock(h->inputs[i], 0, NULL); h->inputLocked[i] = 1; }
  return IOSurfaceGetBaseAddress(h->inputs[i]);
}

void *mbane_output(void *handle) {
  mbane_program *h = handle;
  if (!h->outputLocked) { IOSurfaceLock(h->output, kIOSurfaceLockReadOnly, NULL); h->outputLocked = 1; }
  return IOSurfaceGetBaseAddress(h->output);
}

static void release_cpu(mbane_program *h) {
  for (int i = 0; i < h->inputCount; i++)
    if (h->inputLocked[i]) { IOSurfaceUnlock(h->inputs[i], 0, NULL); h->inputLocked[i] = 0; }
  if (h->outputLocked) { IOSurfaceUnlock(h->output, kIOSurfaceLockReadOnly, NULL); h->outputLocked = 0; }
}

static void run(mbane_program *h) {
  @autoreleasepool {
    NSError *error = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(h->model,
      @selector(evaluateWithQoS:options:request:error:), 21, @{}, h->request, &error);
    h->error[0] = 0;
    if (!ok) put_error(h->error, sizeof h->error, @"ANE evaluation failed", error);
  }
}

int mbane_eval_async(void *handle) {
  mbane_program *h = handle;
  if (h->pending) return 0;
  release_cpu(h);
  h->pending = 1;
  dispatch_async(queue(), ^{ run(h); dispatch_semaphore_signal(h->done); });
  return 1;
}

int mbane_wait(void *handle, char *err, int errlen) {
  mbane_program *h = handle;
  if (!h->pending) return 1;
  dispatch_semaphore_wait(h->done, DISPATCH_TIME_FOREVER);
  h->pending = 0;
  if (h->error[0]) { strlcpy(err, h->error, (size_t)errlen); return 0; }
  return 1;
}

/** Releases one holder; the last unloads the model (closing its ANE client) and frees the buffers. */
void mbane_free(void *handle) {
  mbane_program *h = handle;
  if (!h || --h->holders > 0) return;
  @autoreleasepool {
    if (h->pending) { dispatch_semaphore_wait(h->done, DISPATCH_TIME_FOREVER); h->pending = 0; }
    release_cpu(h);
    if (h->model) {
      NSError *error = nil;
      ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(h->model, @selector(unloadWithQoS:error:), 21, &error);
      [h->model release];
    }
    [h->request release];
    if (h->done) dispatch_release(h->done);
    for (int i = 0; i < h->inputCount; i++) if (h->inputs[i]) CFRelease(h->inputs[i]);
    if (h->output) CFRelease(h->output);
  }
  free(h);
}
