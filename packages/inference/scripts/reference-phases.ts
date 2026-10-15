/** Observe the operation the supplied checkout's scheduler actually invokes.
 * Both the named-width and numeric/range contracts are preserved verbatim. */
type Loose = any;
export function observeTargetPhases<Model extends object>(model: Model, observe: (ids: Loose) => void, head?: (out: Loose) => void): Model {
  const phases = model as Loose, overrides: Record<string, unknown> = {};
  const watch = (run: Loose) => (...args: Loose[]) => { observe(args[0]); return run(...args); };
  const table = (entries: Loose, transform = watch) => {
    const listed = Object.freeze(Object.fromEntries(Object.entries(entries).map(([rows, run]) => [rows, transform(run)])));
    return listed;
  };
  const ranges = (entries: Loose[]) => entries.map(range => ({...range,operation:watch(range.operation)}));
  for (const name of ["decode","verify","appendKnown","prefillTail"]) {
    if (phases[name]) overrides[name] = Array.isArray(phases[name]) ? ranges(phases[name]) : table(phases[name]);
  }
  for (const name of ["prompt","multiSequencePrompt","promptChunk","multiSequencePromptChunk"]) if (phases[name]) overrides[name] = ranges(phases[name]);
  if (phases.prefillChunk) overrides.prefillChunk = {...phases.prefillChunk,
    gpu:watch(phases.prefillChunk.gpu),
    ane:(phases.prefillChunk.ane ?? []).map((entry: Loose)=>({...entry,forward:watch(entry.forward)}))};
  if (head && phases.logits) {
    const watchHead = (run: Loose) => (...args: Loose[]) => {
      const out = run(...args);
      try { head(out); } catch (error) { out.dispose(); throw error; }
      return out;
    };
    overrides.logits = Array.isArray(phases.logits)
      ? phases.logits.map((range: Loose) => ({...range, operation: watchHead(range.operation)}))
      : table(phases.logits, watchHead);
  }
  return new Proxy(model, {get(target,key) {
    if (typeof key === "string" && key in overrides) return overrides[key];
    const value = Reflect.get(target,key,target);
    return typeof value === "function" ? value.bind(target) : value;
  }});
}
