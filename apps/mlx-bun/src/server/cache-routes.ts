import type { KvCacheService } from "@mlx-bun/inference/state";
import { errorResponse } from "./http";

export type CacheRouteServices = Pick<KvCacheService, "closeSession" | "flush" | "report">;

/** Administrative HTTP policy over the loaded model's cache service, preserving
 * main's `/admin/cache/*` wire contract. Both routes borrow the service: no
 * execution lease is taken, a session close never waits for a flush in
 * progress, and a close neither deletes checkpoints nor rejects an unknown
 * session. The flush answer adds the service's own report of saved state. */
export function createCacheRoutes(cache: CacheRouteServices) {
  return { async handle(request: Request): Promise<Response | null> {
    const route = `${request.method} ${new URL(request.url).pathname}`;
    if (route !== "POST /admin/cache/session/close" && route !== "POST /admin/cache/flush") return null;
    try {
      if (route === "POST /admin/cache/session/close") {
        let body: unknown;
        try {
          const text = await request.text();
          body = text.trim() ? JSON.parse(text) : {};
        } catch (error) {
          if (request.signal.aborted) return errorResponse(error, route, undefined, request.signal);
          return Response.json({ error: { message: "invalid JSON body" } }, { status: 400 });
        }
        const id = body && typeof body === "object" && !Array.isArray(body) && "session_id" in body ? body.session_id : undefined;
        if (typeof id === "string") cache.closeSession(id);
        return Response.json({ closed: typeof id === "string" });
      }
      const result = await cache.flush();
      const saved = cache.report().saved;
      return Response.json({ ...result, entries: saved?.entries ?? 0,
        longest_durable_prefix_tokens: saved?.longestDurablePrefixTokens ?? 0 },
        { status: result.durable ? 200 : 503 });
    } catch (error) { return errorResponse(error, route, undefined, request.signal); }
  } };
}
