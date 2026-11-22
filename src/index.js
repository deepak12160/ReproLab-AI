// ReproLab AI - Cloudflare Worker gateway.
// Responsibilities: (1) fail closed unless a password is configured, (2) HTTP Basic auth for every request except /healthz,
// (3) inject the internal X-API-Key so the unchanged frontend works, (4) forward to the FastAPI container.
import { Container, getContainer } from "@cloudflare/containers";

export class ReproLabContainer extends Container {
  defaultPort = 8000;      // uvicorn in paper2product/Dockerfile
  sleepAfter = "30m";      // container sleeps after 30 minutes without requests (its local disk is then lost)
  pingEndpoint = "/healthz";

  constructor(ctx, env) {
    super(ctx, env);
    const e = {
      P2P_SANDBOX: "local",
      P2P_DATA_DIR: "/data",
      P2P_API_KEY: internalKey(env),
    };
    for (const k of ["GEMINI_API_KEY", "GITHUB_TOKEN", "HF_TOKEN", "P2P_LLM_MODEL", "P2P_LLM_PROVIDER",
                     "P2P_MAX_UPLOAD_MB", "P2P_RATE_LIMIT_PER_MIN"]) {
      if (env[k]) e[k] = String(env[k]);
    }
    this.envVars = e;
  }

  onError(error) {
    console.error("container error", String(error));
  }
}

function internalKey(env) {
  return env.P2P_API_KEY || ("gw-" + (env.APP_PASSWORD || "unset"));
}

function timingSafeEqual(a, b) {
  const enc = new TextEncoder();
  const x = enc.encode(a), y = enc.encode(b);
  let d = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) d |= (x[i] || 0) ^ (y[i] || 0);
  return d === 0;
}

function authorised(request, env) {
  const h = request.headers.get("authorization") || "";
  if (!h.startsWith("Basic ")) return false;
  let decoded = "";
  try { decoded = atob(h.slice(6)); } catch { return false; }
  const pw = decoded.slice(decoded.indexOf(":") + 1);       // username is ignored
  return timingSafeEqual(pw, env.APP_PASSWORD);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/gateway-health") {
      return Response.json({ status: "ok", gateway: true, container_binding: !!env.REPROLAB });
    }
    if (!env.APP_PASSWORD) {
      return new Response("ReproLab AI is not configured: set the APP_PASSWORD secret (npx wrangler secret put APP_PASSWORD). " +
                          "The gateway refuses to serve an unauthenticated code-running app.", { status: 503 });
    }
    if (!authorised(request, env)) {
      return new Response("Authentication required", { status: 401, headers: { "WWW-Authenticate": 'Basic realm="ReproLab AI"' } });
    }
    const headers = new Headers(request.headers);
    headers.set("x-api-key", internalKey(env));
    headers.delete("authorization");
    const container = getContainer(env.REPROLAB, "main");
    try {
      return await container.fetch(new Request(request, { headers }));
    } catch (e) {
      console.error("proxy failure", String(e));
      return new Response("The ReproLab backend container is starting or unavailable. Retry in a few seconds.",
                          { status: 502, headers: { "Retry-After": "5" } });
    }
  },
};
