export async function onRequest(context) {
  const backendUrl = context.env.BACKEND_URL;
  if (typeof backendUrl !== "string" || backendUrl.length === 0) {
    return Response.json({ pages: "ok", backend_status: "unconfigured" }, { status: 503 });
  }

  try {
    const upstream = backendUrl.replace(/\/$/, "") + "/healthz";
    const headers = new Headers();
    if (typeof context.env.BACKEND_BEARER === "string" && context.env.BACKEND_BEARER.length > 0) {
      headers.set("Authorization", `Bearer ${context.env.BACKEND_BEARER}`);
    }
    if (typeof context.env.P2P_API_KEY === "string" && context.env.P2P_API_KEY.length > 0) {
      headers.set("X-API-Key", context.env.P2P_API_KEY);
    }

    const response = await fetch(upstream, { headers });
    return Response.json({ pages: "ok", backend_status: response.status }, { status: 200 });
  } catch (_) {
    return Response.json({ pages: "ok", backend_status: "unreachable" }, { status: 503 });
  }
}
