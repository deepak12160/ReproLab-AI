export async function onRequest(context) {
  const backendUrl = context.env.BACKEND_URL;
  if (typeof backendUrl !== "string" || backendUrl.length === 0) {
    return Response.json({ error: "backend is not configured" }, { status: 502 });
  }

  try {
    const incoming = new URL(context.request.url);
    const upstream = new URL(backendUrl.replace(/\/$/, "") + incoming.pathname + incoming.search);

    const headers = new Headers(context.request.headers);
    headers.delete("Authorization");
    headers.delete("Host");

    if (typeof context.env.BACKEND_BEARER === "string" && context.env.BACKEND_BEARER.length > 0) {
      headers.set("Authorization", `Bearer ${context.env.BACKEND_BEARER}`);
    }
    if (typeof context.env.P2P_API_KEY === "string" && context.env.P2P_API_KEY.length > 0) {
      headers.set("X-API-Key", context.env.P2P_API_KEY);
    }

    const response = await fetch(upstream.toString(), {
      method: context.request.method,
      headers,
      body: context.request.method === "GET" || context.request.method === "HEAD"
        ? undefined
        : context.request.body
    });

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers
    });
  } catch (_) {
    return Response.json({ error: "backend unreachable" }, { status: 502 });
  }
}
