async function sha256(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return new Uint8Array(digest);
}

async function constantTimeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const aDigest = await sha256(a);
  const bDigest = await sha256(b);
  let different = 0;
  for (let i = 0; i < aDigest.length; i++) {
    different |= aDigest[i] ^ bDigest[i];
  }
  return different === 0;
}

export async function onRequest(context) {
  const password = context.env.APP_PASSWORD;
  if (typeof password !== "string" || password.length === 0) {
    return new Response("Service unavailable", { status: 503 });
  }

  const authorization = context.request.headers.get("Authorization") || "";
  const match = /^Basic\s+([A-Za-z0-9+/=_-]+)$/i.exec(authorization.trim());
  let suppliedPassword = "";
  if (match) {
    try {
      suppliedPassword = atob(match[1]).split(":").slice(1).join(":");
    } catch (_) {
      suppliedPassword = "";
    }
  }

  if (!(await constantTimeEqual(suppliedPassword, password))) {
    return new Response("Authentication required", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="ReproLab AI", charset="UTF-8"',
        "Cache-Control": "no-store"
      }
    });
  }

  return context.next();
}
