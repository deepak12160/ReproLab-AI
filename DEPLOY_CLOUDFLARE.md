# Deploying ReproLab AI on Cloudflare

## Why the original deploy failed
The repository root had **no Wrangler config and no package.json** (the app lives in `paper2product/`), so
`npx wrangler deploy` (it installed wrangler 4.166.0 and stopped) had nothing valid to deploy. Even with a config, a
standard Worker cannot run this app: it needs PyMuPDF/numpy/scikit-learn, SQLite on a real filesystem, and subprocesses
for the code sandbox.

## Architecture
`Browser -> Cloudflare Worker (src/index.js) -> Cloudflare Container (paper2product/Dockerfile, FastAPI on :8000)`
- The Worker is a gateway: HTTP Basic auth (password = `APP_PASSWORD`), then forwards to the container and adds the
  internal `X-API-Key`. It refuses to serve (503) if `APP_PASSWORD` is not set.
- The container is your unchanged FastAPI app and frontend (frontend uses relative URLs, no localhost).
- Requires the **Workers Paid plan** (Containers is not on the free plan) and Docker for the build.

## Dashboard settings (Workers & Pages -> Create -> Import a repository)
- Root directory: `/` (repo root, where `wrangler.jsonc` is)
- Build command: `npm ci`  - Deploy command: `npx wrangler deploy`
- Variables and secrets: add as **Secrets** (below)

## Secrets
| Name | Required | Purpose |
|---|---|---|
| `APP_PASSWORD` | yes | login password for the site (any username) |
| `GEMINI_API_KEY` | no | enables LLM MODE; without it the app shows DEMO MODE |
| `P2P_API_KEY` | no | internal key between Worker and container (derived from APP_PASSWORD if unset) |
| `GITHUB_TOKEN`, `HF_TOKEN` | no | optional integrations |

CLI: `npm ci && npx wrangler login && npx wrangler secret put APP_PASSWORD && npx wrangler secret put GEMINI_API_KEY && npx wrangler deploy`

## Test the deployment
```
curl https://<name>.<subdomain>.workers.dev/gateway-health            # gateway only, no auth
curl -u me:<APP_PASSWORD> https://<name>.<subdomain>.workers.dev/healthz   # reaches the container
curl -u me:<APP_PASSWORD> https://<name>.<subdomain>.workers.dev/api/integrations   # LLM MODE or DEMO MODE
```
Then open the URL in a browser and sign in.

## Limitations (honest)
- Container disk is **ephemeral**: SQLite data, uploads and reports are lost when the container sleeps (30 min idle),
  restarts or is redeployed. Download reports you need. Durable storage (R2/D1/external DB) is not implemented.
- `max_instances` is 1 because state is local to the instance.
- The code sandbox is the **local, NOT isolated** mode inside the container. Keep `APP_PASSWORD` private; do not open it publicly.
- Not verified here: the actual Cloudflare deploy and container build (no Cloudflare credentials, no Docker daemon).
  Verified: config accepted by wrangler 4.166.0, Worker bundles, auth works in workerd, app tests 215 passed / 3 skipped.
