# Deploy — Render (free tier) + UptimeRobot keep-alive

The whole chatbot backend is one Render web service. Your portfolio site stays
on Lovable; the widget and admin call this backend over HTTPS.

```
kartikworks.co.in (Lovable SPA)
 ├─ index.html <script>  ──────────────► api.kartikworks.co.in ─┐
 └─ /admin (ChatbotAdmin section)  ────► (X-Admin-Token)  ─────┤
                                                                 ▼
                               Render web service (FastAPI, free)
                                 /query /query/stream /widget/* /admin-api/*
                                 daily re-ingestion (Neon-gated)
                                 UptimeRobot pings /health every 5 min
```

External services already in use (keep the same keys): Chroma Cloud,
Neon Postgres (POSTGRES_URI), Portkey, Jina, Logfire.

---

## 1. One-time setup

### Card on file
Render requires a payment method even for the free tier. Get the card issue
sorted first — without it the account can't be created.

### Nice-to-have: generate a long random admin token
```sh
openssl rand -hex 32
```

### Push this repo to GitHub
```sh
cd "E:\1 rag marathon"
git init  # if not already
git add .
git commit -m "Add admin layer, widget, Render deploy config"
git remote add origin https://github.com/<you>/vantage-rag.git
git push -u origin main
```

## 2. Create the web service (Render dashboard)

1. [render.com](https://render.com) → **New → Web Service** → connect the
   `vantage-rag` repo.
2. Render auto-detects `render.yaml` (the blueprint). If it doesn't, choose
   **Native Docker** runtime manually: build = `docker build .`,
   start = `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
3. Pick plan **Free**.
4. In **Environment**, set the `sync: false` secrets as env vars
   (values from your working `.env`):
   - `PORTKEY_API_KEY`, `PORTKEY_PRIMARY_SLUG` (`openrouter`),
     `PORTKEY_PRIMARY_MODEL` (`qwen/qwen3.8-27b:free`)
   - `PORTKEY_MODEL_GUARDRAIL` (`@openrouter/qwen/qwen3.8-27b:free`),
     `PORTKEY_MODEL_PLANNER` (`@gemini/gemini-3.6-flash`),
     `PORTKEY_MODEL_RESPONDER` (`@policy/openai/gpt-oss-20b`)
   - `CHROMA_HOST`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`,
     `CHROMA_COLLECTION`
   - `JINA_API_KEY`
   - `POSTGRES_URI` (Neon — the **unpooled** URL, `?sslmode=require`)
   - `LOGFIRE_TOKEN`
   - `ADMIN_TOKEN` (your random hex string)
   - `GROQ_API_KEY`, `GROQ_FALLBACK_API_KEY`, `GEMINI_API_KEY`
5. Deploy. Watch logs for the startup summary (Chroma collection ready,
   scheduler started, admin tables ready).

## 3. DNS (NS1 / nsone.net)

The domain uses nameservers `dns[1-4].p04.nsone.net` (managed DNS on NS1).

1. In your NS1 dashboard → zone `kartikworks.co.in` → add a **CNAME** record:
   - name: `api`
   - answer/canonical: `<your-service>.onrender.com`
   - TTL: default (60–300s)
2. Render → service → **Settings → Custom Domains**:
   add `api.kartikworks.co.in`. Render provisions the TLS cert automatically.

The apex `kartikworks.co.in` already points at Lovable; nothing changes there.

## 4. Keep-alive (free plan spins down after ~15 min idle)

1. Sign up at [UptimeRobot](https://uptimerobot.com) (free).
2. New monitor: **HTTP(s)**, URL `https://api.kartikworks.co.in/health`,
   interval **5 minutes**, alert when down.
3. The ping holds the service awake. If it ever does sleep, the first request
   cold-starts in a few seconds and the widget shows a friendly "retry" message.

## 5. Site integration

Edit `D:\kartikworks` (details in the repo's own to-do) — the gist:

```html
<!-- index.html, before </body> -->
<script src="https://api.kartikworks.co.in/widget/chat.js"
        data-api="https://api.kartikworks.co.in"
        data-title="Ask Kartik"
        data-subtitle="Powered by Vantage RAG"></script>
```

Push + republish via Lovable. The admin "Chatbot" section (cron toggle,
run-now, job/chat logs, stats) calls `/admin-api/*` with `VITE_ADMIN_TOKEN`.

## Verification

```sh
curl https://api.kartikworks.co.in/health
curl -X POST https://api.kartikworks.co.in/query \
  -H "Content-Type: application/json" \
  -d '{"q":"What projects has Kartik worked on?"}'
```

## Going paid later (true always-on)

When the card works: Render service → **Settings → Instance Type** → change to
a paid plan. The sleep behavior disappears; nothing else changes.

## Notes / limits of the free tier

- 512 MB RAM — fine because the image has no torch. If OOM appears under load,
  upgrading instance type fixes it.
- No persistent disk: the `data/` corpus and `processed_data/` live inside the
  image/container, so the daily tick re-reads the same baked-in docs. Raw
  document changes require a redeploy (push → Render rebuild).
- `pdfplumber` (scanned-PDF fallback) is not installed in prod; image-heavy
  PDFs may extract partially via pypdf.
- No Redis: gateway cache is off and the rate limiter uses in-memory storage.