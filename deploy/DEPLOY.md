# Deploy in three steps

Free, no card required. About 25 minutes, most of it waiting.

```
Neon  ──── Postgres, 61 MB of data
  │
Render ─── the API + demo UI, deployed from GitHub
  │
HF ─────── the model, called over the network
```

---

## Step 1 — the database (Neon)

1. Sign up at **<https://neon.tech>** with GitHub.
2. **Create project** → name it `text2sql` → any region.
3. On the dashboard, copy the **connection string**. It looks like:

   ```
   postgresql://neondb_owner:AbC123xyz@ep-cool-name-12345.us-east-2.aws.neon.tech/neondb?sslmode=require
                └── user ──┘ └ password ┘ └──────────── host ───────────────────┘ └ db ┘
   ```

4. **Send that string to your assistant, or run the seeding yourself:**

   ```powershell
   $env:PGHOST="ep-cool-name-12345.us-east-2.aws.neon.tech"
   $env:PGDATABASE="neondb"
   $env:APP_DB_USER="neondb_owner"
   $env:APP_DB_PASSWORD="AbC123xyz"
   $env:PGSSLMODE="require"

   env\Scripts\python.exe scripts/load_schema.py
   env\Scripts\python.exe scripts/generate_data.py
   ```

   Ten to fifteen minutes over the network. Verify:

   ```powershell
   env\Scripts\python.exe -m pytest tests/test_database_integrity.py -q
   ```

> Neon's free database sleeps after 5 minutes idle. The first request after a
> sleep takes a few seconds to wake it — worth knowing before a demo, not
> during one.

---

## Step 2 — the app (Render)

1. Sign up at **<https://render.com>** with GitHub.
2. **New → Blueprint**.
3. Pick the repository. Render finds `render.yaml` and configures everything
   itself — no form filling.
4. It then asks for the five secrets. Fill them from the Neon string:

   | key | value |
   |---|---|
   | `PGHOST` | `ep-cool-name-12345.us-east-2.aws.neon.tech` |
   | `PGDATABASE` | `neondb` |
   | `APP_DB_USER` | `neondb_owner` |
   | `APP_DB_PASSWORD` | the password from the string |
   | `HF_TOKEN` | a Hugging Face token with **inference** permission |

5. **Apply**. First build takes ~5 minutes.

---

## Step 3 — check it

```bash
curl https://text2sql-api-XXXX.onrender.com/health
```

`"status":"ok"` with `"database":true` means both halves are talking. Then open
the URL in a browser for the demo UI.

---

## What to expect

**The free tier sleeps after 15 minutes idle, so the first visitor waits
~50 seconds.** That is the one real downside, and it matters if you are sending
the link to someone. Lead with the Hugging Face Space instead — it is static,
always instant — and keep this URL for showing live SQL execution.

**It serves the base model, not the fine-tuned one** (10.82 % rather than
50.99 % strict execution accuracy). The adapter needs a GPU; the free tier has
none. Say so rather than let someone assume otherwise: the deployment
demonstrates the *pipeline*, the benchmark demonstrates the *fine-tuning*.

To serve the fine-tuned model, run `MODEL_BACKEND=local` on a CUDA host, or
put a vLLM server with the LoRA adapter behind the API.

**The Hugging Face Space calls this API from the browser.** The API grants
CORS to `*.hf.space` origins only (see `src/api/main.py`); add any other
front-end origin as a comma-separated `CORS_ORIGINS` environment variable on
Render. Never `*` — the API is unauthenticated, and an open grant would let any
page spend this deployment's inference quota from its visitors' browsers.

---

## Costs

| | |
|---|---|
| Neon free tier | 0.5 GB — this database is 61 MB |
| Render free web service | 750 hours/month, sleeps when idle |
| Hugging Face Inference | free monthly credits; a query costs a fraction of a cent |

Nothing here needs a card. The one thing that would is serving the fine-tuned
model on a GPU.
