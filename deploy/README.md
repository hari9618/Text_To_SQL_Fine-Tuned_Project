# Deploying the live demo

Phase 13/14. Gets the service onto a public URL you can open in an interview.

Two pieces have to live somewhere:

| piece | where | cost |
|---|---|---|
| PostgreSQL, ~328k rows | **Neon** free tier | free |
| The API + demo UI | **Hugging Face Space** (Docker) | free |

Both have free tiers that comfortably fit this project, and you already have a
Hugging Face account.

---

## Why this split

The API container is deliberately tiny — 26 files, 171 KB of source, no torch —
because the model is called over the network rather than loaded in-process. So
the only heavy thing is the database, and that belongs in a managed service
rather than in the same container as the web server.

**The database is regenerated remotely, not copied.** `scripts/generate_data.py`
is seeded, so pointing it at Neon reproduces the byte-identical database rather
than shipping a dump around. That also means the schema fingerprint
(`d03619e711661bc5`) still matches, which is what keeps the deployed service
comparable to the benchmark.

---

## Step 1 — Postgres on Neon (~5 min)

1. Sign up at <https://neon.tech> (GitHub login works).
2. Create a project. Any region; pick the one nearest you.
3. Copy the **connection string**. It looks like:
   ```
   postgresql://USER:PASSWORD@ep-xxx-yyy.region.aws.neon.tech/neondb?sslmode=require
   ```
4. Load the schema and generate the data **from your laptop**, pointed at Neon:

   ```powershell
   $env:PGHOST      = "ep-xxx-yyy.region.aws.neon.tech"
   $env:PGPORT      = "5432"
   $env:PGDATABASE  = "neondb"
   $env:APP_DB_USER = "<user from the connection string>"
   $env:APP_DB_PASSWORD = "<password from the connection string>"
   $env:PGSSLMODE   = "require"

   env\Scripts\python.exe scripts/load_schema.py
   env\Scripts\python.exe scripts/generate_data.py
   ```

   Ten to fifteen minutes over the network. Verify it landed:

   ```powershell
   env\Scripts\python.exe -m pytest tests/test_database_integrity.py -q
   ```

> Neon's free tier suspends an idle database after five minutes. The first
> request after a suspension takes a few seconds to wake it — worth knowing
> before you demo, not during.

---

## Step 2 — the Space (~5 min)

1. <https://huggingface.co/new-space> → SDK **Docker** → **Blank** → visibility
   **Public**.
2. Clone it and copy this project in:

   ```bash
   git clone https://huggingface.co/spaces/<you>/text2sql
   cd text2sql
   # copy from the project root:
   #   src/  requirements.txt  deploy/Dockerfile -> Dockerfile  deploy/README_SPACE.md -> README.md
   git add -A && git commit -m "Enterprise Text-to-SQL" && git push
   ```

3. In the Space: **Settings → Variables and secrets**, add these as **secrets**
   (never as plain variables — a variable is visible to anyone who opens the
   Space):

   | secret | value |
   |---|---|
   | `PGHOST` | `ep-xxx-yyy.region.aws.neon.tech` |
   | `PGDATABASE` | `neondb` |
   | `APP_DB_USER` | Neon user |
   | `APP_DB_PASSWORD` | Neon password |
   | `PGSSLMODE` | `require` |
   | `HF_TOKEN` | a token with *Inference Providers* permission |
   | `MODEL_BACKEND` | `hf` |
   | `HF_PROVIDER` | `featherless-ai` |

The Space builds and comes up at
`https://huggingface.co/spaces/<you>/text2sql`.

---

## Step 3 — check it

```bash
curl https://<you>-text2sql.hf.space/health
```

`status: ok` and `database: true` means both halves are talking. Then open the
Space and ask something.

---

## Which model the demo serves

**The public demo serves the base model, not the fine-tuned one** — 43.71 %
rather than 70.86 % strict execution accuracy. It renders **prompt v2**
(`PROMPT_VERSION`, default `v2`), whose business glossary is worth +9.47 pp and
needs no GPU. The adapter needs a GPU, and both
the free Space tier and this project's laptop have none.

That is worth saying out loud in an interview rather than hiding: the deployed
system demonstrates the *pipeline* — schema context, validation, execution,
self-correction — while the fine-tuning result is demonstrated by the
**benchmark**, which was measured properly on 453 held-out questions with the
adapter loaded on a GPU.

To serve the fine-tuned model you would either:

- upgrade the Space to a GPU tier, set `MODEL_BACKEND=local`, and mount the
  adapter; or
- run a vLLM server with the LoRA adapter loaded and point the API at it. This
  is the right answer at any real request volume anyway — one copy of the
  weights serving many API replicas, instead of every replica holding its own
  5.7 GB.

---

## Security before you make it public

The service executes model-written SQL. The layers that make that acceptable
are in `docker/README.md`, and all of them still apply here. Two things change
once the URL is public:

1. **Anyone who can ask a question can read any row the role can read.** The
   data is synthetic, so this is fine for a demo — it would not be with real
   data.
2. **Each request costs a model call.** There is no rate limiting; a Space that
   gets traffic will burn Inference Provider credits. Keep the Space private
   until you need it, or put a token in front.

Neither is a flaw in the design — both are stated as out of scope in
`docker/README.md` — but a public URL is the moment they stop being theoretical.
