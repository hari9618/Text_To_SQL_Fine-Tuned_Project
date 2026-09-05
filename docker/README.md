# Docker deployment

Phase 13.

Two images, because they have different jobs and different risk profiles:

| image | contains | why separate |
|---|---|---|
| `docker/Dockerfile` | `src/` only | serves public requests; nothing it does not need should be in it |
| `docker/Dockerfile.tools` | `src/`, `scripts/`, `dataset/`, `database/`, `tests/` | one-shot data generation and maintenance |

Neither contains `torch`. The default backend calls a remotely served model, so
the API image is roughly 200 MB rather than several gigabytes. Serving the
fine-tuned adapter in-process is a different image — see the bottom of this
file.

---

## Honest status

**These images have not been built.** Docker is not installed on the
development machine (Windows 11, 8 GB RAM, no GPU), so what has been verified is:

| | |
|---|---|
| `docker-compose.yml` parses, 3 services, 1 volume | ✅ verified |
| Every pin in `requirements.txt` matches an installed, working package | ✅ verified |
| `uvicorn src.api.main:app` starts and serves `/health`, `/query`, `/schema` | ✅ verified locally |
| The `HEALTHCHECK` command returns 0 when healthy, 1 when the port is dead | ✅ verified locally |
| `docker build` succeeds | ❌ **not run** |
| The composed stack starts | ❌ **not run** |

The Dockerfiles are written against a service that demonstrably runs; they have
not themselves been executed. Treat the build as unverified until it is.

---

## Running it

Everything runs from the **project root**, not from `docker/`.

```bash
# 1. Credentials come from the git-ignored .env, same file local dev uses.
cp .env.example .env      # then fill it in

# 2. Start the database and the API.
docker compose -f docker/docker-compose.yml up --build

# 3. The database starts with the schema applied but EMPTY.
#    Generating ~250k rows takes minutes, so it is a deliberate step.
docker compose -f docker/docker-compose.yml --profile tools run --rm seed
```

Then:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question":"How many customers are based in India?"}'
```

Interactive docs: <http://localhost:8000/docs>

---

## Configuration

All through environment variables; nothing secret is in any committed file.

| variable | default | meaning |
|---|---|---|
| `APP_DB_USER` | `texttosql_app` | database role — **not** a superuser |
| `APP_DB_PASSWORD` | *(required)* | compose refuses to start without it |
| `PGDATABASE` | `enterprise_sql` | database name |
| `MODEL_BACKEND` | `hf` | `hf`, `local`, or `stub` |
| `HF_TOKEN` | *(empty)* | required when `MODEL_BACKEND=hf` |
| `HF_PROVIDER` | `featherless-ai` | measured: `auto` routes to a provider that returns HTTP 402 about half the time |
| `SQL_STATEMENT_TIMEOUT_MS` | `30000` | per-statement ceiling |
| `DB_POOL_MAX` | `4` | pool size |
| `API_PORT` | `8000` | host port |

---

## Which model is actually being served

**This matters more than any other setting**, because it decides the accuracy
of every answer:

| `MODEL_BACKEND` | model | strict execution accuracy |
|---|---|---:|
| `hf` *(default)* | base Qwen3-8B, served remotely | **10.82 %** |
| `local` | base + Phase 9 LoRA adapter, 4-bit | **50.99 %** |
| `stub` | canned SQL, no model | n/a — tests only |

The default is the *base* model because the development machine has no GPU.
Deploying the fine-tuned model means running `local` on a CUDA host, which
needs a different image:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# ... python, then:
#   pip install torch transformers peft bitsandbytes accelerate
# and mount the adapter, which is deliberately not baked into any image:
#   volumes: [ "../models/finetuned/final_adapter:/adapters/text2sql:ro" ]
#   environment: { MODEL_BACKEND: local, ADAPTER_DIR: /adapters/text2sql }
#   deploy: { resources: { reservations: { devices: [{ capabilities: [gpu] }] }}}
```

At real request volume the better answer is a dedicated inference server
(vLLM with LoRA enabled) with the API calling it over HTTP, rather than every
API replica holding its own 5.7 GB copy of the weights.

---

## Security posture

Not incidental to this project: the service **executes text a language model
wrote** against a production-shaped database.

1. **The role is not a superuser.** `texttosql_app` owns its own database and
   nothing else. A model cannot be prompted into rights it does not have.
2. **Every session is read-only.** `default_transaction_read_only` is set on
   each pooled connection at checkout, not once at creation, so a reset session
   cannot silently lose the guarantee.
3. **Every statement is time-limited.** 30 s by default; a runaway join becomes
   a recorded failure rather than a hung worker.
4. **SQL is statically rejected** unless it is a single read-only statement
   over tables that exist — before it reaches the database.
5. **Rows are capped** server-side at 1000, regardless of what the caller asks
   for.
6. **The container does not run as root** (uid 10001).
7. **The database is not published to the host.** The API reaches it over the
   compose network.
8. **No secret is in any image layer.** `.env` is in `.dockerignore`;
   credentials arrive as environment variables. `docker history` shows nothing.
9. **No traceback, path or connection string is ever returned to a caller** —
   see the exception handler in `src/api/main.py`.

Layers 1–3 hold even if the model is adversarially prompted, because none of
them depend on the model behaving.

---

## What is deliberately not here

- **No TLS.** Terminate at a load balancer or ingress; an app container issuing
  its own certificates is an anti-pattern.
- **No auth.** The service has no notion of users. Anything internet-facing
  needs an API gateway in front — and note that a caller who can ask questions
  can read any row the role can read.
- **No rate limiting.** Each request costs a model call; unbounded traffic is
  an unbounded bill. Belongs at the gateway.
- **No data in the image.** `models/`, `experiments/` and generated data are in
  `.dockerignore`. Data is seeded into a volume, deliberately.
