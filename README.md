# Enterprise Text-to-SQL: Fine-Tuning and Self-Correction

Convert natural-language business questions into **executable, validated SQL**
against an enterprise-style PostgreSQL database — and *measure* whether
fine-tuning an open-source LLM actually improves it.

> **Headline result: fine-tuning took strict execution accuracy from
> 10.82 % to 50.99 % on 453 unseen questions.**
> Same prompt, same schema, same database, same executor, same metrics. Only
> the weights changed.

The interesting part is not the number. It is *where the number came from* —
most of the gain turns out to be the model learning which columns to return,
not becoming dramatically better at SQL logic. That distinction is only
visible because the evaluation was built to expose it. See
[Reading the result honestly](#reading-the-result-honestly).

---

## Contents

- [The problem](#the-problem)
- [Pipeline](#pipeline)
- [Results](#results)
- [Reading the result honestly](#reading-the-result-honestly)
- [How it is evaluated](#how-it-is-evaluated)
- [The API](#the-api)
- [Getting started](#getting-started)
- [Reproducing the experiment](#reproducing-the-experiment)
- [Repository layout](#repository-layout)
- [What went wrong](#what-went-wrong)
- [Limitations](#limitations)

---

## The problem

A business user asks:

> *"Which five customers spent the most last year?"*

Answering that requires knowing which of twelve tables matter, how they join,
and how to express the aggregation in correct PostgreSQL. A general-purpose LLM
handed the whole schema hallucinates columns, picks the wrong join path, and
writes SQL that fails at execution — or worse, SQL that runs and returns
plausible but wrong rows.

Three ideas attack that, each measured independently rather than assumed:

1. **Schema retrieval** — show the model only the relevant tables.
2. **Fine-tuning (QLoRA)** — teach the model to use that schema reliably.
3. **Self-correction** — when SQL fails, feed the database's error back and repair it.

Two of the three helped. One did not. Both outcomes are reported.

---

## Pipeline

```
Natural-language question
        |
        v
Schema context  ->  full schema, or a retrieved subset
        |
        v
Qwen3-8B  (+ QLoRA adapter)
        |
        v
Generated SQL
        |
        v
Static validation   ->  parseable? single statement? read-only? real tables?
        |
        v
PostgreSQL          ->  read-only session, 30 s statement timeout
        |
   +----+----------------------+
   |                           |
success                     failure
   |                           |
   v                           v
Result                    Repair: error fed back to the model
                               |
                               v
                     corrected SQL -> PostgreSQL -> Result
```

Exposed as a FastAPI service (`POST /query`).

---

## Results

453 unseen test questions. Execution-based scoring throughout.

| metric | 1. Base | 2. Base + retrieval | 3. **Fine-tuned** |
|---|---:|---:|---:|
| **strict execution accuracy** | 10.82 % | 9.27 % | **50.99 %** |
| projection-tolerant accuracy | 45.92 % | 43.93 % | 50.99 % |
| executable SQL | 98.90 % | 92.27 % | 95.81 % |
| schema hallucination | 0.66 % | 3.31 % | 1.99 % |
| syntax errors | 0.00 % | 0.00 % | 0.22 % |

### By difficulty

| tier | n | Base | Fine-tuned | Δ |
|---|---:|---:|---:|---:|
| easy | 126 | 0.0 % | 23.0 % | +23.0 |
| medium | 63 | 52.4 % | 52.4 % | **+0.0** |
| hard | 144 | 3.5 % | **84.7 %** | **+81.2** |
| enterprise | 84 | 13.1 % | 52.4 % | +39.3 |

### Schema retrieval made things worse

Retrieval was tuned on the **validation** split (k=4 seed tables, foreign-key
expansion 2) and measured once on test: strict accuracy fell from 10.82 % to
9.27 %, and schema hallucination rose five-fold, from 0.66 % to 3.31 %.

The mechanism is visible in the numbers: with twelve tables the entire schema
is only 5,455 characters, so retrieval saves little context while occasionally
hiding a table the query needs — and a model that cannot see a table invents
one. **Retrieval is a fix for a problem this database does not have.** It would
very likely pay off at 200 tables; at 12 it costs accuracy.

That is why the fine-tuning dataset was built on the **full** schema.

### Not yet measured

| # | configuration | status |
|---|---|---|
| 4 | Fine-tuned + schema retrieval | **not measured** — needs a GPU run |
| 5 | Fine-tuned + repair | **not measured** — needs a GPU run |

Everything for both is built, tested and staged: eval packs export, the
notebooks are verified against the real chat template, and the scorers are
validated with synthetic inputs. They are listed here rather than omitted,
because a gap in an ablation table invites the reader to assume the missing row
would have agreed with the others.

Run `scripts/ablation_report.py` to regenerate
[`experiments/ABLATION.md`](experiments/ABLATION.md) as configurations land.

---

## Reading the result honestly

A 4.7× improvement deserves scrutiny rather than a victory lap.

**Most of the gain is column selection, not SQL reasoning.**

| | Base | Fine-tuned |
|---|---:|---:|
| right rows, **wrong columns** | 159 | **0** |
| genuinely wrong rows | 240 | 203 |

The benchmark's largest failure mode was the model returning correct data under
a different projection — *"show orders in 2023"* never says which columns to
return, and the base model guessed differently from the gold query 159 times.
Fine-tuning eliminated that entirely. Genuinely wrong answers fell far less
(240 → 203).

This is why the harness tracks **projection-tolerant accuracy** alongside strict
accuracy: it separates "wrong columns" from "wrong rows". Strict accuracy alone
would have made fine-tuning look like a reasoning breakthrough. Projection
tolerance alone (45.92 % → 50.99 %) would have made it look marginal. Both
together tell the truth:

- Learning the house style for projections: **large, real, and somewhat
  particular to this benchmark.**
- Genuine reasoning improvement: **smaller, but concentrated where it matters
  most** — the `hard` tier (multi-table joins, subqueries) went from 3.5 % to
  84.7 %.

**Fine-tuning also made some things worse.** Executable SQL fell 3.09 points and
hallucination rose 1.33: 19 queries now fail where 5 did before. Final training
loss of 0.006 against an evaluation loss of 0.335 says the adapter fit 2,133
examples hard, and a tightly-fit model is more willing to emit a confident wrong
identifier. Those 19 failures are exactly what the repair loop targets.

**`medium` did not move at all** (52.4 % → 52.4 %). Unexplained, and worth
investigating rather than glossing over.

---

## How it is evaluated

**Execution-based, never string matching.** These are textually different and
semantically identical, and both count as correct:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

Every prediction is executed against PostgreSQL and its result set fingerprinted
(MD5 over stringified, sorted rows). Order-sensitive fingerprints are compared
only where the gold query has an `ORDER BY`.

Tracked: execution accuracy, projection-tolerant accuracy, syntax error rate,
schema hallucination rate, repair success rate, latency, token usage. Eleven
mutually exclusive outcome classes, so every failure lands in exactly one bucket.

### Guardrails that keep the numbers meaningful

| guardrail | what it prevents |
|---|---|
| **Leakage-free splits by template equivalence group**, never by row | a paraphrase of a test question appearing in training |
| **Prompt fingerprint** `8288e41a496531a9` asserted at every stage | measuring prompt engineering instead of fine-tuning |
| **Schema fingerprint** `d03619e711661bc5` asserted at every stage | comparing runs that saw different databases |
| **Dataset content hashes** for train/validation | training on data other than what was audited |
| **The GPU host never receives gold SQL** — asserted at both ends | a "model" that copies the answer |
| **Oracle self-test**: gold SQL through the scorer must return 100 % | a broken scorer silently deflating every result |
| **`DATA_AS_OF` constant** instead of `NOW()` | a benchmark whose answers change overnight |
| **Retrieval tuned on validation, measured once on test** | hyperparameters fitted to the test set |

The oracle self-test is run before every scoring pass. A perfect model must
score 100 %; if it does not, the fault is in the harness and every real number
would be wrong in the same way.

---

## The API

```powershell
env\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question":"How many customers are based in India?"}'
```

```json
{
  "ok": true,
  "sql": "SELECT COUNT(*) FROM customers WHERE country = 'India'",
  "columns": ["count"],
  "rows": [["1023"]],
  "repaired": false,
  "timings": {"generation_ms": 9950.4, "execution_ms": 21.8, "total_ms": 9977.5}
}
```

Endpoints: `POST /query`, `GET /schema`, `GET /health`, `GET /docs`.

### Which model is served

**This decides the accuracy of every answer:**

| `MODEL_BACKEND` | model | strict accuracy |
|---|---|---:|
| `hf` *(default)* | base Qwen3-8B, served remotely | 10.82 % |
| `local` | base + the LoRA adapter, 4-bit | 50.99 % |
| `stub` | canned SQL | tests only |

The default is the **base** model because the development laptop has no GPU.
Serving the fine-tuned adapter means running `local` on a CUDA host.

### Safety

The service executes text a language model wrote. Three layers hold regardless
of how the model behaves:

1. **The database role is not a superuser.** It owns one database and nothing else.
2. **Every session is read-only**, with a statement timeout, applied on pool
   checkout rather than once at creation.
3. **SQL is statically rejected** unless it is a single read-only statement over
   tables that exist — before it reaches the database.

Plus: rows capped server-side, container runs as non-root, no traceback or
connection string is ever returned to a caller, and no secret exists in any
image layer. Full write-up in [`docker/README.md`](docker/README.md).

---

## Getting started

Requires **Python 3.11** and a local **PostgreSQL 18** server.

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env      # then fill it in; .env is git-ignored

python scripts/init_db.py        # database + non-superuser application role
python scripts/load_schema.py    # apply database/schema.sql
python scripts/generate_data.py  # ~328,000 rows, fixed seed, about a minute

python -m pytest tests/ -q       # 114 tests
```

### Database

12 tables across four domains — sales, catalogue, logistics, HR. See
[`database/ERD.md`](database/ERD.md).

| Table | Rows | | Table | Rows |
|---|---:|---|---|---:|
| customers | 12,000 | | order_items | 154,069 |
| orders | 55,000 | | payments | 57,088 |
| products | 2,500 | | shipments | 40,850 |
| categories | 120 | | inventory | 6,266 |
| suppliers | 250 | | employees | 500 |
| warehouses | 15 | | departments | 12 |

All synthetic, no real personal data, fixed random seed — so the database is
reproducible, which is what makes the base and fine-tuned runs comparable.

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
docker compose -f docker/docker-compose.yml --profile tools run --rm seed
```

> **These images have not been built.** Docker is not installed on the
> development machine. The compose file parses, every pin resolves, and the
> healthcheck command was tested against a live and a dead port — but
> `docker build` has not run. See [`docker/README.md`](docker/README.md).

---

## Reproducing the experiment

The benchmark and the fine-tune are frozen artefacts, not moving targets.

| artefact | fingerprint |
|---|---|
| prompt template | `8288e41a496531a9` |
| rendered schema | `d03619e711661bc5` |
| SFT train split (2,133) | `7804d4386c9b1004` |
| SFT validation split (459) | `43a469080798f5aa` |
| test split (453) | `ec7ddcae4f9d90d4` |

```powershell
# harness self-test - no model calls, must score 100 %
env\Scripts\python.exe scripts/run_baseline.py --oracle --sample 60 --tag oracle

# the frozen baseline
env\Scripts\python.exe scripts/run_baseline.py --provider featherless-ai --workers 6

# export what a GPU host needs (questions only; gold SQL never leaves)
env\Scripts\python.exe scripts/export_eval_pack.py
env\Scripts\python.exe scripts/export_eval_pack.py --retrieval keyword

# score predictions generated on the GPU host
env\Scripts\python.exe scripts/score_finetuned.py --predictions predictions.jsonl
env\Scripts\python.exe scripts/score_repair.py    --repairs repairs.jsonl

# assemble the ablation
env\Scripts\python.exe scripts/ablation_report.py
```

### Fine-tuning

QLoRA on a free Kaggle T4, **4 h 53 m for one epoch**
(`training/kaggle/kaggle_qlora_qwen3_8b.ipynb`).

| | |
|---|---|
| base | `Qwen/Qwen3-8B` @ `b968826d9c46`, 4-bit NF4 + double quantisation |
| adapter | LoRA r=16, α=32, dropout 0.05, all 7 attention and MLP projections |
| trainable | 43,646,976 of 4,761,498,624 (**0.917 %**) |
| loss | **completion-only** — the prompt is masked out |
| supervised token share | **3.01 %** |
| train loss | 4.10 → 0.006 |
| eval loss | 0.475 → 0.340 → 0.335 |

**Completion-only loss is the detail that matters most.** Each example is ~1,490
prompt tokens and ~46 completion tokens, and the prompt is ~97 % schema,
byte-identical across all 2,133 examples. Training on the full sequence would
send ~97 % of the gradient into memorising a schema the model is *handed* at
inference — and the loss curve would look excellent throughout, because
predicting a constant is easy.

---

## Repository layout

```
src/
├── api/          FastAPI service (Phase 12)
├── evaluation/   benchmark runner, metrics, reporting
├── model/        prompts, schema context, model backends
├── retrieval/    lexical schema retriever (Phase 6)
└── sql/          validator, executor, repair loop
database/         schema.sql, ERD.md, seed.sql
dataset/          generation, validation, SFT formatting
scripts/          operational entry points
training/         QLoRA training code and Kaggle notebooks
evaluation/       harness assets
experiments/      frozen results: baseline, retrieval, finetuned, repair
models/finetuned/ the LoRA adapter (weights git-ignored)
docker/           Dockerfile, compose, deployment notes
tests/            114 tests
```

---

## What went wrong

Every one of these cost real time and is documented in `CLAUDE.md` §13 so it is
not re-derived.

**The chat-template trap — one wasted GPU run.** Qwen3's template renders a
finished assistant turn as `<|im_start|>assistant\n<think>\n\n</think>\n\nSELECT
...`, so every training target was preceded by an empty think block. At
inference, `apply_chat_template(add_generation_prompt=True)` *without*
`enable_thinking=False` stops at `<|im_start|>assistant\n` and leaves the model
to invent a think block it never saw as a starting point. The first Phase 10 run
returned reasoning prose, stray `</think>` tags and junk lead tokens
(`handgun`, `girl`, `apiro`). **The guard that missed it checked only that the
inference prompt was a *prefix* of the training rendering — it was; the entire
problem lived in the gap after it.** The check now asserts that generation
resumes at the assistant content itself.

**A model that loaded in full precision, silently.** A signature-filtering
helper stripped `quantization_config` from `from_pretrained` because that
function is `(cls, path, *args, **kwargs)` and has no named parameters to match.
The 8B model loaded at 16 GB into 12 GB of RAM and the runtime was killed with
no error message. There is now an explicit `Linear4bit` assertion.

**Latency contaminated by retry backoff.** The first baseline reported a mean of
22,124 ms against a median of 1,376 ms. That was HuggingFace's rate limiter, not
the model. Timing now restarts per attempt.

**Tuning on the test set — caught and undone.** Retrieval hyperparameters were
first swept on `test.jsonl`. The sweep was redone on validation, and test was
measured once.

**An HTTP 402 storm misattributed to concurrency.** It was the provider:
`provider=auto` routes to one that fails about half the time. Measured, not
guessed — `featherless-ai` succeeds 6/6, and 6 workers are 6× faster than 1.

---

## Limitations

- **Configurations 4 and 5 are unmeasured.** Both need a GPU this machine does
  not have. Everything for them is built and verified.
- **One epoch, one seed, one run.** No confidence intervals, no seed variance.
  The 40-point gap is far larger than plausible run-to-run noise, but that is an
  argument, not a measurement.
- **Twelve tables is a small enterprise schema.** The retrieval result almost
  certainly reverses at 200 tables.
- **The benchmark is template-generated.** Leakage-free by construction, but
  narrower than real user questions.
- **`medium` difficulty did not improve** and is not yet explained.
- **The Docker images have never been built.**
- **The API's default backend is the base model**, not the fine-tuned one.
- **No authentication or rate limiting.** A caller who can ask questions can
  read any row the database role can read.

---

## License

Not yet specified.
