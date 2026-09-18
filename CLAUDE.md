# CLAUDE.md

Instructions for any AI assistant working on this repository.
**These rules override default assistant behaviour.**

---

## 1. What this project is

**Enterprise Text-to-SQL Fine-Tuning and Self-Correction System.**

A production-style system that converts natural-language business questions into
executable SQL against an enterprise-style PostgreSQL database, then *proves*
whether fine-tuning actually helped.

This is an **educational portfolio project**. The goal is not merely a working
system. The goal is to understand *why* each component exists, *what* problem it
solves, *how* it is evaluated, and *whether* fine-tuning measurably improves it.

---

## 2. Target architecture

```
User question (natural language)
        |
        v
Schema retrieval  ->  relevant tables / columns / relationships only
        |
        v
Fine-tuned open-source LLM
        |
        v
Generated SQL
        |
        v
SQL validation (syntax + schema grounding)
        |
        v
PostgreSQL execution
        |
   +----+--------------------+
   |                         |
 success                  failure
   |                         |
   v                         v
Result                  SQL repair
                             |
                             v
                       corrected SQL -> PostgreSQL -> Result
```

Eventually exposed through a FastAPI service.

**Do not redesign this architecture unless explicitly asked.**

---

## 3. Why schema retrieval and fine-tuning are separate

They solve different problems and both are required:

| Component        | Question it answers                                          |
| ---------------- | ------------------------------------------------------------ |
| Schema retrieval | *What database information should the model see?*            |
| Fine-tuning      | *How should the model use that information to write SQL?*    |

An enterprise database has many tables. Sending the entire schema on every
request wastes context, increases latency, and increases hallucination. Schema
retrieval selects only the relevant subset.

---

## 4. Phases

Work strictly in order. **Do not implement future phases prematurely.**

| Phase  | Description                                       | Status      |
| ------ | ------------------------------------------------- | ----------- |
| 0      | Environment setup                                 | **complete** |
| 1      | PostgreSQL setup                                  | **complete** |
| 2      | Design enterprise-style database schema           | **complete** |
| 3      | Generate realistic synthetic database data        | **complete** |
| 4      | Create Text-to-SQL benchmark dataset              | **complete** |
| 5      | Implement base-model baseline                     | **complete** |
| 6      | Implement schema retrieval                        | **complete** |
| 7      | Prepare fine-tuning dataset                       | **complete** |
| 8      | Understand and implement LoRA / QLoRA             | **complete** |
| 9      | Fine-tune on cloud GPU                            | **complete** |
| 10     | Evaluate fine-tuned model                         | **complete** |
| 11     | SQL error detection and repair                    | **complete** |
| 12     | Production-style API                              | **complete** |
| 13     | Docker / deployment                               | **complete** (build unrun) |
| 14     | Final benchmark, ablation study, documentation    | **complete** (5/5 configs) |

Update this table as phases complete.

**All five ablation configurations are measured.** See the results table
below and `experiments/ABLATION.md`.

--- | --- | --- | --- | --- |
| 5 fine-tuned + repair | `text2sql-repairpack.zip` | `kaggle_repair_finetuned.ipynb` | `repairs.jsonl` | `scripts/score_repair.py` |

Then `scripts/ablation_report.py` regenerates `experiments/ABLATION.md` with
all five rows.

---

## 5. Critical ML experiment principle

**A baseline must exist before fine-tuning.** The evaluation pipeline compares:

1. Base model
2. Base model + schema retrieval
3. Fine-tuned model
4. Fine-tuned model + schema retrieval
5. Fine-tuned model + schema retrieval + SQL repair

**Never claim fine-tuning improves performance without measuring it.**

---

## 6. Evaluation rules

Primary metric is **execution-based**, not string matching. These are different
strings but semantically identical, and must both count as correct:

```sql
SELECT COUNT(*)          FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

Tracked metrics:

- execution accuracy (does the result set match?)
- syntax error rate
- schema hallucination rate (invented tables/columns)
- SQL repair success rate
- latency
- token usage / cost where applicable

The test set must contain **unseen** questions. Guard against test leakage when
building splits.

Difficulty tiers:

- **easy** — SELECT, WHERE, ORDER BY, LIMIT
- **medium** — GROUP BY, HAVING, COUNT, SUM, AVG
- **hard** — JOIN, multiple JOINs, subqueries, CTEs
- **very hard** — window functions, nested aggregation, date comparisons, conditional aggregation
- **enterprise** — ambiguous terminology, multiple possible joins, NULL handling, business rules, cross-domain queries

---

## 7. Model strategy

- **Baseline model (decided): `Qwen3-8B-Instruct`.** This is the Phase 5
  baseline and the model fine-tuned in Phase 9, so base and fine-tuned numbers
  compare like for like. The *Instruct* variant, not the base checkpoint —
  Text-to-SQL is an instruction-following task, and comparing a fine-tuned
  adapter against a non-instruct base would overstate the gain.
- Training method: **QLoRA / LoRA** (parameter-efficient), never full fine-tuning
- **Served remotely, not downloaded.** The 8 GB / no-GPU laptop cannot host an
  8B model, so inference goes through Hugging Face Inference Providers
  (`Qwen/Qwen3-8B`, revision `b968826d9c46`) while PostgreSQL stays local. No
  torch/transformers installed. Weights are downloaded only for Phase 9
  fine-tuning, on a cloud GPU.
- **Frozen baseline (Phase 5, approved):** strict execution accuracy 10.82 %,
  projection-tolerant 45.92 %, executable SQL 98.90 %, schema hallucination
  0.66 %. See `experiments/baseline/OFFICIAL_BASELINE.md`. Never recompute or
  adjust these; re-run deliberately if the configuration changes.

### Fine-tuning dataset (Phase 7, ready — training NOT started)

Built by `scripts/prepare_sft_dataset.py` into `dataset/sft/`. Chat format,
three turns, gold SQL only in the assistant turn.

| | train | validation |
| --- | ---: | ---: |
| records | 2,133 | 459 |
| content hash | `7804d4386c9b1004` | `43a469080798f5aa` |

- Prompt is **imported** from `src/model/prompt.py` (fingerprint
  `8288e41a496531a9`) and the full schema (`d03619e711661bc5`)
  is rendered identically to the frozen baseline. Changing either invalidates
  both the baseline and this dataset — deliberately, so they cannot drift apart.
- **Full schema, not retrieved:** Phase 6 measured retrieval on this 12-table
  database and strict accuracy *fell* from 10.82 % to 9.27 %.
- Evaluation metadata (`template_id`, `slot_values`, `referenced_tables`,
  execution fingerprints, split labels, difficulty) is carried in a `meta`
  object **outside** `messages` and never reaches the prompt.
- `dataset/test/test.jsonl` (sha `ec7ddcae4f9d90d4`) is read only for
  contamination checking. Never trained on, never written.
- Suggested `max_seq_len` for Phase 8: **2048**
  (longest example 1602 tokens).

### Fine-tuned adapter (Phase 9, complete)

Trained on Kaggle, **1 epoch, 4 h 53 m** on a Tesla T4 (notebook
`training/kaggle/kaggle_qlora_qwen3_8b.ipynb`). Adapter lives in
`models/finetuned/` — weights are git-ignored, configs are committed.

| | |
| --- | --- |
| base | `Qwen/Qwen3-8B` @ `b968826d9c46`, 4-bit NF4 + double quant |
| adapter | LoRA r=16, alpha=32, dropout 0.05, all 7 attn+MLP projections |
| trainable | 43,646,976 of 4,761,498,624 (0.917 %) |
| supervised token share | 3.01 % — prompt masked, loss on SQL only |
| optimiser steps | 134 (effective batch 16: T4 x2 halved the expected 266) |
| train loss | 4.10 -> 0.006 |
| eval loss | 0.475 (step 50) -> 0.340 (step 100) -> **0.335** (final) |
| truncated examples | 0 |

Resolved package versions (recorded, not pinned — pinning is what broke this
on Colab): torch 2.10.0+cu128, transformers 5.16.1, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.14.0, trl 1.12.0.

**Loss is not the result.** Final train loss of 0.006 against an eval loss of
0.335 means the adapter fits the 2,133 training templates hard. Whether that
transfers is a Phase 10 question, answered by execution accuracy on the 453
unseen test examples — not by this curve.

### Phase 10 evaluation (split across two machines)

Generation needs a GPU; scoring needs the local PostgreSQL. So Phase 10 runs in
three steps and the halves never meet:

```
scripts/export_eval_pack.py  ->  text2sql-evalpack.zip
                                        |
                       training/kaggle/kaggle_generate_finetuned.ipynb
                                        |
                                predictions.jsonl
                                        |
                          scripts/score_finetuned.py  ->  verdict
```

* **The GPU host never sees gold SQL.** The eval pack carries `{id, question}`
  only, and both the exporter and the notebook assert that no other field is
  present. The model cannot copy an answer it was never given.
* **`extract_sql` runs locally**, not on the GPU host, using the baseline's own
  function. A second copy on the remote side could drift and silently move the
  score.
* **Fingerprints are checked at both ends.** The notebook recomputes the prompt
  (`8288e41a496531a9`) and schema (`d03619e711661bc5`) hashes from the shipped
  files and aborts on mismatch; `score_finetuned.py` re-checks the header the
  notebook wrote. A run with a different prompt or a drifted database cannot be
  compared to the frozen baseline, so it is refused rather than reported.
* **Decoding is greedy, `max_new_tokens=512`** — the same as the baseline's
  `temperature=0, max_tokens=512`. Sampling would make the delta partly noise.

Harness self-test (same idea as `run_baseline.py --oracle`): feed gold SQL in
as synthetic predictions and the scorer must return 100 %.

```powershell
env\Scripts\python.exe scripts/score_finetuned.py --predictions <gold> --tag oracle
# -> EXECUTION ACCURACY 100.00 % (453/453)   verified 2026-09-05
```

### Chat-template trap (Phase 10, cost one GPU run)

Qwen3's template renders a *finished* assistant turn as

```
<|im_start|>assistant
<think>

</think>

SELECT ...
```

so every SFT target was preceded by an empty think block. At inference,
`apply_chat_template(..., add_generation_prompt=True)` **without**
`enable_thinking=False` stops at `<|im_start|>assistant
` and leaves the model
to produce `<think>` itself — a starting point it never saw in training. The
first Phase 10 run did exactly that and returned reasoning prose, stray
`</think>` tags, junk lead tokens (`handgun`, `girl`, `apiro`) and repetition
loops that hit the 512-token cap.

**Always pass `enable_thinking=False` when generating with this adapter.**

The guard that missed it checked only that the inference prompt was a *prefix*
of the training rendering. It was — the whole problem sat in the gap after it.
The check now asserts that generation resumes at the assistant content itself,
and `predictions.jsonl` carries a `prompt_format` marker so a resume cannot
splice rows from two different formats together.

Verifiable locally without a GPU: the template is plain Jinja and ships in
`models/finetuned/final_adapter/chat_template.jinja`.

### Phase 10 result (frozen 2026-09-05)

Fine-tuned adapter, full schema, no retrieval, no repair — the Phase 4 setup
with different weights. 453 unseen test examples, greedy decoding, generated on
a Kaggle T4 in 30m38s and scored locally against PostgreSQL.

| metric | base | fine-tuned | delta |
| --- | ---: | ---: | ---: |
| **strict execution accuracy** | 10.82 % | **50.99 %** | **+40.17** |
| projection-tolerant accuracy | 45.92 % | 50.99 % | +5.07 |
| executable SQL | 98.90 % | 95.81 % | -3.09 |
| schema hallucination | 0.66 % | 1.99 % | +1.33 |
| syntax errors | 0.00 % | 0.22 % | +0.22 |

By difficulty:

| tier | n | base | fine-tuned | delta |
| --- | ---: | ---: | ---: | ---: |
| easy | 126 | 0.0 % | 23.0 % | +23.0 |
| medium | 63 | 52.4 % | 52.4 % | +0.0 |
| hard | 144 | 3.5 % | **84.7 %** | +81.2 |
| enterprise | 84 | 13.1 % | 52.4 % | +39.3 |

**What actually improved.** `projection_mismatch_only` went 159 -> **0**: the
baseline's single largest failure mode was returning the right rows under a
different column projection, and fine-tuning eliminated it entirely. Genuinely
wrong row sets fell far less, 240 -> 203. So most of the +40 pp is the model
learning this project's column conventions from the gold SQL — a real and
useful thing to learn, but not the same as becoming much better at SQL logic.
The `hard` tier (+81.2 pp) is where reasoning genuinely improved.

**What got worse.** Executable SQL fell 3.09 pp and hallucination rose 1.33 pp:
14 more queries reference something that is not in the schema, and one has a
syntax error. Fitting 2,133 examples hard (final train loss 0.006) made the
model more willing to emit a confident wrong identifier. Phase 11 repair
targets exactly these 19 failures.

Artefacts: `experiments/finetuned/` (results, summary, `FINETUNED_REPORT.md`).
Harness self-test on the same path returns 100 % — see the Phase 10 section
above.

### Phases 11-14 (built 2026-09-05)

**Phase 11 - repair.** `src/sql/repair.py` diagnoses a query (static check,
then execution) and `repair_loop` retries once with the database's own error
fed back. `src/model/repair_prompt.py` holds the prompt, import-free so it can
ship to a GPU host verbatim (fingerprint `b66ccdd66e919abd`).

* **Only detectable failures are repaired.** A query that runs and returns the
  wrong rows is indistinguishable from a correct one without gold, so it is
  left alone. Repair that needed gold would not work in production.
* **Two rates, not one.** *Success* = it now executes; *correctness* = it now
  returns the gold rows. A repair that turns a crash into a confident wrong
  answer scores on the first and not the second. Verified with two synthetic
  repair files: one that changes nothing (0 % / 0 %) and one that always
  returns `SELECT 1` (100 % / 0 %).
* `max_repairs=1`. Unbounded retries turn one bad question into an unbounded
  latency spike.

**Phase 12 - API.** `src/api/` on FastAPI. `POST /query`, `GET /schema`,
`GET /health`, `/docs`. 21 tests in `tests/test_api.py`, all against the real
database with a stub model — the model is the one component that cannot be
tested deterministically, so it is replaced and everything around it is tested
for real. Verified over real HTTP with both the stub and the live model
("How many customers are based in India?" -> `SELECT COUNT(*) FROM customers
WHERE country = 'India'` -> 1023).

Three safety layers, none of which depend on the model behaving: a non-superuser
role, read-only sessions with statement timeouts applied on pool checkout, and
static rejection of anything that is not a single read-only statement over real
tables.

`MODEL_BACKEND` picks what is served: `hf` (base model, 10.82 %), `local` (the
adapter, 50.99 %, needs CUDA), `stub` (tests). **The default is `hf`, the base
model**, because this laptop has no GPU.

**Phase 13 - Docker.** `docker/Dockerfile` (API, no torch, non-root,
healthcheck), `docker/Dockerfile.tools` (seeding), `docker-compose.yml`
(postgres + api + one-shot seed).

**Docker cannot be installed here**: Docker Desktop is absent and so is WSL2,
which it requires on Windows - that needs admin rights, a ~600 MB download and
a reboot, on 8 GB of RAM.

So `scripts/verify_docker_image.py` verifies everything a build would, short of
the build: it replays every `COPY` into a staging tree, creates a clean venv,
installs only `requirements.txt`, imports every module with just that tree on
the path, boots uvicorn from it with credentials passed purely as environment
variables, and runs the HEALTHCHECK command against both a live and a dead
port. **27 checks, all passing.** API payload is 25 files / 147 KB.

That rules out the failure that actually breaks most Dockerfiles - a forgotten
`COPY` or a dependency only ever installed by hand. Still unproven: `useradd`,
layer caching, and whether `psycopg[binary]` has a linux/amd64 wheel.

**Phase 14 - ablation.** `scripts/ablation_report.py` collects every measured
configuration into `experiments/ABLATION.md`, and lists unmeasured ones as
**not measured** rather than omitting them.

Configuration 4 (fine-tuned + retrieval) needs a retrieved-schema eval pack:
`scripts/export_eval_pack.py --retrieval keyword`. The generation notebook
handles both modes from one file, switched by `manifest.schema.mode`, and
`score_finetuned.py` routes results by the header's `schema_mode` so
configuration 3 cannot be overwritten by configuration 4. Retrieved schemas
average 2,407 chars / 5.08 tables against 5,455 / 12 full; 3 of 453 questions
retrieve everything, which is the retriever degenerating, not a bug.

### Configuration 4 result (frozen 2026-09-07)

Fine-tuned adapter with a retrieved schema subset, k=4 / expand=2. Mean prompt
727 tokens against ~1,490 for the full schema.

| metric | cfg 3 full schema | cfg 4 retrieved | delta |
| --- | ---: | ---: | ---: |
| strict execution accuracy | **50.99 %** | 41.72 % | **-9.27** |
| executable SQL | 95.81 % | 94.92 % | -0.89 |
| schema hallucination | 1.99 % | **3.53 %** | +1.54 |

By difficulty:

| tier | n | cfg 3 | cfg 4 | delta |
| --- | ---: | ---: | ---: | ---: |
| easy | 126 | 23.0 % | 19.0 % | -4.0 |
| medium | 63 | 52.4 % | **60.3 %** | +7.9 |
| hard | 144 | **84.7 %** | 63.9 % | **-20.8** |
| very_hard | 36 | 8.3 % | 5.6 % | -2.8 |
| enterprise | 84 | 52.4 % | 39.3 % | -13.1 |

**Retrieval hurts the fine-tuned model six times harder than it hurt the base
model** (-9.27 pp against -1.55 pp). Two reasons, and they compound:

* The adapter was trained *exclusively* on full-schema prompts, so a retrieved
  subset is out of distribution for it in a way it was not for the base model.
* The damage concentrates on `hard` (-20.8) and `enterprise` (-13.1) — exactly
  the tiers that need multiple tables. When the retriever drops a table a join
  needs, the model invents it: hallucination nearly doubles, 1.99 % -> 3.53 %.

`medium` is the one tier that *improved* (+7.9). Those are single-table
aggregations where a smaller schema is genuinely less distracting — which is
the effect retrieval is supposed to have, visible only where the query does not
need a join.

**The release gate blocks configuration 4** on hallucination (+2.87 pp against a
2.0 pp tolerance) despite its +30.90 pp gain over the baseline. Improvement
alone does not earn a release.

Conclusion, now measured rather than assumed: **retrieval is the wrong tool for
this 12-table schema, for the base model and the fine-tuned model alike.** Keep
the full schema.

### Configuration 5 result — repair (frozen 2026-09-07)

The fine-tuned adapter, full schema, one repair attempt on detectable failures.
19 failures re-generated on a Kaggle T4 in 6 m 30 s.

| metric | cfg 3 | **cfg 5 + repair** | delta |
| --- | ---: | ---: | ---: |
| strict execution accuracy | 50.99 % | **52.10 %** | +1.11 |
| executable SQL | 95.81 % | **98.23 %** | +2.42 |
| schema hallucination | 1.99 % | **0.66 %** | -1.33 |

**Repair repaid the entire fine-tuning regression.** Executable SQL and
hallucination both return to the base model's level (98.90 % / 0.66 %) while
keeping the accuracy gain. That is the whole argument for a self-correction
loop, and it is now measured rather than asserted.

**The two rates diverge sharply, which is the point of reporting both:**

| | | |
| --- | ---: | --- |
| repair *success* — now executes | 11 / 19 | **57.9 %** |
| repair *correctness* — now returns the gold rows | 5 / 19 | **26.3 %** |
| model returned identical SQL | 4 / 19 | |
| mean repair latency | 9,297 ms | |

Six of the eleven "successes" turned a crash into a **confident wrong answer**.
A single success rate would have reported 57.9 % and hidden that entirely.

**Repair works when the error names the fix, and fails when it names only the
symptom.** All five genuinely-correct repairs were `column "method" does not
exist` — PostgreSQL identifies the bad identifier, the schema shows
`payment_method`, and the model renames it. The failures were `datediff does
not exist` (says the function is wrong, not what the right date arithmetic is)
and `column reference is ambiguous` (says the reference is ambiguous, not which
table was meant). The model qualified or substituted, and guessed wrong.

Both remaining `unknown_column` cases it could not fix are genuinely missing
columns — `shipped_date` does not exist in `orders` at all — where it returned
identical SQL rather than inventing something. That is the right failure.

`max_repairs=1` is doing real work here: 4 of 19 returned the same SQL
unchanged, so a second attempt would mostly have re-spent latency.

### Failure analysis (2026-09-18)

`scripts/failure_analysis.py` re-executes every remaining configuration-5
failure next to its gold query and buckets it; report in
`experiments/FAILURE_ANALYSIS.md`. Of 217 failures: **118 `projection_only`**
(right rows, different column set — four easy templates fail 100 % because the
templates disagree on which columns to return), 87 `wrong_rows`, 4
`wrong_values`, 8 detectable. Row-level accuracy is 78.1 % (99.2 % on easy) —
**a diagnostic, never the headline.** Real errors cluster by template into
business definitions (~30, e.g. what "spending" means), month representation
(`m09`, 24/24, `EXTRACT(MONTH)` vs `DATE_TRUNC`), a missing `DATA_AS_OF`
reference date in the prompt (`x14`), and partitioned window functions
(`v02`, 12/12 — **no training example contains `PARTITION BY`**).

This was done on the test set. Any fix it motivates is decided on validation
and re-measured as a new configuration; 52.10 % stays frozen.

### Hardware constraint (important)

Local laptop: Intel Core i5-8365U, **8 GB RAM**, Intel UHD Graphics 620.
**There is no NVIDIA GPU. Never assume CUDA exists locally.**

| Local machine                         | Cloud GPU                     |
| ------------------------------------- | ----------------------------- |
| PostgreSQL                            | QLoRA fine-tuning of the 8B   |
| Python development                    | larger-model inference        |
| dataset generation and processing     |                               |
| SQL validation and execution          |                               |
| evaluation harness                    |                               |
| retrieval development                 |                               |
| API development                       |                               |
| lightweight inference where practical |                               |

Prefer free cloud GPU resources before paid ones.

---

## 8. Dependencies

Likely eventual stack: Python, PostgreSQL, PyTorch, Transformers, PEFT, TRL,
bitsandbytes, SQLGlot, FastAPI, Docker, vLLM.

**Install a dependency only when its phase begins.** With 8 GB RAM, avoid heavy
packages that are not yet needed. Pin versions in `requirements.txt`.

---

## 9. Project structure

### Currently on disk

Each holds a `.gitkeep` so the empty directory survives a commit (Git tracks
files, not directories).

```
configs/       # configuration files
data/          # raw, processed, train, validation, test splits
database/      # schema.sql, ERD.md, seed.sql
dataset/       # dataset generation, validation, formatting
evaluation/    # evaluation harness and metrics
scripts/       # operational entry points (init_db, load_schema, generate_data)
src/           # application source code (src/sql/config.py)
tests/         # pytest suite (93 database integrity tests)
training/      # SFT / LoRA training code
```

### Not yet created

Added as their phase begins:

```
models/        # baseline, finetuned adapters    (Phase 5+)
experiments/   # experiment configs and results  (Phase 5+)
docker/        # containerisation                (Phase 13+)
```

### Code layout convention (decided)

Pipeline code lives **inside `src/` as an importable Python package**:

```
src/
├── __init__.py
├── retrieval/    # schema retrieval                       (Phase 6+)
├── sql/          # validator.py, executor.py, repair.py   (Phase 1+, 11)
└── api/          # FastAPI service                        (Phase 12+)
```

Import style is absolute from the package root:

```python
from src.sql.executor import run_query
```

Subpackages are created when their phase begins, each with an `__init__.py`.

`database/`, `dataset/`, `training/`, `evaluation/`, and `scripts/` stay at the
top level — they are workflow and asset directories, not application code
imported by the running service.

Run commands from the **project root** so `src` resolves on the import path.

---

## 10. Code quality rules

1. Use Python type hints where appropriate.
2. Keep functions small and testable.
3. No hard-coded paths — resolve paths relative to the project root.
4. Configuration comes from config files / environment variables.
5. **Never hard-code secrets.** Database credentials live in environment variables.
6. Document non-obvious components.
7. Add tests for important functionality.
8. Avoid unnecessary abstractions.
9. Prefer reproducible experiments; record experiment configurations.
10. Never commit model weights or secrets.

---

## 11. How to work with the developer

The developer is a **beginner in LLM fine-tuning** and is learning while
building. Therefore:

- Explain important ML concepts (LoRA, QLoRA, tokenization, SFT, quantization,
  etc.) **before** implementing them. Do not assume prior knowledge.
- If a command changes the project, explain what it does.
- Work in **small milestones**. After each one, explain how to verify it.
- If something fails, debug it before moving on. Do not skip validation.
- Do not blindly generate large amounts of code.

When asked to **"implement X"**, respond in this order:

1. What X is.
2. Where it fits in the architecture.
3. Which files will be created or changed.
4. The implementation.
5. How to test it.

When asked for an **explanation**, explain the concept first — do not
immediately write code.

---

## 12. Environment notes

- OS: Windows 11, PowerShell.
- Virtual environment lives at `env/` (Python 3.11.9) and is git-ignored.
  Activate with `.\env\Scripts\Activate.ps1`.
- Git repository root is the project folder; default branch is `master`.


---

## 13. Operational notes (learned the hard way)

### Running inference

```powershell
# harness self-test, no API calls, must score 100%
env\Scripts\python.exe scripts/run_baseline.py --oracle --sample 25 --tag oracle

# smoke test  (--sample strides the split; --limit returns near-duplicates
#              because the split is ordered by template)
env\Scripts\python.exe scripts/run_baseline.py --sample 8 --provider featherless-ai --workers 6 --tag smoke

# full run
env\Scripts\python.exe scripts/run_baseline.py --provider featherless-ai --workers 6
```

**Always pass `--provider featherless-ai --workers 6`.** Measured, not guessed:

| setting | result |
| --- | --- |
| `provider=auto` | routes to `nscale`, which returns HTTP 402 constantly (3/6 calls) |
| `provider=featherless-ai` | 6/6 calls succeed |
| `workers=1` | 60 s per example |
| `workers=6` | 10 s per example (**6x faster**) |

An early 402 storm was misattributed to concurrency; it was the provider. Do
not re-derive this.

### HTTP 402 "Payment Required"

- Means the account's monthly inference credits are exhausted or throttled.
- **Per account, not per token.** Issuing a new HF token does not help —
  verified.
- It *recovers*: the throttle clears after ~33 s. Treated as retryable with
  long backoff (`THROTTLE_BACKOFF_S`), not fatal. Only 401/403 are fatal.
- Buying pre-paid credits removes it. A full 453-example run costs ~$0.02–0.15.

### Checkpointing

Generations are appended to `experiments/*/generations.jsonl` and flushed per
call. `--resume` replays them at zero API cost; `--only-checkpointed` scores
just what exists. A run cut short is never wasted — but note that `--tag`
changes the checkpoint filename, so resuming needs the matching tag.

### Measurement traps already hit

- **Latency must exclude retry backoff.** Timing the whole retry loop reported
  22,124 ms mean against a 1,376 ms median — that was HuggingFace's rate
  limiter, not the model. `latency.generation_clean` uses first-attempt calls.
- **Tune on `validation`, never `test`.** Retrieval hyperparameters were first
  swept on test by mistake and redone on validation.
- **Check both accuracy figures.** Strict accuracy alone is misleading here:
  35 % of the test set returns correct rows under a different column
  projection, because many questions never say which columns to return.
