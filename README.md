<div align="center">

# Enterprise Text-to-SQL — Fine-Tuning & Self-Correction

**Does fine-tuning actually improve text-to-SQL? I measured it instead of assuming.**

[![Model](https://img.shields.io/badge/🤗_Model-qwen3--8b--text2sql--qlora-FFD21E?style=flat-square)](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-3,087_examples-FFD21E?style=flat-square)](https://huggingface.co/datasets/hari-krishna-ai/enterprise-text-to-sql-benchmark)
[![Space](https://img.shields.io/badge/🤗_Demo-live_explorer-FFD21E?style=flat-square)](https://huggingface.co/spaces/hari-krishna-ai/enterprise-text-to-sql)
<br>
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![PyTorch](https://img.shields.io/badge/QLoRA-PEFT-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-164_passing-2f7d52?style=flat-square)

### 43.71 % → 70.86 %

strict execution accuracy on **453 held-out questions**
<br><sub>and the second iteration's real lesson: <b>almost none of that gain is better SQL</b></sub>

</div>

---

## The result

Same prompt, same schema, same database, same executor, same GPU. **Only the weights changed.**

| # | configuration | strict accuracy | executable SQL | hallucination |
|---|---|---:|---:|---:|
| 1 | Base Qwen3-8B | 43.71 % | 95.58 % | 0.22 % |
| 2 | Base + schema retrieval | *not measured on v2* | — | — |
| 3 | **Fine-tuned (QLoRA)** | **68.43 %** | 94.48 % | 3.75 % |
| 4 | Fine-tuned + retrieval | *not measured on v2* | — | — |
| 5 | **Fine-tuned + self-correction** | **70.86 %** | **98.23 %** | **0.66 %** |

Trained on **one free Kaggle T4** in 7 h 05 m. 43.6 M trainable parameters — **0.917 %** of the model.
Full table: [`experiments/v2/ABLATION.md`](experiments/v2/ABLATION.md).

> [!IMPORTANT]
> **The headline moved +30.68 points. Genuinely wrong answers fell by one query.**
>
> This project ran twice. Iteration 1 scored 52.10 % and the failure analysis showed most of its
> gain was the model learning which *columns* this benchmark wanted, not better SQL. So I rebuilt
> the benchmark to state its conventions, added a business glossary to the prompt, and retrained.
>
> Scoring **both pipelines on the identical v2 test set** shows exactly what iteration 2 bought:
>
> | | v1 pipeline | v2 pipeline |
> |---|---:|---:|
> | strict accuracy | 40.18 % | **70.86 %** |
> | `projection_only` failures — right rows, wrong columns | 174 | **33** |
> | **`wrong_rows` — genuinely wrong answers** | **85** | **84** |
> | row-level accuracy | 78.6 % | 78.1 % |
>
> The model finds the right rows about 78 % of the time, and **that did not change.** The entire
> +30.68 pp is the benchmark and prompt finally agreeing on what a correct answer looks like.
> Reproduce it: `python scripts/rescore.py --version v2 --all`.

<details>
<summary><b>Iteration 1 (frozen) — 10.82 % → 52.10 %</b></summary>

<br>

v1 is kept byte-for-byte reproducible, never retro-fitted. Its numbers on its own benchmark:

| # | configuration | strict accuracy | executable SQL | hallucination |
|---|---|---:|---:|---:|
| 1 | Base Qwen3-8B | 10.82 % | 98.90 % | 0.66 % |
| 2 | Base + schema retrieval | 9.27 % | 92.27 % | 3.31 % |
| 3 | Fine-tuned (QLoRA) | 50.99 % | 95.81 % | 1.99 % |
| 4 | Fine-tuned + retrieval | 41.72 % | 94.92 % | 3.53 % |
| 5 | **Fine-tuned + self-correction** | **52.10 %** | 98.23 % | 0.66 % |

The v1 base model scored 10.82 % largely because it was never told which columns to return, so
**the +40 pp headline overstated what fine-tuning taught.** Measuring that honestly is what
motivated iteration 2, and it is why the v2 base model starts at 43.71 % rather than 10.82 %:
same model, told the conventions.

</details>

---

## The pipeline

```mermaid
flowchart TD
    Q["Business question<br/><i>Who are the top 15 customers by revenue?</i>"] --> P[Prompt + full schema<br/>frozen template]
    P --> M["Qwen3-8B + QLoRA adapter"]
    M --> S[Generated SQL]
    S --> V{"Static validation<br/>parseable? one statement?<br/>read-only? real tables?"}
    V -->|rejected| R
    V -->|passes| X{"PostgreSQL<br/>read-only, 30s timeout"}
    X -->|error| R["Self-correction<br/>database error fed back"]
    X -->|success| OUT([Rows])
    R --> M2["Qwen3-8B + adapter<br/>repair prompt"]
    M2 --> X2{"PostgreSQL"}
    X2 -->|success| OUT
    X2 -->|still fails| FAIL([Reported as failed])

    style M fill:#c8623a,color:#fff
    style M2 fill:#c8623a,color:#fff
    style OUT fill:#2f7d52,color:#fff
    style FAIL fill:#b3402f,color:#fff
```

---

## Try it

<table>
<tr>
<td width="33%" align="center">

### 🔍 Try it
**[Live demo →](https://huggingface.co/spaces/hari-krishna-ai/enterprise-text-to-sql)**

Ask it anything, watch every step.
⚠️ runs the **base** model — no GPU to host the adapter.

</td>
<td width="33%" align="center">

### 📦 Use the model
**[Model card →](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)**

Copy-paste loading snippet.
⚠️ needs `enable_thinking=False`

</td>
<td width="33%" align="center">

### 📊 Benchmark yours
**[Dataset →](https://huggingface.co/datasets/hari-krishna-ai/enterprise-text-to-sql-benchmark)**

```python
load_dataset(
 "hari-krishna-ai/"
 "enterprise-text-to-sql-benchmark")
```

</td>
</tr>
</table>

---

## Why the evaluation is the interesting part

<details open>
<summary><b>Correctness is judged by execution, never by string similarity</b></summary>

<br>

These are different strings and identical in meaning. Any honest benchmark must count both correct:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

Every prediction is **executed against PostgreSQL** and its result set fingerprinted (MD5 over
sorted, stringified rows). Row order is compared only where the gold query has an `ORDER BY`.

</details>

<details>
<summary><b>Guardrails that keep the number honest</b></summary>

<br>

| guardrail | what it prevents |
|---|---|
| Splits by **template equivalence group**, **zero overlap**; v2 **pins the v1 split** — same 453 test ids, new templates train-only | a paraphrase of a test question appearing in training, or an iteration quietly getting an easier test set |
| Prompt fingerprint asserted at every stage (`8288e41a…` v1, `4e72cc5f…` v2) | measuring prompt engineering instead of fine-tuning, or mixing the two iterations |
| Schema fingerprint `d03619e711661bc5` asserted at every stage | comparing runs that saw different databases |
| The GPU host that generates **never receives gold SQL** | a "model" that copies the answer |
| Oracle self-test: gold SQL through the scorer must return **100 %** | a broken evaluator silently deflating every result |
| `DATA_AS_OF` constant instead of `NOW()` | a benchmark whose answers change overnight |
| Retrieval tuned on **validation**, measured once on test | hyperparameters fitted to the test set |

</details>

<details>
<summary><b>Two results that went against expectations — and are reported anyway</b></summary>

<br>

**Schema retrieval made things worse. Twice.**

Showing only the retrieved tables cost the base model 1.55 pts and the fine-tuned model **9.27**.
The adapter only ever saw full-schema prompts, so subsets are out of distribution for it — and the
damage lands on `hard` (−20.8) and `enterprise` (−13.1), the tiers that need joins. Drop a table a
join needs and the model invents one.

With 12 tables the whole schema is 5,455 characters. **Retrieval solves a problem this database
does not have.**

**Self-correction: two rates, not one.**

| | v1 | v2 |
|---|---:|---:|
| repair **success** — now executes | 11 / 19 · 57.9 % | 17 / 25 · 68.0 % |
| repair **correctness** — returns the gold rows | 5 / 19 · **26.3 %** | 11 / 25 · **44.0 %** |

The two rates diverge on purpose. In v1, six of eleven "successes" turned a crash into a *confident
wrong answer*; reporting one rate would have claimed 57.9 % and hidden that entirely.

v2 repairs far better (26.3 % → 44.0 %) for a reason worth stating: the v2 adapter's failures are
**window-function syntax** — `WITHIN GROUP is required for ordered-set aggregate rank` — where
PostgreSQL names the fix precisely and the model applies it. v1's failures were semantic guesses
the error text could only describe as a symptom. **Repair works when the error names the fix.**

In both iterations repair paid back the entire fine-tuning regression: executable SQL and
hallucination return past the *base* model's level (94.48 → 98.23 %, 3.75 → 0.66 %) while keeping
the accuracy gain. That is the whole argument for a self-correction loop, measured twice.

</details>

<details>
<summary><b>Accuracy by difficulty — where reasoning actually improved</b></summary>

<br>

Benchmark v2, base vs fine-tuned (configuration 3), same hardware and prompt:

| tier | n | base | fine-tuned | Δ |
|---|---:|---:|---:|---:|
| easy | 126 | 92.9 % | **100.0 %** | +7.1 |
| medium | 63 | 69.8 % | **96.8 %** | +27.0 |
| **hard** | 144 | 7.6 % | **54.9 %** | **+47.2** |
| very_hard | 36 | 0.0 % | 13.9 % | +13.9 |
| enterprise | 84 | 30.9 % | 46.4 % | +15.5 |

`hard` is where the adapter earns its keep: the base model writes 7.6 % of multi-join queries
correctly and the fine-tune writes 54.9 %. `very_hard` — window functions and nested aggregation —
goes from **nothing at all** to 13.9 %, which is the one place iteration 2's five new training
templates (`PARTITION BY`, `EXTRACT(EPOCH …)`) show up. It is still the weakest tier by a wide
margin and the obvious target for a third iteration.

</details>

<details>
<summary><b>Where the remaining 29 % goes — every failure classified</b></summary>

<br>

The headline says *how many* were wrong. `scripts/failure_analysis.py` says *why*: every one of the
132 remaining failures is re-executed next to its gold query and sorted into a bucket.
Full report: [`experiments/v2/FAILURE_ANALYSIS.md`](experiments/v2/FAILURE_ANALYSIS.md).

| bucket | n | of test set | what it means |
|---|---:|---:|---|
| `wrong_rows` | **84** | 18.5 % | genuinely different rows — a real logic error |
| `projection_only` | 33 | 7.3 % | right rows, different column set |
| `wrong_values` | 7 | 1.5 % | right rows, a computed value not returned |
| detectable (hallucination / execution / syntax) | 8 | 1.8 % | the only failures self-correction can see |

**Iteration 2 inverted this distribution.** In v1, `projection_only` was the largest bucket at 118
and real logic errors were 87. Now real logic errors dominate at 84 while projection noise has
collapsed to 33. The benchmark stopped punishing the model for guessing column lists, and what is
left is mostly SQL the model genuinely got wrong — which is the failure mode worth working on.

Counting projection-only cases as correct gives a **row-level accuracy of 78.1 %**. That is a
diagnostic, **not** the headline: the benchmark was frozen before any of this was looked at, and
loosening a metric after seeing test results is exactly the move this project refuses to make.

**What is still broken**, in order of size:

| root cause | n | what it would take |
|---|---:|---|
| **Business definitions** — *"highest spending customers"*: the gold sums discounted line items, the model sums `orders.total_amount`. The v2 glossary states this rule and the model still misapplies it on multi-join questions | ~30 | training examples that *use* the rule, not just a prompt that states it |
| **Window functions and nested aggregation** — `very_hard` sits at 13.9 %. Five new training templates were not enough | ~31 | many more partitioned-window examples; this is a capability gap, not a convention gap |
| **Entity linking** — *"handled by GlobalEx"*: the model looks for a customer, not a carrier | ~10 | value-aware schema context saying which column holds `'GlobalEx'` |

The honest summary for an interviewer: **70.9 % strict, 78.1 % on rows. Iteration 2 fixed the
measurement; iteration 3 would have to fix the reasoning, and the failure analysis says exactly
where.**

</details>

---

## Training

<details open>
<summary><b>QLoRA on one free T4 — and the detail that mattered most</b></summary>

<br>

| | |
|---|---|
| base | `Qwen/Qwen3-8B` @ `b968826d9c46`, 4-bit NF4 + double quant |
| adapter | LoRA r=16, α=32, dropout 0.05, all 7 attention + MLP projections |
| trainable | **43,646,976 of 4,761,498,624 — 0.917 %** |
| loss | **completion-only**, prompt masked to `-100` |
| supervised token share | **3.01 %** |
| train / eval loss | 1.75 → **0.083** / 0.184 → **0.145** |
| hardware | one free Kaggle T4, 1 epoch, 135 steps, 7 h 05 m |

**The loss curve is the clearest sign iteration 2 worked.** v1 ended at train 0.006 / eval 0.335 —
a model that had memorised its 2,133 templates. v2 ends at train 0.083 / eval **0.145**: it fits
the training set *less* hard and generalises more than twice as well to held-out data. Loss is
still not the result, though — execution accuracy is, and that is the table at the top.

**Why completion-only loss is the whole game here.** Each example is ~1,490 prompt tokens and ~46
completion tokens, and the prompt is ~97 % schema — byte-identical across all 2,133 examples.
Training on the full sequence sends ~97 % of the gradient into memorising a schema the model is
*handed* at inference. The loss curve would look excellent throughout, because predicting a
constant is easy, while the ability you care about barely moved.

</details>

---

## Production concerns

<details>
<summary><b>Three safety layers — none of which trust the model</b></summary>

<br>

The service executes text a language model wrote. These hold regardless of what it writes:

1. **The database role is not a superuser.** It owns one database and nothing else.
2. **Every session is read-only with a statement timeout**, applied on pool checkout rather than
   once at creation — a reset session cannot silently lose the guarantee.
3. **SQL is statically rejected** unless it is a single read-only statement over tables that exist.

**32 adversarial tests** attack them. Two attacks get past the validator — `pg_read_file` (a single
read-only SELECT over no tables) and writes smuggled through a CTE (parses as SELECT). The role
permissions and the read-only transaction stop them respectively. That is what defence-in-depth
means, and it is now evidence rather than an argument.

</details>

<details>
<summary><b>A release gate, not just a benchmark</b></summary>

<br>

`scripts/release_gate.py` exits non-zero and **blocks a release**. Improvement alone is not enough:

```
[PASS] evaluated on the full held-out split       453 examples (need 453)
[PASS] candidate and baseline rendered the same prompt   4e72cc5f722ce436
[PASS] prompt matches this version's frozen prompt       4e72cc5f722ce436
[PASS] scored against the same database as the baseline  5a018e291c3df4ad
[PASS] accuracy gain clears the bar              +27.15 pp (need >= 5.0)
[PASS] executable SQL did not regress too far     +2.65 pp (tolerance 4.0)
[PASS] schema hallucination did not rise too far  +0.44 pp (tolerance 2.0)
[PASS] no difficulty tier collapsed               5 tiers compared
RELEASE APPROVED
```

Verified in both directions — it **blocks** v1's configuration 4 on hallucination (+2.87 pp against
a 2.0 tolerance) despite its +30.90 pp accuracy gain, and caught a `medium`-tier collapse of
−7.9 pp that the mean alone would have hidden.

It also caught a real mistake during iteration 2: pointed at the v2 candidate it **blocked on the
prompt fingerprint**, because the check was hard-coded to v1's. The fix was not to loosen it but to
make it version-aware and *stronger* — the candidate and baseline must now prove they rendered the
**same** prompt as each other, and that it is the one the benchmark version froze. Comparing a
glossary prompt against a plain one would otherwise have been reported as a fine-tuning gain.

</details>

---

## Quick start

```bash
python -m venv env && env\Scripts\activate      # Windows
pip install -r requirements.txt

cp .env.example .env                            # then fill in credentials

python scripts/init_db.py                       # database + non-superuser role
python scripts/load_schema.py
python scripts/generate_data.py                 # ~328k rows, fixed seed, ~1 min

python -m pytest tests/ -q                      # 164 tests
python -m uvicorn src.api.main:app --reload     # http://localhost:8000
```

<details>
<summary><b>Reproducing the experiment</b></summary>

<br>

```bash
# harness self-test — no model calls, must score 100 %
python scripts/run_baseline.py --oracle --sample 60 --tag oracle

# the frozen baseline
python scripts/run_baseline.py --provider featherless-ai --workers 6

# export what a GPU host needs (questions only; gold SQL never leaves)
python scripts/export_eval_pack.py --version v2

# score predictions generated on the GPU host
python scripts/score_finetuned.py --version v2 --predictions predictions_v2_final.jsonl
python scripts/score_repair.py    --version v2 --repairs repairs.jsonl

# assemble the ablation and gate the release
python scripts/ablation_report.py --version v2
python scripts/release_gate.py

# classify every remaining failure (re-executes gold and prediction side by side)
python scripts/failure_analysis.py --version v2

# the controlled comparison: replay v1's predictions against v2 gold, no GPU needed
python scripts/rescore.py --version v2 --all
```

Every script takes `--version v1|v2`; **v1 is the default and stays byte-for-byte reproducible.**
`src/benchmark_versions.py` resolves the dataset, prompt module and experiment directory, so the
two iterations cannot contaminate each other.

Generation needs a GPU; scoring needs PostgreSQL. They live on different machines, so the eval pack
carries `{id, question}` only — **the GPU host never sees a gold answer**, asserted at both ends.

</details>

---

## Repository

```
src/
├── api/          FastAPI service + demo UI
├── evaluation/   benchmark runner, 11-class outcome taxonomy, reporting
├── model/        frozen prompts, schema context, model backends
├── retrieval/    lexical schema retriever
└── sql/          validator, read-only executor, repair loop
database/         schema.sql, ERD.md
dataset/          generation, validation, SFT formatting, splits
training/         QLoRA training code + Kaggle notebooks
experiments/      frozen v1 results + v2/ second iteration + failure analyses
scripts/          operational entry points
deploy/ docker/   Space, Neon + Render, container images
tests/            164 tests (93 database, 24 API, 32 adversarial, 13 benchmark v2)
```

---

## Limitations

Stated plainly, because they bound what the number means.

- **The live demo serves the base model, not the fine-tuned one.** The adapter needs a GPU and the
  free Render tier has none, so what an interviewer clicks is the base model behind prompt v1.
  The 70.86 % figure comes from Kaggle T4 runs that anyone can reproduce from the committed eval
  pack and notebooks — it is not what the hosted API returns.
- **+30.68 pp of the v2 gain is convention, not capability.** Genuinely wrong row sets fell 85 → 84
  between iterations. Row-level accuracy is ~78 % in both. The number went up because the benchmark
  and prompt stopped disagreeing about what a correct answer looks like.
- **Every test question came from the same generator as training.** Real users write abbreviations,
  typos and genuinely ambiguous requests. **This is the biggest caveat on 70.86 %.**
- **`very_hard` sits at 13.9 %.** Window functions and nested aggregation remain largely unsolved;
  five new training templates moved it off zero and no further.
- **One epoch, one seed, one run, per iteration.** No variance estimate.
- **Schema-specific.** It learned *this* database's conventions — which is most of the gain.
- **Silent wrong answers are the real risk**, and self-correction cannot help: a query that runs
  and returns the wrong rows raises no error to feed back. 84 of the 132 failures are exactly this.
- **`docker build` has never been run** — Docker is not installed on the development machine.
  `scripts/verify_docker_image.py` verifies the image contents and entrypoint instead (27 checks).

---

<div align="center">

**[Demo](https://huggingface.co/spaces/hari-krishna-ai/enterprise-text-to-sql)** ·
**[Model](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)** ·
**[Dataset](https://huggingface.co/datasets/hari-krishna-ai/enterprise-text-to-sql-benchmark)**

Apache 2.0 · all data synthetic

</div>
