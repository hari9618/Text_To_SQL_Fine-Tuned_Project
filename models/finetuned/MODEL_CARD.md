---
base_model: Qwen/Qwen3-8B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - text-to-sql
  - sql
  - postgresql
  - qlora
  - lora
  - peft
  - qwen3
language:
  - en
---

# Qwen3-8B Text-to-SQL (QLoRA adapter)

A LoRA adapter that turns **Qwen3-8B** into a PostgreSQL text-to-SQL model for a
12-table enterprise schema (sales, catalogue, logistics, HR).

**Strict execution accuracy went from 10.82 % to 50.99 %** on 453 held-out
questions — same prompt, same schema, same database, same executor. Only the
weights changed.

The evaluation is execution-based: every prediction is run against a real
PostgreSQL database and its result set compared to the gold query's. No string
similarity anywhere.

> **Read the [Where the gain came from](#where-the-gain-actually-came-from)
> section before quoting the headline number.** Most of the improvement is the
> model learning this database's column conventions, not becoming dramatically
> better at SQL logic. That distinction is measurable, and it is reported.

---

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER = "<your-username>/qwen3-8b-text2sql-qlora"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", revision="b968826d9c46",   # pin: the exact weights trained on
    quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

SYSTEM = """You are an expert PostgreSQL analyst. You convert business \
questions into correct, executable PostgreSQL queries.

Rules:
- Output ONLY the SQL query. No explanation, no commentary.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax.
- When a foreign key is nullable, consider whether LEFT JOIN is needed to \
avoid silently dropping rows."""

USER = """Database schema:

{schema}

Question: {question}

PostgreSQL query:"""


def to_sql(question: str, schema: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": USER.format(schema=schema, question=question) + " /no_think"},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,          # <-- REQUIRED, see below
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
```

### ⚠️ `enable_thinking=False` is not optional

Qwen3's chat template renders a *finished* assistant turn as:

```
<|im_start|>assistant\n<think>\n\n</think>\n\nSELECT ...
```

Every training target was therefore preceded by an **empty think block**. With
`add_generation_prompt=True` and no `enable_thinking` argument, the template
stops at `<|im_start|>assistant\n` and leaves the model to produce `<think>`
itself — a starting point it never saw in training.

Omitting it produces reasoning prose, stray `</think>` tags, junk lead tokens
and repetition loops. This cost one full GPU evaluation run to discover.

---

## Prompt format

The adapter expects **the exact prompt above**, with the schema rendered as
`CREATE TABLE` statements plus `--` foreign-key comments, and the user turn
ending in `/no_think`.

| | |
|---|---|
| prompt template fingerprint | `8288e41a496531a9` |
| schema rendering fingerprint | `d03619e711661bc5` |
| schema size | 5,455 characters, 12 tables |

A different prompt will not reproduce these numbers. The fingerprints exist so
that can be checked rather than assumed.

---

## Training

QLoRA on a single **Tesla T4** (free Kaggle tier), **one epoch, 4 h 53 m**.

| | |
|---|---|
| base model | `Qwen/Qwen3-8B` @ `b968826d9c46` |
| quantisation | 4-bit NF4 + double quantisation, fp16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| target modules | `q,k,v,o,gate,up,down` — all attention and MLP projections |
| trainable parameters | **43,646,976 of 4,761,498,624 (0.917 %)** |
| sequence length | 2048 (longest example 1,602 tokens; nothing truncated) |
| effective batch / optimiser steps | 16 / 134 |
| learning rate | 2e-4, cosine, 3 % warmup |
| optimiser | paged 8-bit AdamW |
| loss | **completion-only** — prompt masked to `-100` |
| supervised token share | **3.01 %** |
| train loss | 4.10 → 0.006 |
| eval loss | 0.475 → 0.340 → **0.335** |
| seed | 20260808 |

### Why completion-only loss matters here

Each example is ~1,490 prompt tokens and ~46 completion tokens, and the prompt
is **~97 % database schema, byte-identical across all 2,133 examples**.

Training on the full sequence would send ~97 % of the gradient into memorising a
schema the model is *handed* at inference time. The loss curve would look
excellent throughout — predicting a constant is easy — while the ability you
actually care about barely moved.

Masking the prompt is what makes the remaining 3 % of tokens the thing being
learned.

---

## Dataset

Template-generated from a 12-table PostgreSQL database (~328,000 synthetic rows,
fixed seed). All data is synthetic; no real personal information.

| split | records | purpose |
|---|---:|---|
| train | 2,133 | fine-tuning |
| validation | 459 | monitoring, hyperparameter choices |
| test | 453 | **held out, measured once per configuration** |

**Splits are leakage-free by template equivalence group, not by row.**
Verified: 74 training templates, 48 test templates, **zero overlap**, and no
shared question strings. So the reported accuracy is generalisation to query
*patterns the model never saw*, not to new slot values in familiar patterns.

Difficulty tiers: `easy` (SELECT/WHERE/ORDER BY) → `medium` (GROUP BY, HAVING,
aggregates) → `hard` (joins, subqueries, CTEs) → `very_hard` (window functions,
nested aggregation) → `enterprise` (ambiguous terminology, NULL handling,
business rules).

---

## Evaluation

Execution-based. Every prediction runs against PostgreSQL and its result set is
fingerprinted (MD5 over sorted, stringified rows). These two are different
strings and both count as correct:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

Order is compared only where the gold query has an `ORDER BY`.

### Ablation — 453 held-out questions

| metric | base model | **+ this adapter** | base + retrieval | adapter + retrieval |
|---|---:|---:|---:|---:|
| **strict execution accuracy** | 10.82 % | **50.99 %** | 9.27 % | 41.72 % |
| projection-tolerant accuracy | 45.92 % | 50.99 % | 43.93 % | 41.72 % |
| executable SQL | 98.90 % | 95.81 % | 92.27 % | 94.92 % |
| schema hallucination | 0.66 % | 1.99 % | 3.31 % | 3.53 % |
| syntax errors | 0.00 % | 0.22 % | 0.00 % | 0.00 % |

### By difficulty

| tier | n | base | **adapter** | Δ |
|---|---:|---:|---:|---:|
| easy | 126 | 0.0 % | 23.0 % | +23.0 |
| medium | 63 | 52.4 % | 52.4 % | **+0.0** |
| hard | 144 | 3.5 % | **84.7 %** | **+81.2** |
| very_hard | 36 | — | 8.3 % | — |
| enterprise | 84 | 13.1 % | 52.4 % | +39.3 |

---

## Where the gain actually came from

A 4.7× improvement deserves scrutiny, not a victory lap.

| failure mode | base | adapter |
|---|---:|---:|
| right rows, **wrong columns** | 159 | **0** |
| genuinely wrong rows | 240 | 203 |

The benchmark's single largest failure was returning **correct data under a
different column projection** — *"show orders in 2023"* never says which columns
to return. That went to zero. Genuinely wrong answers fell far less.

So: **most of the +40 points is the model learning this project's column
conventions from the gold SQL.** That is a real and useful thing to learn, and
it is not the same as becoming much better at SQL logic. The `hard` tier
(+81.2 pp) is where reasoning genuinely improved.

A concrete example of what was learned — the base model wrote:

```sql
WHERE o.status = 'Completed'    -- returns 0 rows: no such value exists
```

The adapter writes:

```sql
WHERE status = 'completed'      -- the actual value in this database
```

The schema declares `status VARCHAR(20)` and never enumerates the allowed
values. The base model guessed a plausible enum; the adapter learned the real
one.

---

## Example outputs

All produced by this adapter, all verified correct by execution.

**easy** — *"Show payments with status completed."*
```sql
SELECT payment_id, order_id, amount, status FROM payments WHERE status = 'completed'
```

**medium** — *"How many orders are there in each status?"*
```sql
SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status ORDER BY order_count DESC
```

**hard** — *"Who are the top 15 customers by revenue?"*
```sql
SELECT c.customer_name,
       SUM(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100)) AS revenue
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
GROUP BY c.customer_id, c.customer_name
ORDER BY revenue DESC LIMIT 15
```

**enterprise** — *"What is our actual revenue, excluding cancelled and returned orders?"*
```sql
SELECT SUM(o.total_amount) AS revenue FROM orders o
WHERE o.status NOT IN ('cancelled', 'returned')
```

---

## Limitations

Stated plainly, because they bound what the numbers mean.

- **Schema-specific.** Trained on one 12-table schema. It has learned *this*
  database's conventions — that is most of the measured gain — so accuracy on a
  different schema will be far lower.
- **Half its answers are still wrong.** 50.99 % is a large improvement over
  10.82 %; it is not production-grade unaided.
- **It broke things too.** Executable SQL fell 3.09 pp and schema hallucination
  rose 1.33 pp: 19 queries fail where 5 did before. A final training loss of
  0.006 against an eval loss of 0.335 means it fit 2,133 examples hard, and a
  tightly-fit model emits confident wrong identifiers more readily.
- **`medium` did not improve at all** (52.4 % → 52.4 %). Unexplained.
- **Schema retrieval makes it worse**, and worse than it made the base model
  (−9.27 pp vs −1.55 pp): the adapter only ever saw full-schema prompts, so
  subsets are out of distribution for it. Give it the whole schema.
- **Test questions are template-generated**, from the same generator as
  training. Real user phrasing — abbreviations, typos, ambiguity — is untested.
  This is the biggest caveat on the headline number.
- **One epoch, one seed, one run.** No variance estimate.
- **Silent wrong answers are the real risk.** SQL that runs and returns the
  wrong rows is indistinguishable from a correct answer without a gold
  reference. Never put this in front of users without showing them the SQL.

## Intended use

Assisted SQL authoring against this schema, with a human reading the generated
query. **Not** an autonomous agent over a production database.

If you deploy it, do what the source project does and do not rely on the model
behaving: a non-superuser role, read-only sessions with statement timeouts, and
static rejection of anything that is not a single read-only statement over
tables that exist.

## Citation / source

Full pipeline, evaluation harness, ablation study and reproduction instructions:
the source repository accompanying this adapter.

Every number here comes from `experiments/` in that repository and is
reproducible from the frozen fingerprints listed above.
