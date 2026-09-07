"""Enterprise Text-to-SQL — ZeroGPU demo Space.

Serves the fine-tuned QLoRA adapter (hari-krishna-ai/qwen3-8b-text2sql-qlora)
on ZeroGPU, and shows the pipeline the benchmark actually measured:

    question -> prompt (frozen template) -> Qwen3-8B + adapter
             -> static validation (sqlglot)
             -> the SQL, with the trace

Execution against PostgreSQL is deliberately *not* in this Space. The benchmark
executes every prediction against a real database; a public Space has no
database attached, so it stops at static validation and says so rather than
implying a result it cannot produce.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import gradio as gr
import spaces
import sqlglot
import torch
from peft import PeftModel
from sqlglot import exp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER = os.getenv("ADAPTER_REPO", "hari-krishna-ai/qwen3-8b-text2sql-qlora")
BASE = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46"      # the exact weights the adapter was trained on

EXPECTED_PROMPT_FP = "8288e41a496531a9"
EXPECTED_SCHEMA_FP = "d03619e711661bc5"

with open("schema_context.txt", encoding="utf-8") as fh:
    SCHEMA = fh.read()

SCHEMA_FP = hashlib.sha256(SCHEMA.encode("utf-8")).hexdigest()[:16]

# --------------------------------------------------------------------------
# The frozen prompt. Byte-identical to the one used for training and for every
# number in the ablation below -- a reworded prompt would change the results.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert PostgreSQL analyst. You convert business \
questions into correct, executable PostgreSQL queries.

Rules:
- Output ONLY the SQL query. No explanation, no commentary.
- Use only the tables and columns given in the schema.
- Write a single SELECT statement. Never modify data.
- Use PostgreSQL syntax.
- When a foreign key is nullable, consider whether LEFT JOIN is needed to \
avoid silently dropping rows."""

USER_TEMPLATE = """Database schema:

{schema}

Question: {question}

PostgreSQL query:"""

PROMPT_FP = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + USER_TEMPLATE + "\x00" + "v1").encode("utf-8")
).hexdigest()[:16]

TABLES = set(re.findall(r"CREATE TABLE (\w+)", SCHEMA))

# --------------------------------------------------------------------------
# Model. ZeroGPU requires placement on cuda at module level; a PyTorch CUDA
# emulation mode makes that work outside @spaces.GPU functions.
# 4-bit NF4 matches the configuration every benchmark number was measured with.
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
if not getattr(tokenizer, "chat_template", None):
    # the adapter ships the template as a separate file; older transformers
    # will not pick it up on its own, and a default template would render a
    # prompt the model never saw
    from huggingface_hub import hf_hub_download

    tokenizer.chat_template = open(
        hf_hub_download(ADAPTER, "chat_template.jinja"), encoding="utf-8").read()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

_base = AutoModelForCausalLM.from_pretrained(
    BASE, revision=REVISION,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16),
    device_map={"": 0},
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(_base, ADAPTER).eval()


def extract_sql(raw: str) -> str:
    """Pull the statement out of the reply. Locates SQL; never repairs it."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I).strip()
    fenced = re.search(r"```(?:sql|postgresql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    match = re.search(r"\b(WITH|SELECT)\b", text, re.I)
    if match:
        text = text[match.start():]
    if ";" in text:
        text = text.split(";", 1)[0]
    return " ".join(text.split()).strip()


def validate(sql: str) -> tuple[bool, str]:
    """The same static check the benchmark applies before execution."""
    if not sql:
        return False, "no SQL statement was produced"
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s]
    except Exception as exc:  # sqlglot raises several parse error types
        return False, f"does not parse: {str(exc).splitlines()[0][:160]}"
    if len(statements) != 1:
        return False, f"expected one statement, found {len(statements)}"
    if not isinstance(statements[0], (exp.Select, exp.Union, exp.Subquery)):
        return False, f"not a read-only SELECT ({type(statements[0]).__name__})"

    cte_names = {c.alias_or_name for s in statements
                 for c in s.find_all(exp.CTE)}
    referenced = {t.name.lower() for s in statements
                  for t in s.find_all(exp.Table) if t.name}
    unknown = sorted(referenced - {t.lower() for t in TABLES}
                     - {c.lower() for c in cte_names})
    if unknown:
        return False, f"tables not in the schema: {', '.join(unknown)}"
    return True, f"single read-only statement over {len(referenced)} real table(s)"


@spaces.GPU(duration=60)
def generate(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": USER_TEMPLATE.format(schema=SCHEMA, question=question)
                    + " /no_think"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        # REQUIRED. Training rendered an empty <think></think> block before
        # every target; omitting it here gives the model a starting point it
        # never saw, and the output becomes unusable.
        enable_thinking=False,
    )
    enc = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "", "Type a question, or pick an example below.", ""
    if len(question) > 2000:
        return "", "That question is too long (2000 character limit).", ""

    started = time.perf_counter()
    try:
        raw = generate(question)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return "", f"### Generation failed\n\n`{type(exc).__name__}: {exc}`", ""
    elapsed = (time.perf_counter() - started) * 1000

    sql = extract_sql(raw)
    ok, detail = validate(sql)

    trace = [
        "**Pipeline**",
        "",
        f"- ✅ **Prompt built** — frozen template `{PROMPT_FP}`, "
        f"schema `{SCHEMA_FP}` (12 tables, {len(SCHEMA):,} chars)",
        f"- ✅ **SQL generated** — fine-tuned adapter, greedy decoding, "
        f"{elapsed:,.0f} ms on ZeroGPU",
        (f"- ✅ **Static validation passed** — {detail}" if ok
         else f"- ❌ **Static validation rejected it** — {detail}"),
        "- ⏸️ **Execution** — not available in this Space (no database "
        "attached). The benchmark executes every query against real "
        "PostgreSQL; here the pipeline stops at validation.",
        "",
        "> Valid SQL is **not** a correct answer. A query can parse, reference "
        "only real tables, run cleanly and still return the wrong rows — the "
        "most dangerous failure mode, and why the benchmark scores by "
        "execution against gold result sets rather than by inspection.",
    ]
    return (sql or "(no SQL produced)"), "\n".join(trace), raw


EXAMPLES = [
    "Show payments with status completed.",
    "How many orders are there in each status?",
    "Who are the top 15 customers by revenue?",
    "What is our actual revenue, excluding cancelled and returned orders?",
    "Which products have never been ordered?",
    "Show the average product price per category.",
]

CSS = """
.headline { font-size: 15px; line-height: 1.6; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="Enterprise Text-to-SQL", theme=gr.themes.Soft(),
               css=CSS) as demo:
    gr.Markdown(
        """
# Enterprise Text-to-SQL — fine-tuned Qwen3-8B

Ask a business question in plain English. A **QLoRA-fine-tuned Qwen3-8B**
turns it into PostgreSQL against a 12-table enterprise schema, and the SQL is
statically validated before you see it.

**Fine-tuning took strict execution accuracy from 10.82 % → 50.99 %, and
52.10 % with a self-correction loop**, measured on 453 held-out questions by
*executing* every query against a real database — never by string similarity.

[📦 Model](https://huggingface.co/hari-krishna-ai/qwen3-8b-text2sql-qlora)
&nbsp;·&nbsp; adapter: 43.6 M trainable params (0.917 % of the model)
&nbsp;·&nbsp; trained on one free T4 in 4 h 53 m
""", elem_classes="headline")

    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question", lines=3,
                    placeholder="Who are the top 15 customers by revenue?")
                run = gr.Button("Generate SQL", variant="primary")
                gr.Examples(EXAMPLES, inputs=question, label="Try one")
            with gr.Column(scale=3):
                sql_out = gr.Code(label="Generated SQL", language="sql")
                trace_out = gr.Markdown()
                with gr.Accordion("Raw model output", open=False):
                    raw_out = gr.Textbox(label="", lines=6, show_copy_button=True)

        run.click(ask, inputs=question, outputs=[sql_out, trace_out, raw_out])
        question.submit(ask, inputs=question,
                        outputs=[sql_out, trace_out, raw_out])

    with gr.Tab("Benchmark"):
        gr.Markdown(
            """
## Ablation — 453 held-out questions

Every column scored on the same questions, same harness, same database. Only
the named component changes.

| metric | 1 base | 2 + retrieval | 3 fine-tuned | 4 FT + retrieval | 5 **FT + repair** |
|---|---:|---:|---:|---:|---:|
| **strict execution accuracy** | 10.82 % | 9.27 % | 50.99 % | 41.72 % | **52.10 %** |
| projection-tolerant | 45.92 % | 43.93 % | 50.99 % | 41.72 % | **52.10 %** |
| executable SQL | 98.90 % | 92.27 % | 95.81 % | 94.92 % | **98.23 %** |
| schema hallucination | 0.66 % | 3.31 % | 1.99 % | 3.53 % | **0.66 %** |

### By difficulty — base → fine-tuned

| tier | n | base | fine-tuned | Δ |
|---|---:|---:|---:|---:|
| easy | 126 | 0.0 % | 23.0 % | +23.0 |
| medium | 63 | 52.4 % | 52.4 % | **+0.0** |
| hard | 144 | 3.5 % | **84.7 %** | **+81.2** |
| enterprise | 84 | 13.1 % | 52.4 % | +39.3 |

### Where the gain actually came from

| failure mode | base | fine-tuned |
|---|---:|---:|
| right rows, **wrong columns** | 159 | **0** |
| genuinely wrong rows | 240 | 203 |

The benchmark's largest failure was returning correct data under a different
column projection — *"show orders in 2023"* never says which columns. That went
to zero. Genuinely wrong answers fell far less.

**So most of the +40 points is the model learning this database's column
conventions, not becoming dramatically better at SQL logic.** The `hard` tier
(+81.2 pp) is where reasoning genuinely improved. `medium` did not move at all,
and that is unexplained.

### Schema retrieval made things worse — twice

Showing only the retrieved tables cost the base model 1.55 points and the
fine-tuned model **9.27**. The adapter only ever saw full-schema prompts, so
subsets are out of distribution for it; and the damage lands on `hard` (−20.8)
and `enterprise` (−13.1), the tiers that need joins. Drop a table a join needs
and the model invents one.

With 12 tables the whole schema is 5,455 characters. **Retrieval solves a
problem this database does not have.**

### Self-correction: two rates, not one

| | | |
|---|---:|---|
| repair **success** — now executes | 11 / 19 | 57.9 % |
| repair **correctness** — returns the gold rows | 5 / 19 | 26.3 % |

**Six of eleven "successes" turned a crash into a confident wrong answer.** One
rate would have claimed 57.9 % and hidden that. Repair works when the error
*names the fix* (`column "method" does not exist` → rename to `payment_method`)
and fails when it names only the *symptom* (`ambiguous column reference`).
""")

    with gr.Tab("How it works"):
        gr.Markdown(
            f"""
## Training

QLoRA on one **free Kaggle T4**, one epoch, 4 h 53 m.

| | |
|---|---|
| base | `Qwen/Qwen3-8B` @ `{REVISION}`, 4-bit NF4 + double quant |
| adapter | LoRA r=16, α=32, dropout 0.05, all 7 attention + MLP projections |
| trainable | **43,646,976 of 4,761,498,624 — 0.917 %** |
| loss | **completion-only**, prompt masked to `-100` |
| supervised token share | **3.01 %** |
| train / eval loss | 4.10 → 0.006 / 0.475 → **0.335** |

**Why completion-only loss is the detail that matters.** Each example is ~1,490
prompt tokens and ~46 completion tokens, and the prompt is ~97 % schema,
byte-identical across all 2,133 examples. Training on the full sequence sends
~97 % of the gradient into memorising a schema the model is *handed* at
inference — and the loss curve looks excellent throughout, because predicting a
constant is easy.

## How it is evaluated

By **execution**, never string similarity. Every prediction runs against
PostgreSQL and its result set is fingerprinted. These are different strings and
both count as correct:

```sql
SELECT COUNT(*)           FROM customers WHERE country = 'India';
SELECT COUNT(customer_id) FROM customers WHERE country = 'India';
```

Guardrails: splits are leakage-free by **template group** — 74 train templates,
48 test templates, **zero overlap** — the prompt and schema fingerprints are
asserted at every stage, the GPU host that generates predictions never receives
gold SQL, and an oracle self-test (gold SQL through the scorer) must return
100 % before any real number is trusted.

## Reproducibility

| | |
|---|---|
| prompt fingerprint | `{PROMPT_FP}` |
| schema fingerprint | `{SCHEMA_FP}` |
| base model revision | `{REVISION}` |

## Limitations

- **Schema-specific.** It learned *this* database's conventions — that is most
  of the measured gain. Accuracy elsewhere will be far lower.
- **Roughly half its answers are still wrong.**
- **Test questions are template-generated**, from the same generator as
  training. Real user phrasing is untested — the biggest caveat on the number.
- **One epoch, one seed, one run.** No variance estimate.
- **This Space does not execute anything.** Valid SQL is not a correct answer.
""")

demo.queue(max_size=16).launch()
