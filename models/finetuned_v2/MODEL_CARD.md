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

# Qwen3-8B Text-to-SQL (QLoRA adapter, iteration 2)

A LoRA adapter that turns **Qwen3-8B** into a PostgreSQL text-to-SQL model for a
12-table enterprise schema (sales, catalogue, logistics, HR).

**Strict execution accuracy 43.71 % → 68.43 %** on 453 held-out questions, and
**70.86 %** with a one-shot self-correction loop. Same prompt, same schema, same
database, same executor, same GPU — only the weights changed.

Evaluation is execution-based: every prediction runs against a real PostgreSQL
database and its result set is compared to the gold query's. No string
similarity anywhere.

> ### Read this before quoting the headline
>
> This is the second iteration. The first scored 52.10 % on its own benchmark,
> and its failure analysis showed most of that gain was the model learning which
> **columns** the benchmark wanted — not better SQL. So the benchmark was
> rebuilt to state its conventions, a business glossary was added to the prompt,
> and the model retrained.
>
> Scoring **both pipelines on the identical v2 test set**:
>
> | | iteration 1 | iteration 2 |
> |---|---:|---:|
> | strict accuracy | 40.18 % | **70.86 %** |
> | right rows, wrong columns | 174 | **33** |
> | **genuinely wrong rows** | **85** | **84** |
> | row-level accuracy | 78.6 % | 78.1 % |
>
> **Genuinely wrong answers fell by one query.** The model finds the right rows
> about 78 % of the time and that did not change. The +30.68 pp is the benchmark
> and the prompt finally agreeing on what a correct answer looks like.
>
> This is the most useful thing the project has to say, and it is only visible
> because the harness scores projection separately instead of reporting one
> number.

---

## Two things that will silently break your results

**1. Pass `enable_thinking=False`.** Qwen3's chat template renders a finished
assistant turn with an empty `<think></think>` block, so every training target
was preceded by one. Without this flag at inference the model is asked to resume
from a position it never saw in training, and returns reasoning prose, stray
`</think>` tags and repetition loops. This cost one full GPU evaluation run.

**2. Use prompt v2, not a prompt of your own.** This adapter was trained against
a specific system prompt containing a business glossary and a `DATA_AS_OF`
reference date (fingerprint `4e72cc5f722ce436`). Swapping the prompt is worth
roughly ±10 pp on its own and makes any comparison meaningless.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER = "<your-username>/qwen3-8b-text2sql-qlora-v2"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", revision="b968826d9c46",
    quantization_config=bnb, device_map={"": 0},
    dtype=torch.float16, attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_V2},   # see src/model/prompt_v2.py
    {"role": "user", "content": f"Database schema:\n\n{SCHEMA}\n\n"
                                f"Question: {question}\n\nPostgreSQL query: /no_think"},
]
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,          # <- required
)
out = model.generate(**tok(text, return_tensors="pt").to(model.device),
                     max_new_tokens=512, do_sample=False)
```

---

## Results

453 held-out questions, benchmark v2, greedy decoding, 4-bit on one Kaggle T4.

| metric | base | fine-tuned | + repair |
|---|---:|---:|---:|
| **strict execution accuracy** | 43.71 % | **68.43 %** | **70.86 %** |
| projection-tolerant | 47.68 % | 68.43 % | 70.86 % |
| executable SQL | 95.58 % | 94.48 % | **98.23 %** |
| schema hallucination | 0.22 % | 3.75 % | **0.66 %** |

By difficulty:

| tier | n | base | fine-tuned | Δ |
|---|---:|---:|---:|---:|
| easy | 126 | 92.9 % | **100.0 %** | +7.1 |
| medium | 63 | 69.8 % | **96.8 %** | +27.0 |
| hard | 144 | 7.6 % | **54.9 %** | **+47.2** |
| very_hard | 36 | 0.0 % | 13.9 % | +13.9 |
| enterprise | 84 | 30.9 % | 46.4 % | +15.5 |

**Fine-tuning costs something, and self-correction pays it back.** The adapter
raises schema hallucination 0.22 % → 3.75 %: fitting the training set makes the
model more willing to emit a confident wrong identifier. One repair attempt on
detectable failures returns it to 0.66 % and executable SQL to 98.23 %, past the
base model's level, while keeping the accuracy gain.

Repair rates are reported as two numbers on purpose — 17 of 25 failures now
execute (68.0 % success) but only 11 return the gold rows (44.0 % correctness).
A single rate would hide that six repairs turned a crash into a confident wrong
answer.

---

## Training

| | |
|---|---|
| base | `Qwen/Qwen3-8B` @ `b968826d9c46`, 4-bit NF4 + double quant |
| adapter | LoRA r=16, alpha=32, dropout 0.05, all 7 attention + MLP projections |
| trainable | 43,646,976 of 4,761,498,624 — **0.917 %** |
| loss | completion-only, prompt masked to `-100` |
| supervised token share | **2.41 %** |
| sequence length | 2048 (longest example 1,944 tokens, 0 truncated) |
| optimiser | lr 2e-4, cosine, warmup 3 %, max grad norm 0.3, seed 20260808 |
| batch | 1 x 16 grad-accum = effective 16 |
| hardware | one free Kaggle T4, 1 epoch, 135 steps, **7 h 05 m** |
| train loss | **0.083** |
| eval loss | 0.184 (step 50) → 0.143 (100) → **0.145** (final) |

**The loss curve is the clearest evidence iteration 2 worked.** Iteration 1
ended at train 0.006 / eval 0.335 — a model that had memorised its 2,133
templates. This one ends at train 0.083 / eval **0.145**: it fits the training
set less hard and generalises more than twice as well. Loss is still not the
result; execution accuracy is.

**Why completion-only loss matters here.** Each example is ~1,814 prompt tokens
and ~45 completion tokens, and the prompt is almost entirely schema —
byte-identical across all 2,175 examples. Training on the full sequence would
send ~97 % of the gradient into memorising a schema the model is *handed* at
inference. The loss curve would look excellent while the ability you care about
barely moved.

**Fitting a 2,048-token sequence on a 16 GB T4** needed the lm_head and
cross-entropy to run only on the ~45 supervised positions rather than all
~1,860 (`logits_to_keep`). Their 151k-vocabulary logits plus fp32 copies were
~3 GB per step. The loss is identical; the first attempt without this died with
CUDA OOM at step 34 of 136.

Resolved package versions (recorded, not pinned): torch 2.10.0+cu128,
transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, accelerate 1.15.0,
trl 1.13.0.

---

## Limitations

- **Schema-specific.** It learned *this* database's conventions, which is most
  of the gain. It will not transfer to another schema without retraining.
- **`very_hard` sits at 13.9 %.** Window functions and nested aggregation remain
  largely unsolved; five new training templates moved the tier off zero and no
  further.
- **Every test question came from the same generator as training.** Real users
  write abbreviations, typos and genuinely ambiguous requests. This is the
  biggest caveat on 70.86 %.
- **One epoch, one seed, one run.** No variance estimate.
- **Silent wrong answers are the real risk.** 84 of the 132 remaining failures
  execute fine and return the wrong rows, and self-correction cannot see them:
  there is no error to feed back.
- **Hallucination is higher than the base model's** before repair (3.75 % vs
  0.22 %). Run the repair loop.

## License

Apache 2.0, inheriting the base model's licence. All training data is synthetic.
