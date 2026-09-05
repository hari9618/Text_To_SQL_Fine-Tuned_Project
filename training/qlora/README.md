# Phase 8/9 — QLoRA fine-tuning on a free Colab GPU

**Nothing has been trained yet.** This directory contains everything needed to
run the fine-tune reproducibly on a free Google Colab T4. No local GPU, no paid
cloud.

| File | Purpose |
|---|---|
| `Phase9_QLoRA_Qwen3_8B_Text2SQL.ipynb` | Colab notebook — run this |
| `train_qlora.py` | The training script (works anywhere with CUDA) |
| `requirements-colab.txt` | Pinned dependencies, Colab only |

---

## What QLoRA is, briefly

Fine-tuning all 8.19 billion parameters of Qwen3-8B is impossible on free
hardware: the weights alone are 16.4 GB in bf16, and Adam would need roughly
three times that again for optimiser state. Well over 60 GB.

**LoRA** freezes the base model and inserts small trainable low-rank matrices
beside each attention and MLP projection. Only those train — here **41.9 M
parameters, about 0.5 % of the model**. The rest never receives a gradient.

**QLoRA** adds one more idea: quantise the frozen base to **4-bit NF4**, so the
weights occupy ~4.4 GB instead of 16.4 GB, while the adapters stay in 16-bit.
Quantisation error matters little because those weights are never updated.

That combination is what fits an 8B fine-tune into a 16 GB T4.

---

## Resource requirements

### VRAM — expected ~9–11 GB of the T4's 15 GB usable

| Component | Estimate |
|---|---:|
| Base model, 4-bit NF4 + double quantisation | ~4.4 GB |
| LoRA adapters (r=16, fp16) | ~0.09 GB |
| Optimiser state (paged 8-bit AdamW, adapters only) | ~0.17 GB |
| Activations, seq 2048, batch 1, gradient checkpointing | ~2–3 GB |
| CUDA context, kernels, fragmentation | ~1.5–2 GB |
| **Total** | **~9–11 GB** |

Headroom is deliberate — free Colab occasionally hands out a T4 with a few
hundred MB already consumed.

**If you hit OOM**, in this order: `--max-seq-len 1792` (the longest example is
1,602 tokens, so nothing truncates), then `--lora-r 8`, then
`--grad-accum 16 --batch-size 1`.

### Disk — ~20 GB, against Colab's ~78 GB

| Item | Size |
|---|---:|
| Qwen3-8B weights downloaded from the Hub | ~16.4 GB |
| Dependencies on top of Colab's preinstalled stack | ~2 GB |
| Training data (`train` 14 MB + `validation` 3 MB) | 17 MB |
| Checkpoints (3 kept × ~170 MB) | ~0.5 GB |
| Final adapter | ~80–160 MB |

The 16.4 GB download is the slow part of the first run — allow 5–10 minutes.
The smoke test warms that cache, so the real run starts immediately.

### Steps and time

```
train examples          2,133
per_device_batch_size       1
gradient_accumulation       8
effective batch size        8

optimiser steps/epoch     266
epochs                      2
total optimiser steps     533
forward/backward passes 4,266
```

At roughly **2–2.5 s per forward/backward** on a T4 for an 8B model at 4-bit
with gradient checkpointing:

| | |
|---|---|
| One epoch | ~1.5 hours |
| Two epochs | **~3 hours** |
| Smoke test (8 steps) | ~10 min, mostly the model download |

> **Free Colab sessions get reclaimed.** Checkpoints are written every 50 steps.
> Keep the working directory on Google Drive (cell 3) and a dead session costs
> minutes, not hours — re-run with `--resume`.

---

## The detail that matters most

Each training example is about **1,473 prompt tokens and 45 completion
tokens**, and the prompt is ~97 % database schema, byte-identical across all
2,133 examples.

Train on the full sequence and roughly **97 % of the gradient signal goes into
reproducing a schema the model is handed at inference time.** It would spend its
capacity memorising `CREATE TABLE customers (...)` while learning very little
about writing SQL — and the loss curve would look excellent throughout, because
predicting a constant is easy.

`train_qlora.py` therefore sets the prompt's labels to `-100` and computes loss
**on the SQL only**. The script prints the supervised token share at startup;
it should read around 3 %.

This is why the config uses a fairly high learning rate (2e-4) and only 2
epochs: the real supervised signal is ~96 k tokens, not ~3.2 M.

---

## Exact commands

### In Colab (recommended)

1. Open `Phase9_QLoRA_Qwen3_8B_Text2SQL.ipynb` in Colab
2. `Runtime -> Change runtime type -> T4 GPU`
3. Run cell 1 (GPU check) — stop if it shows no GPU
4. Run cell 2 (install), then **`Runtime -> Restart session`**
5. Run cell 3 (mount Drive, choose an upload option)
6. Run cell 4 — dataset hash check, must print `MATCH` twice
7. Run cell 5 — smoke test
8. Run cell 6 — full training

### Or from a terminal on any CUDA machine

```bash
pip install -r training/qlora/requirements-colab.txt

# prove the pipeline works: 8 steps on 32 examples
python training/qlora/train_qlora.py --smoke --output-dir outputs/smoke

# the real run
python training/qlora/train_qlora.py \
    --train-file dataset/sft/train.jsonl \
    --val-file dataset/sft/validation.jsonl \
    --output-dir outputs/qwen3-8b-text2sql-qlora \
    --epochs 2

# after a disconnect
python training/qlora/train_qlora.py --resume \
    --output-dir outputs/qwen3-8b-text2sql-qlora --epochs 2
```

---

## Getting data in and the adapter out

### Upload — you need three files, 17 MB total

`dataset/sft/train.jsonl`, `dataset/sft/validation.jsonl`,
`training/qlora/train_qlora.py`. Not the whole repository.

| Option | When to use it |
|---|---|
| **A. `git clone`** | The repo is on GitHub. Cleanest and most reproducible. |
| **B. `files.upload()`** | Nothing is pushed anywhere. Cell 3 handles the file placement. |
| **C. Copy from Drive** | Upload once to Drive by hand, reuse across sessions. |

Cell 4 then verifies the content hashes against the Phase 7 freeze
(`train 7804d4386c9b1004`, `validation 43a469080798f5aa`) and **aborts on
mismatch**. If the training data is not the audited data, any later comparison
against the frozen 10.82 % baseline is meaningless — so this is a hard gate, not
a warning.

### Download — the adapter is ~80–160 MB

```
outputs/qwen3-8b-text2sql-qlora/
├── final_adapter/           <- adapter_model.safetensors + adapter_config.json
├── training_config.json     <- every hyperparameter, GPU, dtype, dataset stats
├── training_metrics.json    <- loss curve, runtime, final eval
└── checkpoint-*/            <- resumable checkpoints
```

The 16 GB base model is **not** included and is not needed: Phase 10 loads
`Qwen/Qwen3-8B` and applies the adapter on top.

Place the downloaded contents in `models/finetuned/` in the project repo.

---

## Configuration

| Setting | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-8B` @ `b968826d9c46` | Revision pinned so a repo update cannot silently change what was trained |
| Quantisation | 4-bit NF4 + double quant | Fits 16 GB; NF4 is information-theoretically better than int4 for normally-distributed weights |
| Compute dtype | fp16 on T4, bf16 on Ampere+ | T4 is Turing (7.5) and has no bf16 |
| Attention | SDPA on T4, FlashAttention-2 on Ampere+ | FA2 needs 8.0+ |
| LoRA rank | 16 (alpha 32, dropout 0.05) | Standard starting point; adapter stays ~80 MB |
| Target modules | q, k, v, o, gate, up, down | All attention and MLP projections — the usual QLoRA recipe |
| Learning rate | 2e-4, cosine, 3 % warmup | ~10× a full fine-tune, normal for LoRA |
| Optimiser | paged 8-bit AdamW | "Paged" survives VRAM spikes instead of OOM-ing |
| Max seq len | 2048 | Longest example is 1,602 tokens; nothing truncates |
| Epochs | 2 | Only ~96 k supervised tokens; more epochs risks memorising 2,133 examples |
| Seed | 20260808 | Same seed used throughout the project |

The script auto-detects GPU capability and selects dtype and attention
accordingly, so the same file runs unchanged on a T4 or an A100.

---

## What this deliberately does not do

- **No CPU fallback.** Missing CUDA exits with instructions. An 8B fine-tune on
  CPU is not slow, it is infeasible.
- **No benchmark run.** The 453-example test set is Phase 10, and must go
  through the same harness that produced the frozen baseline.
- **No dataset modification.** `dataset/sft/*` is read-only; `test.jsonl` is
  never opened.
- **No prompt changes.** The training data already embeds the frozen prompt
  (`8288e41a496531a9`) and schema (`d03619e711661bc5`). Training must not
  reformat them, or Phase 10 would measure prompt engineering rather than
  fine-tuning.

---

## Honest caveat

**This script has not been executed end to end.** There is no GPU on the
development machine, so what has been verified locally is: Python syntax, the
GPU gate firing correctly, every notebook cell parsing, and the dataset
integrity check reproducing the frozen hashes exactly.

The `--smoke` flag exists precisely for this reason. Run it first — 8 steps on
32 examples, about 10 minutes — and only start the 3-hour run once it completes
and writes an adapter. Anything wrong with the environment, quantisation, or
label masking will surface there rather than two hours into the real run.
