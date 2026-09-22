# Unattended QLoRA training on Kaggle

Kaggle is used instead of Colab for one reason: **"Save & Run All" executes the
notebook detached.** Upload once, press one button, close the browser, and come
back to a finished adapter. Colab cannot do that — its sessions require a live
browser tab and get reclaimed unpredictably.

| | Colab free | Kaggle |
|---|---|---|
| Max session | ~4 h, reclaimed without warning | **12 h** |
| Unattended | No | **Yes (Save & Run All)** |
| Input files | wiped on every restart | **persist as a Dataset** |
| Weekly GPU | unmetered but throttled | 30 h |

## One-time setup (~10 minutes)

### 1. Create the Kaggle Dataset

1. https://www.kaggle.com/datasets -> **New Dataset**
2. Drag in the three files from `Downloads/kaggle-upload/text2sql-sft/`:
   `train.jsonl`, `validation.jsonl`, `train_qlora.py`
3. Title it **text2sql-sft**, set visibility Private, **Create**

Uploading the training script *as part of the dataset* is deliberate — it makes
the notebook self-contained, so a commit run has no external dependency that
could vanish mid-run.

### 2. Create the notebook

1. https://www.kaggle.com/code -> **New Notebook** -> **File -> Import Notebook**
2. Upload `kaggle_qlora_qwen3_8b.ipynb`
3. In the right-hand panel:
   - **Input -> Add Input** -> Datasets -> your `text2sql-sft`
   - **Accelerator -> GPU T4 x2**
   - **Internet -> On**  (required to download the model)

Internet off is the most common cause of a Kaggle commit failing at the model
download, several minutes in.

### 3. Launch

**Save Version -> Save & Run All (Commit) -> Save**

Close the tab. Kaggle emails you when it finishes. Progress is visible any time
under your notebook's **Versions** tab.

## Expected

| | |
|---|---|
| Runtime | ~5.5 h for 1 epoch (266 optimiser steps at ~75 s) |
| VRAM | ~8.2 GB of 15 GB |
| Output | `/kaggle/working/adapter.zip`, ~80-160 MB |

Well inside the 12 h commit limit, with room for the run to be slower than
measured.

## Getting the adapter

Notebook -> **Output** tab -> download `adapter.zip`. Unzip into
`models/finetuned/` in the project repo. The 16 GB base model is not included
and is not needed: Phase 10 loads `Qwen/Qwen3-8B` and applies the adapter.

## If a run is cut short

Checkpoints persist in the notebook output. Add `--resume` to the training
command in cell 4 and commit again; it continues from the last checkpoint.

## Config

The settings are the ones proven on real hardware during the Colab smoke test:
4-bit NF4, LoRA r=16 on all attention and MLP projections, batch 1 with
gradient accumulation 8, gradient checkpointing on, fp16 (T4 has no bf16).

`--batch-size 2 --no-grad-ckpt` was tried and ran out of memory: without
checkpointing, ~1,500-token activations for two sequences exceed a T4. Batch 2
*with* checkpointing is untested and may work — but the default is the config
that is known to run.

---

## v2 — the second iteration (benchmark v2, prompt v2)

Three Kaggle runs, in this order. All free-tier; total GPU time ~8 h, inside
the 30 h weekly quota.

| # | notebook | inputs | output | time |
|---|---|---|---|---|
| 1 | `kaggle_qlora_qwen3_8b_v2.ipynb` | dataset **`text2sql-sft-v2`** (from `Downloads/text2sql-sft-v2.zip`) | `adapter.zip` | ~6–7 h |
| 2 | `kaggle_generate_finetuned.ipynb` | run 1's output **+** dataset **`text2sql-evalpack-v2`** (from `Downloads/text2sql-evalpack-v2.zip`) | `predictions_v2_final.jsonl` | ~35 min |
| 3 | `kaggle_generate_finetuned.ipynb` | **only** dataset `text2sql-evalpack-v2` — no adapter | `predictions_base_v2_final.jsonl` | ~35 min |

Run 3 is the v2 base-model row. With no adapter in its inputs the notebook
loads the base model alone, same 4-bit load and greedy decoding, so the v2
base and fine-tuned rows are generated on identical hardware. (The v1 base
row came through Hugging Face inference; both accounts' monthly inference
credits are exhausted, and Kaggle is free.)

Runs 2 and 3 can be the same notebook committed twice with a different Input
panel. The output filename says which it was; the header inside says so too,
and the scorer routes on it.

Then, on the laptop:

```powershell
env\Scripts\python.exe scripts/score_finetuned.py --version v2 --predictions "C:\Users\dell\Downloads\predictions_base_v2_final.jsonl"
env\Scripts\python.exe scripts/score_finetuned.py --version v2 --predictions "C:\Users\dell\Downloads\predictions_v2_final.jsonl"
```

Unzip run 1's `adapter.zip` into `models/finetuned_v2/` first so the scorer
can record the training config beside the result.

### v2, attempt 1: out of memory at step 34 — fixed

The first v2 commit died with `CUDA out of memory` (13.4 GB of 14.6 GB in
use) at step 34/136, training otherwise healthy. v2 prompts are ~330 tokens
longer than v1 and the T4 had no headroom left.

Fix, in `train_qlora.py`: the lm_head and cross-entropy now run only on the
~45 supervised SQL positions per sequence (`logits_to_keep`), not on all
~1,900 — the prompt positions are masked out of the loss anyway, and their
151k-vocabulary logits plus fp32 copies were ~3 GB per step. The loss is
identical. The script also pins one GPU (Kaggle's 2-GPU `DataParallel`
would scatter the position index) and the notebook accumulates 16
micro-batches, so effective batch stays 16 as in v1.

**Re-uploading is required**: the notebook reads `train_qlora.py` from the
`text2sql-sft-v2` dataset, so upload a **new version** of that dataset with
the updated script (Dataset page -> New Version -> replace `train_qlora.py`),
then re-import `kaggle_qlora_qwen3_8b_v2.ipynb` and commit.

### v2, attempt 2: failed at the first evaluation — fixed

Attempt 2 cleared the memory limit (step 50 with ~5 GB free, loss 1.75 ->
0.005) and died at the first evaluation: Liger's patched forward skips the
logits in eval mode when no labels are passed, and the label-only loss passes
none. `compute_loss` now calls the model with `skip_logits=False`.

Kaggle discards `/kaggle/working` on a failed commit, which is how two
attempts each lost their step-50 checkpoint. The notebook no longer raises on
a training failure: it writes `TRAINING_FAILED.txt`, prints a banner, keeps
the checkpoints, and packages `checkpoints.zip`-style output instead of
`adapter.zip`. Attach that failed version as an Input of the next run and it
resumes from the latest checkpoint.
