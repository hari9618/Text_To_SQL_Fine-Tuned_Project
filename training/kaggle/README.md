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
