"""QLoRA fine-tuning of Qwen3-8B for Text-to-SQL.

Phase 8/9 (CLAUDE.md numbering). Designed to run on a **free Google Colab T4**.

Run it, do not read it, if you only want the commands — see README.md.

--------------------------------------------------------------------------
Why this script masks the loss to the completion only
--------------------------------------------------------------------------

Each training example is roughly **1,473 prompt tokens and 45 completion
tokens**. The prompt is ~97 % database schema, byte-identical across all 2,133
examples.

Train on the whole sequence and ~97 % of the gradient signal goes into
reproducing a schema the model is *given* at inference time. The model would
spend its capacity memorising `CREATE TABLE customers (...)` and learn almost
nothing about writing SQL. Worse, the loss curve would look excellent while the
thing you care about barely moves.

So labels for the prompt span are set to -100 and only the SQL is trained on.
This is the single most consequential detail in this file.

--------------------------------------------------------------------------
Why 4-bit, and why these dtypes
--------------------------------------------------------------------------

*QLoRA* = the base model frozen and quantised to 4-bit NF4, with small
trainable low-rank adapters in fp16 alongside. Qwen3-8B in bf16 needs ~16.4 GB
of weights alone, which does not fit a 16 GB T4 with room to train. At 4-bit it
is ~4.4 GB, leaving room for activations and optimiser state.

The T4 is Turing (compute 7.5): it has **no bf16 support and no Flash Attention
2** (both need Ampere, 8.0+). The script detects this and selects fp16 +
SDPA attention automatically rather than crashing halfway through loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# One GPU, deliberately. On a 2xT4 Kaggle session the Trainer would wrap the
# model in nn.DataParallel, which scatters every tensor argument along dim 0 -
# including the 1-D position index the label-only loss passes as
# `logits_to_keep`. The v1 run measured no real speed-up from the second card
# either (134 steps at ~131 s vs. 266 at ~75 s). Set before torch is imported;
# an explicit value in the environment wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --------------------------------------------------------------------------
# 1. GPU gate — before importing anything heavy
# --------------------------------------------------------------------------

def require_gpu() -> dict[str, Any]:
    """Refuse to run without CUDA.

    Fine-tuning an 8B model on CPU is not slow, it is infeasible: a single
    optimisation step would take minutes and the full run months. Failing here
    with an explanation is far kinder than letting somebody discover that after
    an hour of watching a progress bar that never moves.
    """
    try:
        import torch
    except ImportError:
        sys.exit(
            "\n[FATAL] PyTorch is not installed.\n"
            "  In Colab:  !pip install -r training/qlora/requirements-colab.txt\n"
        )

    if not torch.cuda.is_available():
        sys.exit(
            "\n" + "=" * 68 + "\n"
            "[FATAL] No CUDA GPU detected. Refusing to train on CPU.\n"
            "=" * 68 + "\n"
            "QLoRA fine-tuning of an 8B model requires a GPU. On CPU a single\n"
            "step takes minutes; the full run would take months.\n\n"
            "In Google Colab:\n"
            "    Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU\n"
            "    then Runtime -> Restart session, and re-run.\n\n"
            "Verify with:  !nvidia-smi\n"
        )

    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "supports_bf16": props.major >= 8,
        "supports_flash_attn2": props.major >= 8,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    print("=" * 68)
    print("GPU CHECK")
    print("=" * 68)
    for k, v in info.items():
        print(f"  {k:<22}{v}")

    if info["total_vram_gb"] < 14:
        print(f"\n[WARNING] {info['total_vram_gb']} GB VRAM is below the ~14 GB this "
              "config expects.\n          Reduce --max-seq-len or --lora-r if you hit OOM.")
    if not info["supports_bf16"]:
        print("\n  Turing-class GPU: using fp16 compute and SDPA attention "
              "(bf16 and FlashAttention-2 need Ampere or newer).")
    print()
    return info


# --------------------------------------------------------------------------
# 2. Configuration
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Every knob, recorded verbatim into the output directory.

    A fine-tune that cannot be reproduced is an anecdote. This object is
    serialised next to the adapter so Phase 10 can state exactly what produced
    the weights it is evaluating.
    """

    model_id: str = "Qwen/Qwen3-8B"
    # Pinned so a repo update cannot silently change what was trained.
    model_revision: str = "b968826d9c46"

    # Resolved at construction time so Kaggle's /kaggle/input mount is picked
    # up automatically; overridable with --train-file / --val-file.
    train_file: str = field(default_factory=lambda: _default_paths()[0])
    val_file: str = field(default_factory=lambda: _default_paths()[1])
    output_dir: str = field(default_factory=lambda: _default_paths()[2])

    max_seq_len: int = 2048          # longest example is 1,602 tokens
    num_epochs: float = 2.0
    per_device_batch_size: int = 1   # 16 GB T4 will not hold more at 2048
    gradient_accumulation_steps: int = 8   # effective batch 8
    learning_rate: float = 2e-4      # standard for LoRA; ~10x full fine-tuning
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 0.3
    seed: int = 20260808

    # LoRA. r=16 is the usual starting point: large enough to learn a new
    # output format, small enough that the adapter stays ~80 MB.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Quantisation
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 5
    save_total_limit: int = 3

    train_on_completion_only: bool = True
    gradient_checkpointing: bool = True

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lora_target_modules"] = list(self.lora_target_modules)
        return d


# --------------------------------------------------------------------------
# 3. Dataset
# --------------------------------------------------------------------------

def _default_paths() -> tuple[str, str, str]:
    """Locate the dataset and output directory for the current host.

    Kaggle mounts input datasets read-only under /kaggle/input/<slug>/ and
    expects results in /kaggle/working/. Detecting that automatically is what
    lets the same script run unattended there with no arguments at all --
    which is the whole point of using Kaggle over Colab.
    """
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in sorted(kaggle_input.iterdir()):
            train = next(candidate.rglob("train.jsonl"), None)
            val = next(candidate.rglob("validation.jsonl"), None)
            if train and val:
                return (str(train), str(val),
                        "/kaggle/working/qwen3-8b-text2sql-qlora")
    return ("dataset/sft/train.jsonl", "dataset/sft/validation.jsonl",
            "outputs/qwen3-8b-text2sql-qlora")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(
            f"\n[FATAL] Dataset not found: {path}\n"
            "  Upload dataset/sft/ to Colab, or set --train-file / --val-file.\n"
            "  See training/qlora/README.md for the upload options.\n"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_tokenised(records: list[dict], tokenizer, max_seq_len: int,
                    completion_only: bool) -> tuple[list[dict], dict[str, Any]]:
    """Apply the chat template and mask the prompt out of the labels.

    The prompt is rendered with ``add_generation_prompt=True`` so it ends
    exactly where the model must begin writing — the same boundary that exists
    at inference. The completion is then appended and only its tokens carry a
    label.

    ``enable_thinking=False`` matches the frozen baseline, which ran with
    Qwen3's reasoning mode disabled. Training with thinking enabled and
    evaluating with it disabled would compare two different inference modes.
    """
    examples: list[dict] = []
    truncated = 0
    prompt_lens: list[int] = []
    completion_lens: list[int] = []

    for record in records:
        messages = record["messages"]
        prompt_msgs, completion = messages[:-1], messages[-1]["content"]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        completion_ids = completion_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids
        if len(input_ids) > max_seq_len:
            truncated += 1
            # Trim the prompt, never the answer: a truncated SQL target would
            # teach the model to stop mid-statement.
            overflow = len(input_ids) - max_seq_len
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + completion_ids

        if completion_only:
            labels = [-100] * len(prompt_ids) + completion_ids[:]
        else:
            labels = input_ids[:]

        examples.append({"input_ids": input_ids, "labels": labels,
                         "attention_mask": [1] * len(input_ids)})
        prompt_lens.append(len(prompt_ids))
        completion_lens.append(len(completion_ids))

    supervised = sum(completion_lens)
    total = sum(prompt_lens) + supervised
    stats = {
        "examples": len(examples),
        "truncated": truncated,
        "mean_prompt_tokens": round(sum(prompt_lens) / len(prompt_lens), 1),
        "mean_completion_tokens": round(sum(completion_lens) / len(completion_lens), 1),
        "max_total_tokens": max(p + c for p, c in zip(prompt_lens, completion_lens)),
        "supervised_token_share_pct": round(100.0 * supervised / total, 2),
    }
    return examples, stats


@dataclass
class Collator:
    """Pad a batch and keep label padding at -100 so it is ignored by the loss."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [self.label_pad_id] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------

def _package_versions() -> dict[str, str]:
    """Resolved versions of everything that affects the numerics."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                "datasets", "trl", "tokenizers"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


def _adapt_kwargs(callable_obj, desired: dict, renames: dict[str, list[str]] | None = None
                  ) -> tuple[dict, list[str]]:
    """Keep only the keyword arguments the installed library actually accepts.

    transformers changes its API across major versions -- `warmup_ratio` and
    `evaluation_strategy` have both come and gone, and `torch_dtype` became
    `dtype`. Pinning a version would avoid this but breaks against Colab's
    fast-moving base image (that is how the bitsandbytes/Triton failure
    happened). Adapting to the installed signature instead makes the script
    version-agnostic, and it reports what it dropped rather than silently
    ignoring a setting that affects training.
    """
    import inspect

    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return desired, []

    # A function declared as (*args, **kwargs) has no named parameters to match
    # against, so filtering would drop everything. `from_pretrained` is exactly
    # that shape -- filtering it silently removed `quantization_config`, the
    # model loaded in full precision, and Colab killed the runtime on CPU RAM.
    # If the callable accepts **kwargs, pass everything through untouched.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return desired, []

    accepted = set(params)

    renames = renames or {}
    kwargs, dropped = {}, []
    for key, value in desired.items():
        if key in accepted:
            kwargs[key] = value
            continue
        alternative = next((a for a in renames.get(key, []) if a in accepted), None)
        if alternative:
            kwargs[alternative] = value
        else:
            dropped.append(key)
    return kwargs, dropped


def make_label_only_trainer(base_trainer_cls):
    """A Trainer that only ever computes logits at supervised positions.

    With completion-only loss, ~97 % of every sequence is prompt whose labels
    are -100. The stock forward still projects *every* position through the
    151,936-way lm_head and cross-entropy - for a 1,944-token v2 example that
    is ~590 MB of fp16 logits, ~1.2 GB once upcast to fp32, and as much again
    in temporaries. That was the allocation that ran the v2 run out of memory
    at step 34/136 (13.4 GB in use on a 14.6 GB T4).

    Transformers' ``logits_to_keep`` accepts a tensor of positions, so the
    lm_head is applied to just the ~45 positions that predict SQL tokens. The
    loss is mathematically identical to the full computation over the same
    label mask; only the wasted work is gone. Falls back to full logits, then
    slicing, if a model does not accept the argument.
    """

    class LabelOnlyLogitsTrainer(base_trainer_cls):
        _supports_keep = True

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            import torch
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            # transformers >= 5 injects skip_logits=True into the *eval* inputs
            # when use_liger_kernel is on and no metrics are computed
            # (Trainer.prediction_step). Liger would then refuse to run
            # without labels - the exact failure of v2 attempt 2. This loss
            # always wants the (sliced) logits, so the flag is dropped here
            # and re-sent as False below.
            inputs.pop("skip_logits", None)
            # Position t predicts token t+1, so keep t where labels[:, t+1]
            # is supervised for any sample in the (padded) batch.
            shifted = labels[:, 1:]
            keep = (shifted != -100).any(dim=0).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:              # defensive: nothing to learn
                keep = torch.tensor([shifted.shape[1] - 1], device=labels.device)

            outputs = None
            if self._supports_keep:
                # Liger's patched forward decides on its own whether to skip
                # the logits (it expects to fuse the loss itself), and in eval
                # mode with no labels it skips them and raises. The v2 run
                # died exactly there, at the first evaluation, step 50. Say
                # explicitly that the logits are wanted; a model whose forward
                # has no such argument gets the call without it.
                for extra in ({"skip_logits": False}, {}):
                    try:
                        outputs = model(**inputs, logits_to_keep=keep, **extra)
                        break
                    except TypeError as exc:
                        if "skip_logits" in str(exc) and extra:
                            continue
                        self._supports_keep = False
                        print("  [note] model rejects logits_to_keep; using full logits")
                        break
            if outputs is None:
                outputs = model(**inputs)
                outputs.logits = outputs.logits[:, keep, :]

            logits = outputs.logits.float()
            targets = shifted[:, keep]
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            if num_items_in_batch is not None:
                # Recent Trainers pass the supervised-token count for the
                # whole accumulation window and skip their own division by
                # the accumulation steps; the loss must be token-averaged
                # over that window here, or gradients come out 16x too large.
                loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100,
                                       reduction="sum") / num_items_in_batch
            else:
                loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
            return (loss, outputs) if return_outputs else loss

    return LabelOnlyLogitsTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-8B for Text-to-SQL")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in output-dir")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="disable gradient checkpointing: faster, "
                             "more VRAM for activations")
    parser.add_argument("--attn-only-lora", action="store_true",
                        help="adapt attention projections only (drop MLP)")
    parser.add_argument("--max-train-examples", type=int, default=None,
                        help="train on the first N examples only")
    parser.add_argument("--full-logits", action="store_true",
                        help="compute logits at every position (the v1 behaviour); "
                             "needs ~3 GB more VRAM at 1,900 tokens")
    parser.add_argument("--smoke", action="store_true",
                        help="8 optimiser steps on a tiny slice, to prove the "
                             "pipeline works before committing hours to it")
    args = parser.parse_args()

    gpu = require_gpu()

    cfg = TrainConfig()
    for attr, value in [
        ("train_file", args.train_file), ("val_file", args.val_file),
        ("output_dir", args.output_dir), ("num_epochs", args.epochs),
        ("max_seq_len", args.max_seq_len), ("lora_r", args.lora_r),
        ("learning_rate", args.learning_rate),
        ("per_device_batch_size", args.batch_size),
        ("gradient_accumulation_steps", args.grad_accum),
    ]:
        if value is not None:
            setattr(cfg, attr, value)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    compute_dtype = torch.bfloat16 if gpu["supports_bf16"] else torch.float16
    attn_impl = "flash_attention_2" if gpu["supports_flash_attn2"] else "sdpa"

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----------------------------------------------------------
    print("=" * 68); print("DATASET"); print("=" * 68)
    train_records = load_jsonl(Path(cfg.train_file))
    val_records = load_jsonl(Path(cfg.val_file))
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.attn_only_lora:
        cfg.lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    if args.max_train_examples:
        train_records = train_records[:args.max_train_examples]
        print(f"  limited to first {len(train_records):,} training examples")

    if args.smoke:
        train_records, val_records = train_records[:32], val_records[:8]
        cfg.num_epochs, cfg.save_steps, cfg.eval_steps = 1.0, 4, 4
        print("  SMOKE MODE: 32 train / 8 validation records")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, revision=cfg.model_revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ex, train_stats = build_tokenised(
        train_records, tokenizer, cfg.max_seq_len, cfg.train_on_completion_only)
    val_ex, val_stats = build_tokenised(
        val_records, tokenizer, cfg.max_seq_len, cfg.train_on_completion_only)

    for name, s in (("train", train_stats), ("validation", val_stats)):
        print(f"  {name:<11}{s['examples']:>6} examples | "
              f"prompt {s['mean_prompt_tokens']:.0f} | completion "
              f"{s['mean_completion_tokens']:.0f} | max {s['max_total_tokens']} | "
              f"truncated {s['truncated']}")
    print(f"\n  Loss is computed on {train_stats['supervised_token_share_pct']} % of "
          f"tokens (the SQL only).")
    print("  The other ~97 % is schema the model is handed at inference; training "
          "on it\n  would spend capacity memorising boilerplate.\n")

    # ---- model ---------------------------------------------------------
    print("=" * 68); print("MODEL"); print("=" * 68)
    print(f"  {cfg.model_id} @ {cfg.model_revision}")
    print(f"  4-bit {cfg.bnb_4bit_quant_type}, compute dtype {compute_dtype}, "
          f"attention {attn_impl}")
    print("  first run downloads ~16 GB of weights; allow 5-10 minutes\n")

    quant = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # Passed directly: from_pretrained accepts **kwargs, so nothing may be
    # filtered here. Dropping quantization_config would load 16 GB of fp16
    # weights into 12 GB of Colab RAM and kill the runtime.
    load_kwargs = {
        "revision": cfg.model_revision,
        "quantization_config": quant,
        "device_map": {"": 0},
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    # `torch_dtype` was renamed to `dtype`; newer versions warn, older ones
    # require the old name. Try the new spelling and fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=compute_dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=compute_dtype, **load_kwargs)

    assert getattr(model, "is_loaded_in_4bit", False) or         any("4bit" in type(m).__name__.lower() for m in model.modules()),         "model did not load in 4-bit - quantization_config was not applied"
    print(f"  4-bit load confirmed | "
          f"VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    peft_model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none", task_type="CAUSAL_LM",
    ))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  trainable {trainable:,} of {total:,} params "
          f"({100.0 * trainable / total:.3f} %)")
    print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")

    # ---- train ---------------------------------------------------------
    steps_per_epoch = max(
        1, len(train_ex) // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps))
    total_steps = int(steps_per_epoch * cfg.num_epochs)
    print("=" * 68); print("TRAINING"); print("=" * 68)
    print(f"  effective batch  {cfg.per_device_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  optimiser steps  {total_steps} ({steps_per_epoch}/epoch x {cfg.num_epochs})")
    print(f"  checkpoint every {cfg.save_steps} steps -> {out_dir}\n")

    desired_args = {
        "output_dir": str(out_dir),
        "num_train_epochs": cfg.num_epochs,
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "per_device_eval_batch_size": cfg.per_device_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        # Newer transformers defaults to reentrant checkpointing, which can
        # silently fail to apply under PEFT. Being explicit keeps activation
        # memory where it should be.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Qwen3 has a 151,669-token vocabulary, so the logits tensor for a
        # 1,500-token sequence is ~910 MB once accelerate upcasts it to fp32 --
        # and cross-entropy needs temporaries on top. That is what ran a T4 out
        # of memory at the loss step. Liger fuses the projection and the loss
        # so the full logits are never materialised.
        "use_liger_kernel": True,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "fp16": not gpu["supports_bf16"],
        "bf16": gpu["supports_bf16"],
        "optim": "paged_adamw_8bit",
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_strategy": "steps",
        "eval_steps": cfg.eval_steps,
        "save_total_limit": cfg.save_total_limit,
        "report_to": [],
        "seed": cfg.seed,
        "dataloader_pin_memory": False,
    }
    ta_kwargs, dropped = _adapt_kwargs(
        TrainingArguments,
        desired_args,
        renames={
            "eval_strategy": ["evaluation_strategy"],
            "evaluation_strategy": ["eval_strategy"],
            "lr_scheduler_type": ["lr_scheduler"],
        },
    )

    # warmup_ratio was removed in some versions; express it in steps instead
    # so the warmup still happens rather than silently vanishing.
    if "warmup_ratio" in dropped:
        import inspect
        if "warmup_steps" in inspect.signature(TrainingArguments).parameters:
            ta_kwargs["warmup_steps"] = max(1, int(cfg.warmup_ratio * total_steps))
            dropped.remove("warmup_ratio")
            print(f"  warmup_ratio unsupported -> warmup_steps="
                  f"{ta_kwargs['warmup_steps']}")

    if "use_liger_kernel" in dropped:
        print("  [warning] use_liger_kernel unsupported by this "
              "transformers version.")
        print("            Loss will materialise full logits; "
              "expect OOM above ~1200 tokens on a 16 GB card.")
    if dropped:
        print(f"  [note] settings unsupported by this transformers version "
              f"and dropped: {dropped}")

    trainer_cls = Trainer if args.full_logits else make_label_only_trainer(Trainer)
    print(f"  loss impl        {'full logits' if args.full_logits else 'label-only logits (lm_head on supervised positions)'}")
    trainer = trainer_cls(
        model=peft_model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=Dataset.from_list(train_ex),
        eval_dataset=Dataset.from_list(val_ex),
        data_collator=Collator(pad_token_id=tokenizer.pad_token_id),
    )

    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    elapsed = time.time() - started

    # ---- save ----------------------------------------------------------
    print("\n" + "=" * 68); print("SAVING"); print("=" * 68)
    adapter_dir = out_dir / "final_adapter"
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  adapter   {adapter_dir}")

    metrics = {
        "train_runtime_s": round(elapsed, 1),
        "train_runtime_h": round(elapsed / 3600, 2),
        "train_loss": result.metrics.get("train_loss"),
        "total_optimizer_steps": total_steps,
        "log_history": trainer.state.log_history,
    }
    try:
        metrics["final_eval"] = trainer.evaluate()
    except Exception as exc:
        metrics["final_eval_error"] = str(exc)[:200]

    (out_dir / "training_config.json").write_text(
        json.dumps({
            "config": cfg.as_dict(),
            "gpu": gpu,
            "compute_dtype": str(compute_dtype),
            "attn_implementation": attn_impl,
            "loss_impl": "full_logits" if args.full_logits else "label_only_logits",
            "dataset_stats": {"train": train_stats, "validation": val_stats},
            "trainable_params": trainable,
            "total_params": total,
            "smoke_run": args.smoke,
            # Packages are loosely pinned because Colab's base image moves
            # fast; the exact resolved versions are captured here instead,
            # so a future run can reproduce or diagnose this one.
            "package_versions": _package_versions(),
        }, indent=2), encoding="utf-8")
    (out_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  config    {out_dir / 'training_config.json'}")
    print(f"  metrics   {out_dir / 'training_metrics.json'}")
    # Throughput expressed so it extrapolates. The first published estimate
    # for this config was 4x optimistic; measure, do not guess.
    sec_per_step = elapsed / max(total_steps, 1)
    full_epoch_steps = max(1, len(train_ex) // (
        cfg.per_device_batch_size * cfg.gradient_accumulation_steps))
    metrics["sec_per_optimizer_step"] = round(sec_per_step, 2)
    print()
    print(f"  runtime          {elapsed / 3600:.2f} h")
    print(f"  sec/optim step   {sec_per_step:.1f}")
    print(f"  -> 1 full epoch  {full_epoch_steps * sec_per_step / 3600:.1f} h "
          f"({full_epoch_steps} steps over {len(train_ex):,} examples)")
    print(f"  final loss {metrics['train_loss']}")
    print("\nDownload the whole output directory. Phase 10 evaluates the adapter;\n"
          "this script never touches dataset/test/test.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
