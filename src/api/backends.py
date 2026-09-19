"""Choosing which model the API serves.

Phase 12.

Three backends, selected by the ``MODEL_BACKEND`` environment variable:

| backend | what it serves | where it runs |
| --- | --- | --- |
| ``hf`` | base Qwen3-8B via HF Inference Providers | anywhere, needs `HF_TOKEN` |
| ``local`` | base + the Phase 9 LoRA adapter, 4-bit | a CUDA host |
| ``stub`` | canned SQL, no model at all | tests and CI |

**The default is ``hf``, which serves the base model, not the fine-tuned one.**
That is a hardware constraint, not a design preference: this project's
development machine has 8 GB of RAM and no GPU, and the adapter needs one. The
accuracy the API delivers therefore depends on which backend is configured —
`hf` is the 10.82 % baseline, `local` is the 50.99 % fine-tuned model. Serving
the adapter in production means running ``local`` on a GPU host, or pushing the
adapter to a vLLM server with LoRA enabled.

``stub`` exists so the pipeline, the validation layer, the repair loop and the
HTTP contract can all be tested without a GPU, a network call, or an API
credit. Every test in ``tests/test_api.py`` uses it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from src.model.base_model import (
    DEFAULT_MODEL_ID,
    GenerationResult,
    InferenceParams,
    TextToSQLModel,
    extract_sql,
)

DEFAULT_BACKEND = "hf"


class StubModel(TextToSQLModel):
    """Returns canned SQL. No network, no weights, no GPU.

    ``responses`` maps a question to the SQL to return; ``default`` covers
    anything unmapped. ``repair_responses`` lets a test drive the repair path
    deterministically: the first call fails, the repair succeeds.
    """

    model_id = "stub/canned-sql"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "SELECT 1",
        repair_responses: dict[str, str] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self._repair = repair_responses or {}
        self.calls: list[str] = []
        self.repair_calls: list[list[dict[str, str]]] = []

    def generate(self, question: str, schema: str) -> GenerationResult:
        self.calls.append(question)
        sql = self._responses.get(question, self._default)
        return GenerationResult(sql=sql, raw_output=sql, latency_ms=1.0, ok=True,
                                finish_reason="stop", provider="stub")

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        self.repair_calls.append(messages)
        content = messages[-1]["content"]
        for needle, sql in self._repair.items():
            if needle in content:
                return GenerationResult(sql=sql, raw_output=sql, latency_ms=1.0,
                                        ok=True, provider="stub")
        return GenerationResult(sql=self._default, raw_output=self._default,
                                latency_ms=1.0, ok=True, provider="stub")

    def describe(self) -> dict[str, Any]:
        return {"kind": "stub", "model_id": self.model_id, "fine_tuned": False,
                "adapters": []}


class LocalAdapterModel(TextToSQLModel):
    """Base Qwen3-8B in 4-bit with the Phase 9 LoRA adapter applied.

    Requires CUDA plus torch, transformers, peft and bitsandbytes. Imported
    lazily so that merely importing this module does not drag ~2 GB of
    dependencies into a laptop that cannot use them.

    The chat template is loaded explicitly from the adapter directory and
    rendered with ``enable_thinking=False``. Both matter: the adapter ships no
    inline template, and omitting the empty think block feeds the model a
    format it was never trained on — which cost this project one wasted GPU run
    during Phase 10.
    """

    def __init__(
        self,
        adapter_dir: str,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = "b968826d9c46",
        params: InferenceParams | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "MODEL_BACKEND=local needs a CUDA GPU. This machine has none.\n"
                "Use MODEL_BACKEND=hf (serves the base model) or run the API on "
                "a GPU host."
            )

        self.model_id = model_id
        self.revision = revision
        self.adapter_dir = adapter_dir
        self.params = params or InferenceParams()
        self._torch = torch

        self._tok = AutoTokenizer.from_pretrained(adapter_dir)
        if not getattr(self._tok, "chat_template", None):
            with open(os.path.join(adapter_dir, "chat_template.jinja"),
                      encoding="utf-8") as fh:
                self._tok.chat_template = fh.read()
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quant,
            device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self._model = PeftModel.from_pretrained(base, adapter_dir)
        self._model.eval()

    def generate(self, question: str, schema: str) -> GenerationResult:
        from src.model.prompt import build_messages

        return self.generate_messages(build_messages(question, schema))

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        torch = self._torch
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        enc = self._tok(text, return_tensors="pt",
                        add_special_tokens=False).to(self._model.device)
        n_in = enc["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.params.max_tokens, do_sample=False,
                pad_token_id=self._tok.pad_token_id)
        elapsed = (time.perf_counter() - started) * 1000
        gen = out[0][n_in:]
        kept = [t for t in gen.tolist() if t != self._tok.pad_token_id]
        raw = self._tok.decode(gen, skip_special_tokens=True)
        return GenerationResult(
            sql=extract_sql(raw), raw_output=raw, latency_ms=elapsed, ok=True,
            finish_reason=("length" if len(kept) >= self.params.max_tokens
                           else "stop"),
            prompt_tokens=int(n_in), completion_tokens=len(kept),
            provider="local-qlora",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local_qlora_adapter",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "fine_tuned": True,
            "adapters": [{"path": self.adapter_dir, "method": "qlora"}],
        }


def build_model(backend: str | None = None) -> TextToSQLModel:
    """Construct the configured backend.

    Failures are loud and specific. A service that silently degrades to a
    different model than the operator intended would report accuracy figures
    that belong to something else.
    """
    name = (backend or os.getenv("MODEL_BACKEND", DEFAULT_BACKEND)).strip().lower()

    builders: dict[str, Callable[[], TextToSQLModel]] = {
        "stub": lambda: StubModel(),
        "hf": lambda: _build_hf(),
        "local": lambda: LocalAdapterModel(
            adapter_dir=os.getenv(
                "ADAPTER_DIR", "models/finetuned/final_adapter")),
    }
    if name not in builders:
        raise RuntimeError(
            f"unknown MODEL_BACKEND {name!r}; expected one of "
            f"{sorted(builders)}")
    return builders[name]()


def _build_hf() -> TextToSQLModel:
    from src.model.base_model import HFInferenceModel

    return HFInferenceModel(
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        # A failover chain. Phase 4 found nscale throttling and featherless
        # fine; 2026-09-19 found the reverse. Free providers flip, so the
        # service tries each in turn rather than depending on one.
        provider=os.getenv("HF_PROVIDER", "nscale,featherless-ai"),
        params=InferenceParams(temperature=0.0, max_tokens=512),
    )
