"""Text-to-SQL generation interface and its implementations.

Phase 4.

The interface is deliberately narrow: question + schema in, SQL + latency +
raw output out. Phase 10 swaps in a fine-tuned adapter behind the same
interface, and Phase 12's API calls it too. Nothing downstream needs to know
whether generation happened locally or over the network.

*Why the model runs remotely.* This project's development machine has 8 GB of
RAM and no NVIDIA GPU. Qwen3-8B needs roughly 16 GB in fp16; even 4-bit
quantisation would generate at a few tokens per second, putting the 453-example
test set well beyond a day of compute. The model is therefore served by
Hugging Face Inference Providers while the database stays local — only the
question and schema cross the network, never the data.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

# Genuinely permanent: a bad token or no access to the model. Retrying these
# only buries the real cause under a pile of generic failures.
FATAL_HTTP_STATUSES = (401, 403)

# 402 "Payment Required / depleted credits" looks permanent but behaves like a
# throttle on the free tier: bursts of concurrent requests trip it, and it
# clears on its own within minutes. Observed directly -- a run died on repeated
# 402s from one provider, then every provider served the same prompt fine a few
# minutes later. So it is retried, but slowly, rather than treated as fatal.
THROTTLE_HTTP_STATUSES = (402, 429, 503)
# Measured: the free-tier throttle clears after ~33s. A first retry at
# 20s fired too early, failed again, and then waited 45s -- ~65s per
# throttled call. Starting just past the measured recovery window
# roughly halves that.
THROTTLE_BACKOFF_S = (36, 45, 75, 120, 180)


@dataclass
class GenerationResult:
    """One model call: what came back, how long it took, and what went wrong."""

    sql: str
    raw_output: str
    latency_ms: float
    ok: bool = True
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider: str | None = None
    attempts: int = 1
    # True for errors that retrying cannot fix: exhausted credits, bad token,
    # forbidden model. Retrying these wastes minutes of backoff and, worse,
    # buries the real cause under a pile of generic failures.
    fatal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "raw_output": self.raw_output,
            "latency_ms": round(self.latency_ms, 2),
            "ok": self.ok,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider": self.provider,
            "attempts": self.attempts,
            "fatal": self.fatal,
        }


# --------------------------------------------------------------------------
# SQL extraction
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"```(?:sql|postgresql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STATEMENT_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(raw: str) -> str:
    """Pull the SQL statement out of a chat model's reply.

    Instruction-following models wrap SQL in prose, markdown fences, and — for
    Qwen3 — sometimes a <think> block even when reasoning is disabled. A
    benchmark that failed to unwrap these would report parsing failures as
    model errors and understate real accuracy.

    This is deliberately forgiving about *formatting* and strict about
    *content*: it never repairs SQL, only locates it. Repair is Phase 11, and
    doing any of it here would contaminate the baseline.
    """
    if not raw:
        return ""

    text = _THINK_BLOCK.sub("", raw)
    text = _OPEN_THINK.sub("", text)  # unterminated block from a length cutoff
    text = text.strip()

    fenced = _FENCED.search(text)
    if fenced:
        text = fenced.group(1).strip()

    # Drop any preamble before the statement actually begins.
    match = _STATEMENT_START.search(text)
    if match:
        text = text[match.start():]

    # Keep only the first statement; trailing prose or a second query would
    # make this a multi-statement input, which the validator rejects.
    if ";" in text:
        text = text.split(";", 1)[0]

    return " ".join(text.split()).strip()


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

class TextToSQLModel(ABC):
    """Anything that turns a question plus a schema into SQL."""

    model_id: str

    @abstractmethod
    def generate(self, question: str, schema: str) -> GenerationResult:
        ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Reproducibility metadata recorded with every run."""


@dataclass
class InferenceParams:
    """Decoding settings. Recorded verbatim in the run summary.

    Greedy decoding (temperature 0) is chosen for reproducibility: a benchmark
    that returns different SQL on each run cannot support a claim that
    fine-tuning changed anything. Qwen advises against greedy decoding in
    *thinking* mode, which is why reasoning is disabled in the prompt.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = 20260808
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            **self.extra,
        }


class HFInferenceModel(TextToSQLModel):
    """Qwen3-8B served by Hugging Face Inference Providers.

    Requires ``HF_TOKEN`` in the environment (loaded from the git-ignored
    ``.env``). The token is never logged or written to a result file.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        provider: str = "auto",
        params: InferenceParams | None = None,
        max_retries: int = 6,
        timeout_s: float = 120.0,
        prompt=None,
    ) -> None:
        """``prompt`` is the prompt module (``src.model.prompt`` by default,
        ``src.model.prompt_v2`` for the v2 benchmark). Passed in rather than
        imported so one model class serves every prompt version."""
        from huggingface_hub import InferenceClient
        from src.model import prompt as prompt_v1

        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to .env (git-ignored):\n"
                "    HF_TOKEN=hf_...\n"
                "Create one at https://huggingface.co/settings/tokens"
            )

        self.model_id = model_id
        # A comma-separated list is a failover chain: each retry moves to the
        # next provider, so one provider being busy, throttled or out of
        # credits does not take the service down. Measured on 2026-09-19:
        # featherless-ai returned 'model is busy' and 402 while nscale answered
        # in 7 s - the exact reverse of Phase 4. Neither is reliable alone.
        self.providers = [p.strip() for p in provider.split(",") if p.strip()] or ["auto"]
        self.provider = self.providers[0]
        self.prompt = prompt or prompt_v1
        self.params = params or InferenceParams()
        self.max_retries = max_retries
        self._clients = {
            p: InferenceClient(model=model_id, provider=p, api_key=token, timeout=timeout_s)
            for p in self.providers
        }
        self._revision: str | None = None

    def resolve_revision(self) -> str | None:
        """The exact commit SHA of the model repo, for reproducibility.

        A model id alone is not a version: repositories are updated in place.
        Recording the SHA is what lets a future run confirm it benchmarked the
        same weights.
        """
        if self._revision is None:
            try:
                from huggingface_hub import HfApi

                info = HfApi(token=os.getenv("HF_TOKEN")).model_info(self.model_id)
                self._revision = info.sha
            except Exception:
                self._revision = None
        return self._revision

    def generate(self, question: str, schema: str) -> GenerationResult:
        return self.generate_messages(self.prompt.build_messages(question, schema))

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        """Send pre-built chat messages.

        Phase 11 repair uses a different prompt entirely, and routing it
        through ``generate(question, schema)`` would nest the repair prompt
        inside the generation template — wrong system prompt, wrong structure.
        ``generate`` is now a thin wrapper over this, so both paths share one
        retry, timing and error-classification implementation.
        """
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
        }
        if self.params.seed is not None:
            kwargs["seed"] = self.params.seed

        last_error = ""
        started = time.perf_counter()

        n_prov = len(self.providers)
        for attempt in range(1, self.max_retries + 1):
            provider = self.providers[(attempt - 1) % n_prov]
            client = self._clients[provider]
            # Restart the clock each attempt. Retry backoff can be minutes on a
            # throttled free tier, and folding that into "generation latency"
            # would report the provider's rate limit as model speed.
            attempt_started = time.perf_counter()
            try:
                response = client.chat_completion(**kwargs)
                elapsed = (time.perf_counter() - attempt_started) * 1000

                choice = response.choices[0]
                raw = choice.message.content or ""
                usage = getattr(response, "usage", None)

                return GenerationResult(
                    sql=extract_sql(raw),
                    raw_output=raw,
                    latency_ms=elapsed,
                    ok=True,
                    finish_reason=getattr(choice, "finish_reason", None),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    provider=provider,
                    attempts=attempt,
                )

            except Exception as exc:  # network, rate limit, provider error
                last_error = f"[{provider}] {type(exc).__name__}: {exc}"[:300]

                status = getattr(getattr(exc, "response", None), "status_code", None)

                def _saw(codes) -> bool:
                    return status in codes or any(
                        f"{c} Client Error" in last_error
                        or f"{c} Server Error" in last_error
                        for c in codes
                    )

                if _saw(FATAL_HTTP_STATUSES):
                    return GenerationResult(
                        sql="", raw_output="",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        ok=False, error=last_error, attempts=attempt, fatal=True,
                    )

                # With a failover chain, a throttled provider is simply skipped:
                # the next attempt goes elsewhere. The long backoff applies only
                # once every provider in the chain has been tried and throttled.
                if _saw(THROTTLE_HTTP_STATUSES):
                    if n_prov > 1 and attempt % n_prov != 0:
                        continue
                    if attempt < self.max_retries:
                        time.sleep(THROTTLE_BACKOFF_S[
                            min(attempt - 1, len(THROTTLE_BACKOFF_S) - 1)])
                        continue
                # `seed` is optional in the provider API; drop it and retry
                # rather than failing the whole run over an unsupported field.
                if "seed" in kwargs and "seed" in last_error.lower():
                    kwargs.pop("seed")
                    continue
                if attempt < self.max_retries:
                    # Other failures ('model is busy', truncated body, timeout):
                    # switch provider at once, back off only after a full cycle.
                    time.sleep(1 if (n_prov > 1 and attempt % n_prov != 0)
                               else min(2 ** attempt, 20))

        return GenerationResult(
            sql="",
            raw_output="",
            latency_ms=(time.perf_counter() - started) * 1000,
            ok=False,
            error=last_error,
            attempts=self.max_retries,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "hf_inference_providers",
            "model_id": self.model_id,
            "model_revision": self.resolve_revision(),
            "provider": self.provider,
            "provider_chain": self.providers,
            "fine_tuned": False,
            "adapters": [],
            "inference_params": self.params.as_dict(),
        }


class OracleModel(TextToSQLModel):
    """Returns the gold SQL. Used only to self-test the harness.

    A perfect model must score 100 % execution accuracy. If it does not, the
    fault is in the evaluation pipeline — prompt, extraction, execution or
    comparison — and any real model's score would be wrong in the same way.
    Running this before spending money on inference is cheap insurance.
    """

    model_id = "oracle/gold-sql"

    def __init__(self, gold_by_question: dict[str, str]) -> None:
        self._gold = gold_by_question

    def generate(self, question: str, schema: str) -> GenerationResult:
        started = time.perf_counter()
        sql = self._gold.get(question, "")
        return GenerationResult(
            sql=sql,
            raw_output=f"```sql\n{sql}\n```",  # exercise the extraction path
            latency_ms=(time.perf_counter() - started) * 1000,
            ok=bool(sql),
            error=None if sql else "question not found in gold map",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "oracle",
            "model_id": self.model_id,
            "fine_tuned": False,
            "note": "harness self-test; returns gold SQL",
        }
