"""Which benchmark and prompt a script is working with.

Phase 15.

There are two frozen artefact sets:

* **v1** — the original benchmark (`dataset/`) and the plain prompt
  (`src/model/prompt.py`). Every published number (10.82 % -> 52.10 %) is v1.
* **v2** — the revised benchmark (`dataset/v2/`, built from
  `dataset/generation/templates_v2.py`) and the prompt with the business
  glossary (`src/model/prompt_v2.py`).

Scripts take ``--version`` and ask this module for paths and modules, so v1
stays byte-for-byte reproducible while v2 is built next to it. Nothing in v1
imports anything from v2.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from src.sql.config import PROJECT_ROOT

VERSIONS = ("v1", "v2")


@dataclass(frozen=True)
class BenchmarkVersion:
    name: str
    dataset_root: Path
    templates_module: str
    prompt_module: str
    experiments_root: Path

    # -- dataset files -----------------------------------------------------
    @property
    def generated(self) -> Path:
        return self.dataset_root / "generated" / "benchmark.jsonl"

    @property
    def statistics(self) -> Path:
        return self.dataset_root / "generated" / "statistics.json"

    @property
    def validated_dir(self) -> Path:
        return self.dataset_root / "validated"

    def split(self, name: str) -> Path:
        return self.dataset_root / name / f"{name}.jsonl"

    @property
    def sft_dir(self) -> Path:
        return self.dataset_root / "sft"

    # -- code --------------------------------------------------------------
    def templates(self) -> ModuleType:
        return importlib.import_module(self.templates_module)

    def prompt(self) -> ModuleType:
        return importlib.import_module(self.prompt_module)

    @property
    def is_v1(self) -> bool:
        return self.name == "v1"


_V1 = BenchmarkVersion(
    name="v1",
    dataset_root=PROJECT_ROOT / "dataset",
    templates_module="dataset.generation.templates",
    prompt_module="src.model.prompt",
    experiments_root=PROJECT_ROOT / "experiments",
)
_V2 = BenchmarkVersion(
    name="v2",
    dataset_root=PROJECT_ROOT / "dataset" / "v2",
    templates_module="dataset.generation.templates_v2",
    prompt_module="src.model.prompt_v2",
    experiments_root=PROJECT_ROOT / "experiments" / "v2",
)


def get(name: str) -> BenchmarkVersion:
    if name == "v1":
        return _V1
    if name == "v2":
        return _V2
    raise ValueError(f"unknown benchmark version {name!r}; expected one of {VERSIONS}")


def add_version_argument(parser) -> None:  # argparse.ArgumentParser
    parser.add_argument(
        "--version", choices=VERSIONS, default="v1",
        help="benchmark/prompt version to operate on (default v1, the frozen one)",
    )
