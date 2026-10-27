"""Validated records for the separate demonstration benchmark."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.decoding.generators import Scalar

DATA_DIR = Path("data/evaluation")


class ExpectedCall(BaseModel):
    """Expected selected function and its scalar input arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    parameters: dict[str, Scalar]


class EvalCase(BaseModel):
    """One demonstration request, expected call and optional regex probes."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    category: str
    prompt: str
    expected: ExpectedCall
    probes: list[str] = Field(default_factory=list)


def load_cases(suite: str, *, data_dir: Path = DATA_DIR) -> list[EvalCase]:
    """Load the ten independent demonstration cases from two small files.

    Args:
        suite: The supported suite name, ``benchmark``.
        data_dir: Directory containing benchmark and regex fixture files.

    Returns:
        Validated cases in their documented order.

    Raises:
        ValueError: If the suite name or fixture data is invalid.
    """
    if suite != "benchmark":
        raise ValueError(f"Unknown evaluation suite: {suite}")
    cases: list[EvalCase] = []
    paths = (data_dir / "benchmark.json", data_dir / "regex/benchmark.json")
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            cases.extend(TypeAdapter(list[EvalCase]).validate_python(raw))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid evaluation cases in {path}: {exc}"
            ) from exc
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases
