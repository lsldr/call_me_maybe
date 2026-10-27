"""Offline unit checks for final bonus verification bookkeeping."""

from __future__ import annotations

import pytest


def test_cli_does_not_initialize_a_model() -> None:
    """CLI parsing accepts only the two pinned profiles and CUDA target."""
    module = pytest.importorskip("scripts.verify_final_bonus")
    parse_args = module.parse_args
    options = parse_args(["--output-dir", "/tmp/verify", "--device", "cuda"])
    assert options.model == "Qwen/Qwen3-0.6B"
    assert options.output.as_posix() == "/tmp/verify"


def test_summary_counts_valid_name_and_exact_independently() -> None:
    """A valid result with wrong arguments does not count as exact."""
    module = pytest.importorskip("scripts.verify_final_bonus")
    rows: list[dict[str, object]] = [
        {"valid": True, "name_match": True, "exact_arguments": True},
        {"valid": True, "name_match": True, "exact_arguments": False},
        {"valid": False, "name_match": False, "exact_arguments": False},
    ]
    assert module.summarize(rows) == {
        "count": 3, "valid": 2, "names": 2, "exact_arguments": 1,
    }
    assert module.validate_accounting(rows) == module.summarize(rows)
