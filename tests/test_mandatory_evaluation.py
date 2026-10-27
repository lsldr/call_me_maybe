"""Model-free checks for the compatibility runner and inspector."""

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from src.evaluation.benchmark import regex_behavior_match


def test_regex_evaluation_detects_wrong_digit_spans() -> None:
    """Digit runs and individual digits yield different substitutions."""
    expected = {
        "source_string": "a12b", "replacement": "X", "regex": r"\d+",
    }
    equivalent = dict(expected, regex="[0-9]+")
    wrong = dict(expected, regex=r"\d")
    assert regex_behavior_match(expected, equivalent, ["c7d", "none"])
    assert not regex_behavior_match(expected, wrong, ["c7d", "none"])


def test_compatibility_runner_delegates_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The old script name forwards options to the maintained runner."""
    runner = pytest.importorskip("scripts.eval_mandatory")
    from src.evaluation import benchmark

    captured: dict[str, Any] = {}

    def fake_run(
        output: Path, *, device: str, tokenizer: str, visual: bool,
    ) -> dict[str, Any]:
        captured.update({
            "output": output, "device": device,
            "tokenizer": tokenizer, "visual": visual,
        })
        return {"passed": True}

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "eval_mandatory", "--output", str(tmp_path), "--device", "mps",
        "--tokenizer", "local", "--visual",
    ])

    assert runner.main() == 0
    assert captured == {
        "output": tmp_path, "device": "mps",
        "tokenizer": "local", "visual": True,
    }


def test_compatibility_runner_reports_execution_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runner = pytest.importorskip("scripts.eval_mandatory")
    from src.evaluation import benchmark

    def fail_run(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("benchmark unavailable")

    monkeypatch.setattr(benchmark, "run_benchmark", fail_run)
    monkeypatch.setattr(sys, "argv", [
        "eval_mandatory", "--output", str(tmp_path),
    ])
    assert runner.main() == 1
    assert "benchmark failed: RuntimeError: benchmark unavailable" in (
        capsys.readouterr().err
    )


def test_inspector_reads_fixture_metadata_without_rewriting(
    tmp_path: Path,
) -> None:
    """The retained builder name is a read-only JSON inspection utility."""
    inspector = pytest.importorskip("scripts.build_mandatory_cases")
    path = tmp_path / "sample.json"
    payload = [{
        "id": "sample", "category": "literal", "prompt": "Replace it.",
    }]
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")

    result = inspector.inspect_file(path)

    assert result["count"] == 1
    assert result["cases"] == [{
        "id": "sample", "category": "literal", "prompt": "Replace it.",
    }]
    assert path.read_text(encoding="utf-8") == original
