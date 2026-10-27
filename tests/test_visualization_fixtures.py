"""Verify checked-in synthetic traces and their model-free regeneration."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.diagnostics.trace_io import TraceReader
from src.diagnostics.trace_schema import TraceRecord


FIXTURE_DIR = Path("data/evaluation/visualization")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_fixture_tree(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    assert manifest["provenance"] == "synthetic_scripted_scores"
    assert manifest["accuracy_claim"] is False
    assert {item["name"] for item in manifest["workloads"]} == {
        "scalar_escapes", "nested_array_object",
        "recover_after_committed_prefix", "regex_scripted_examples",
    }

    for item in manifest["workloads"]:
        directory = root / item["directory"]
        prompt = _read_json(directory / "input.json")
        functions = _read_json(directory / "functions.json")
        expected = _read_json(directory / "expected.json")
        assert len(prompt) == 1
        assert len(functions) == 1
        assert expected["name"] == "fixture"

        reader = TraceReader(path=directory / "trace.jsonl").open()
        records = list(reader)
        assert not reader.partial
        assert records[0].kind == "run_start"
        assert records[0].data["provenance"] == "synthetic_scripted_scores"
        assert records[0].data["model"] == "synthetic-byte-scorer"
        assert records[0].data["revision"] is None
        assert records[0].data["revision_enforced"] is False
        assert records[0].data["tokenizer"] == "synthetic-byte-identity"
        assert records[0].data["no_model_loaded"] is True
        assert records[-1].kind == "run_end"
        assert records[-1].data["status"] == "completed"
        assert records[-1].data["validation"] == "passed"
        assert all(isinstance(record, TraceRecord) for record in records)

        tokens = [record for record in records if record.kind == "token"]
        assert tokens
        for token in tokens:
            raw_hex = token.data["raw_hex"]
            assert isinstance(raw_hex, str)
            bytes.fromhex(raw_hex)
            assert token.data["typed_state_before"] is not None
            assert token.data["typed_state_after"] is not None


def test_checked_in_visualization_fixtures_are_complete() -> None:
    """The committed demo directory contains validated real-decoder traces."""
    _check_fixture_tree(FIXTURE_DIR)


def test_builder_regenerates_complete_fixtures_in_a_temporary_directory(
    tmp_path: Path,
) -> None:
    """Fixture generation is repeatable, isolated, and model-free."""
    module = pytest.importorskip("scripts.build_visualization_fixtures")
    build_fixtures = module.build_fixtures
    trace_paths = build_fixtures(tmp_path / "regenerated")
    assert len(trace_paths) == 4
    _check_fixture_tree(tmp_path / "regenerated")

    retry_root = tmp_path / "regenerated" / "recover_after_committed_prefix"
    retry_records = list(TraceReader(path=retry_root / "trace.jsonl").open())
    starts = [
        record for record in retry_records if record.kind == "attempt_start"
    ]
    parameter_starts = [
        record for record in starts
        if record.data.get("path") == ["parameters", "result"]
    ]
    assert [record.data["attempt"] for record in parameter_starts] == [1, 2]
    first_start, retry_start = parameter_starts
    first_end = next(
        record for record in retry_records
        if record.seq > first_start.seq and record.kind == "attempt_end"
    )
    retry_end = next(
        record for record in retry_records
        if record.seq > retry_start.seq and record.kind == "attempt_end"
    )
    first_tokens = [
        record for record in retry_records
        if first_start.seq < record.seq < first_end.seq
        and record.kind == "token"
    ]
    retry_tokens = [
        record for record in retry_records
        if retry_start.seq < record.seq < retry_end.seq
        and record.kind == "token"
    ]
    assert b"".join(bytes.fromhex(str(event.data["raw_hex"]))
                    for event in first_tokens) == b"p"
    assert b"".join(bytes.fromhex(str(event.data["raw_hex"]))
                    for event in retry_tokens) == b'recovered"'
    assert [event.data["step"] for event in retry_tokens] == list(
        range(len(retry_tokens))
    )


def test_regex_example_fixture_is_display_only() -> None:
    """Regex strings are fixture outputs, with no compiler in this pipeline."""
    workload = _read_json(
        FIXTURE_DIR / "regex_scripted_examples" / "expected.json")
    parameters = workload["parameters"]
    assert parameters["source"] == "cat scattercat oncat catapult"
    assert parameters["whole_word_pattern"] == r"\bcat\b"
    assert parameters["substring_pattern"] == "cat"
    manifest = _read_json(FIXTURE_DIR / "manifest.json")
    regex_item = next(
        item for item in manifest["workloads"]
        if item["name"] == "regex_scripted_examples"
    )
    assert "No regex engine" in regex_item["fixture_note"]


def test_fixture_checks_run_when_ignored_builder_is_unavailable(
    pytester: pytest.Pytester,
) -> None:
    """Missing script imports skip regeneration while fixture checks run."""
    project_root = Path(__file__).parents[1].resolve()
    source = Path(__file__).resolve()
    pytester.makeconftest(
        "import importlib\n"
        "original_import_module = importlib.import_module\n"
        "def blocked_import(name, package=None):\n"
        "    if name.startswith('scripts.'):\n"
        "        raise ModuleNotFoundError(name)\n"
        "    return original_import_module(name, package)\n"
        "importlib.import_module = blocked_import\n"
        "import sys\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
    )
    contents = source.read_text(encoding="utf-8")
    nested_test = (
        "\n\ndef test_fixture_checks_run_when_ignored_builder_is_unavailable("
    )
    pytester.makefile(
        ".py",
        test_visualization_fixtures=contents.split(nested_test, maxsplit=1)[0],
    )
    shutil.copytree(
        project_root / FIXTURE_DIR, pytester.path / FIXTURE_DIR,
    )

    result = pytester.runpytest_subprocess(
        "-q", "test_visualization_fixtures.py",
    )

    result.assert_outcomes(passed=2, skipped=1)
