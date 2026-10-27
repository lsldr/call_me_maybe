"""Model-free checks for the ten-case demonstration benchmark."""

import json
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.evaluation import benchmark
from src.evaluation.cases import EvalCase, ExpectedCall, load_cases
from src.pipeline.function_caller import FunctionCaller


def _case(index: int, prompt: str | None = None) -> EvalCase:
    return EvalCase(
        id=f"demo-{index:02d}", category="demo",
        prompt=prompt or f"Candidate prompt {index}",
        expected=ExpectedCall(
            name="fn_greet", parameters={"name": f"Person {index}"},
        ),
    )


def test_summary_keeps_invalid_and_mismatched_rows_in_denominator() -> None:
    summary = benchmark.summarize_results([
        {
            "valid": True, "exact_match": True, "accepted_match": True,
            "function_correct": True,
        },
        {
            "valid": True, "exact_match": False, "accepted_match": False,
            "function_correct": True,
        },
        {
            "valid": False, "exact_match": False, "accepted_match": False,
            "function_correct": False,
        },
    ])
    assert summary == {
        "total": 3, "valid": 2, "exact": 1, "failed": 1,
        "accepted": 1, "mismatched": 1,
        "function_correct": 2,
        "valid_percent": pytest.approx(200.0 / 3.0),
        "exact_accuracy_percent": pytest.approx(100.0 / 3.0),
        "accepted_accuracy_percent": pytest.approx(100.0 / 3.0),
        "function_accuracy_percent": pytest.approx(200.0 / 3.0),
    }


def test_regex_acceptance_compares_finite_spans_and_results() -> None:
    expected = {
        "source_string": "cat scatter cat",
        "regex": r"\bcat\b",
        "replacement": "dog",
    }
    equivalent = dict(expected, regex=r"(?<!\w)cat(?!\w)")
    assert benchmark.regex_behavior_match(
        expected, equivalent, ["cat", "a cat!", "scatter"],
    )
    wrong = dict(expected, regex="cat")
    assert not benchmark.regex_behavior_match(
        expected, wrong, ["cat", "a cat!", "scatter"],
    )


def test_screened_benchmark_fixture_has_ten_non_original_prompts() -> None:
    cases = load_cases("benchmark")
    originals = benchmark._original_prompt_texts()
    assert len(cases) == 10
    assert not ({case.prompt for case in cases} & originals)


class _FakeCaller:
    def __init__(self) -> None:
        self.decoder = SimpleNamespace(trace_observer=None)
        self.attempt_observer = None


def _stub_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    exact: bool = True,
    elapsed: float = 900.0,
    originals: list[EvalCase] | None = None,
) -> tuple[dict[str, Any], Path]:
    cases = [_case(index) for index in range(10)]
    originals = originals if originals is not None else [_case(90, "original")]
    monkeypatch.setattr(
        benchmark, "load_cases",
        lambda suite: cases,
    )
    monkeypatch.setattr(
        benchmark, "_original_prompt_texts",
        lambda: {case.prompt for case in originals},
    )
    monkeypatch.setattr(
        benchmark, "load_function_definitions", lambda _: {
            "fn_greet": SimpleNamespace(
                model_dump=lambda **_: {"name": "fn_greet"}
            )
        },
    )
    fake = _FakeCaller()
    monkeypatch.setattr(benchmark, "create_caller", lambda *_: fake)
    monkeypatch.setattr(
        benchmark, "_execute_case",
        lambda caller, case, functions, index, **kwargs: {
            "id": case.id, "prompt": case.prompt,
            "expected": case.expected.model_dump(mode="json"),
            "actual": case.expected.model_dump(mode="json"),
            "selected_name": case.expected.name,
            "function_correct": True, "valid": True,
            "exact_match": exact, "accepted_match": exact,
            "behavior_match": None, "elapsed_seconds": elapsed,
            "contexts": [], "attempts": [],
            **({} if exact else {"error": "mismatch"}),
        },
    )
    trace_dir = tmp_path / "traces"

    def allocate(label: str) -> Path:
        assert label == "benchmark_10"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / "1_test_benchmark_10.jsonl"
        path.touch()
        return path

    monkeypatch.setattr(
        "src.diagnostics.trace_paths.allocate_trace", allocate,
    )
    report = benchmark.run_benchmark(tmp_path / "output")
    return report, trace_dir / "1_test_benchmark_10.jsonl"


def test_runs_ten_non_original_cases_once_and_writes_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    report, trace = _stub_run(monkeypatch, tmp_path)
    artifact = json.loads((tmp_path / "output/benchmark.json").read_text())
    assert report["case_count"] == 10
    assert len(report["results"]) == 10
    assert report["summary"]["exact"] == 10
    assert report["passed"] is True
    assert trace.is_file()
    lines = trace.read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "run_start"
    assert json.loads(lines[-1])["kind"] == "run_end"
    assert "source_sha256" not in artifact
    assert "sdk_sha256" not in artifact
    assert "signature" not in json.dumps(artifact).lower()


def test_original_prompt_duplication_is_rejected_before_model_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cases = [_case(index) for index in range(10)]
    originals = [_case(90, cases[3].prompt)]
    monkeypatch.setattr(
        benchmark, "load_cases",
        lambda suite: cases,
    )
    monkeypatch.setattr(
        benchmark, "_original_prompt_texts",
        lambda: {case.prompt for case in originals},
    )
    monkeypatch.setattr(
        benchmark, "create_caller",
        lambda *_: pytest.fail("model factory should not run"),
    )
    with pytest.raises(ValueError, match="must not duplicate original"):
        benchmark.run_benchmark(tmp_path / "output")


def test_mismatch_is_reported_and_does_not_apply_time_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    report, _ = _stub_run(
        monkeypatch, tmp_path, exact=False, elapsed=999_999.0,
    )
    assert report["summary"]["mismatched"] == 10
    assert report["passed"] is False
    assert all(row["elapsed_seconds"] == 999_999.0
               for row in report["results"])


def test_long_elapsed_generation_has_no_time_based_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    report, _ = _stub_run(
        monkeypatch, tmp_path, exact=True, elapsed=999_999.0,
    )
    assert report["passed"] is True
    assert report["summary"]["exact"] == 10


def test_argument_error_preserves_successful_function_selection() -> None:
    class FakeCaller:
        encode: Callable[[str], list[int]] = staticmethod(lambda _: [])
        context_observer: Callable[[str, str], None] | None = None
        selection_observer: Callable[[str], None] | None = None
        attempt_observer: Callable[[str, dict[str, Any]], None] | None = None

        def __init__(self, error: Exception) -> None:
            self.decoder = SimpleNamespace(trace_observer=None)
            self.error = error

        def call(self, *_: Any, **__: Any) -> Any:
            assert self.selection_observer is not None
            self.selection_observer("fn_greet")
            raise self.error

    row = benchmark._execute_case(
        cast(FunctionCaller, FakeCaller(
            ValueError("parameter generation failed"),
        )),
        _case(1), {}, 0,
    )
    assert row["selected_name"] == "fn_greet"
    assert row["function_correct"] is True
    assert row["valid"] is False


def test_oserror_from_trace_or_storage_propagates_per_case() -> None:
    class FakeCaller:
        encode: Callable[[str], list[int]] = staticmethod(lambda _: [])
        context_observer: Callable[[str, str], None] | None = None
        selection_observer: Callable[[str], None] | None = None
        attempt_observer: Callable[[str, dict[str, Any]], None] | None = None

        def __init__(self) -> None:
            self.decoder = SimpleNamespace(trace_observer=None)

        def call(self, *_: Any, **__: Any) -> Any:
            assert self.selection_observer is not None
            self.selection_observer("fn_greet")
            raise OSError("trace write failed")

    with pytest.raises(OSError, match="trace write failed"):
        benchmark._execute_case(
            cast(FunctionCaller, FakeCaller()), _case(1), {}, 0,
        )


def test_failure_footer_error_does_not_mask_original_exception() -> None:
    class FakeWriter:
        is_open = True

        def finish(self, _: dict[str, Any]) -> None:
            raise OSError("footer disk failure")

    original = ValueError("generation failed")
    benchmark._finish_trace_on_error(
        FakeWriter(), original,  # type: ignore[arg-type]
    )
    assert isinstance(original, ValueError)
    assert "footer disk failure" in str(original.args)


def test_local_demo_guards_sdk_encode_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelWrapper:
        def encode(self, _: str) -> str:
            return "sdk encode"

        def decode(self, _: list[int]) -> str:
            return "sdk decode"

    monkeypatch.setitem(
        sys.modules,
        "src.llm.model_wrapper",
        SimpleNamespace(ModelWrapper=FakeModelWrapper),
    )
    from src.evaluation.demo import _guard_sdk_tokenizer

    with ExitStack() as stack:
        _guard_sdk_tokenizer(stack, "local")
        with pytest.raises(AssertionError, match="SDK encode"):
            FakeModelWrapper().encode("x")
        with pytest.raises(AssertionError, match="SDK decode"):
            FakeModelWrapper().decode([1])
    assert FakeModelWrapper().encode("x") == "sdk encode"


def test_renderer_closes_when_trace_open_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class FakeRenderer:
        closed = False

        def open(self) -> "FakeRenderer":
            return self

        def observe(self, _: Any) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class FakeWriter:
        is_open = False

        def __init__(self, **_: Any) -> None:
            pass

        def open(self) -> "FakeWriter":
            raise OSError("trace open failed")

    renderer = FakeRenderer()
    monkeypatch.setattr(
        "src.visualization.live.LiveRenderer", lambda: renderer,
    )
    monkeypatch.setattr(benchmark, "TraceWriter", FakeWriter)
    monkeypatch.setattr(
        "src.diagnostics.trace_paths.allocate_trace",
        lambda _: tmp_path / "trace.jsonl",
    )
    with pytest.raises(OSError, match="trace open failed"):
        benchmark.run_benchmark(tmp_path / "output", visual=True)
    assert renderer.closed is True


def test_cli_returns_nonzero_for_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output", "/tmp/benchmark-test"],
    )
    monkeypatch.setattr(
        benchmark, "run_benchmark", lambda *args, **kwargs: {"passed": False},
    )
    assert benchmark.main() == 1


def test_offline_environment_disables_hub_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark._offline_environment()
    assert all(
        __import__("os").environ[name] == "1"
        for name in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
        )
    )
