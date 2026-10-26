"""Model-free tests for benchmark scoring and failure accounting."""

import json
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

import scripts.eval_prompts as eval_prompts
from scripts.eval_prompts import run_style, semantically_matches_regex
from src.config import RunSettings
from src.decoding.decoder import Decoder
from src.decoding.tracer import TraceEvent
from src.io.reader import load_function_definitions
from src.models.function_def import FunctionDef
from src.models.prompt import FunctionCallResult, PromptItem
from src.prompting.indexed import indexed_context
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


@pytest.fixture(autouse=True)
def isolated_reproducibility_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise source and fixture hashing without generated repo artifacts."""
    root = tmp_path / "reproducibility-root"
    source_file = root / "src" / "pipeline" / "function_caller.py"
    sdk_file = root / "llm_sdk" / "llm_sdk" / "__init__.py"
    source_file.parent.mkdir(parents=True)
    sdk_file.parent.mkdir(parents=True)
    source_file.write_text("# temporary pipeline source\n", encoding="utf-8")
    sdk_file.write_text("# temporary SDK marker\n", encoding="utf-8")

    fixture_dir = root / "benchmarks" / "qwen"
    fixture_dir.mkdir(parents=True)
    for filename in (
        "functions.json", "dev_prompts.json", "dev_expected.json",
    ):
        (fixture_dir / filename).write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(eval_prompts, "ROOT", root)


class FakeCaller(BaseModel):
    """Small caller double that tracks token-scoring calls without a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[dict[str, Any] | Exception]
    functions: dict[str, FunctionDef]
    call_count: int = 0
    cursor: int = 0
    style: Literal[
        "native", "compact", "focused", "boundary", "indexed"
    ] = "native"
    markers: list[str] = Field(default_factory=list)
    fail_encode_for: str | None = None
    context_observer: Callable[[str, str], None] | None = None
    decoder: Decoder | None = None

    def call(self, prompt: PromptItem) -> FunctionCallResult:
        """Return the next fixture response or raise its planned failure."""
        response = self.responses[self.cursor]
        self.cursor += 1
        self.call_count += 2
        if self.context_observer is not None:
            self.context_observer("name", f"name decoding: {prompt.prompt}")
        if (
            self.decoder is not None
            and self.decoder.trace_observer is not None
        ):
            self.decoder.trace_observer(TraceEvent(
                step=0,
                token_id=0,
                raw=b"\xc3",
                allowed_count=2,
                complete=False,
                state="partial",
                byte_count=1,
                max_steps=8,
            ))
        if isinstance(response, Exception):
            raise response
        if self.context_observer is not None:
            self.context_observer(
                "parameter:name", f"argument decoding: {prompt.prompt}"
            )
        response["prompt"] = prompt.prompt
        return FunctionCallResult.model_validate(
            response, context={"functions": self.functions}
        )

    def logits(self, input_ids: list[int]) -> list[float]:
        """Expose a bound method so the evaluator can read call_count."""
        del input_ids
        return []

    def encode(self, text: str) -> list[int]:
        """Use whitespace pieces as deterministic stand-in token IDs."""
        if self.fail_encode_for is not None and self.fail_encode_for in text:
            raise RuntimeError("planned context encoding failure")
        return list(range(len(text.split())))


def get_functions() -> dict[str, FunctionDef]:
    """Load benchmark contracts without initializing model resources."""
    return load_function_definitions(
        Path(__file__).resolve().parents[1]
        / "data"
        / "input"
        / "functions_definition.json"
    )


def get_decoder() -> Decoder:
    """Create a tiny real decoder without loading model assets."""
    byte_to_unicode = build_byte_to_unicode()
    spelling = byte_to_unicode[ord("a")]
    return Decoder(
        vocab=VocabTable(
            vocab={spelling: 0}, tokenizer={}, n_logits=1
        )
    )


def result_for(prompt: str, name: str) -> dict[str, Any]:
    """Create a valid greeting result for a specific fixture prompt."""
    guest = prompt.removeprefix("Greet ")
    return {
        "prompt": prompt,
        "name": name,
        "parameters": {"name": guest},
    }


def test_regex_semantic_match_and_mismatch() -> None:
    """Require matching source/replacement and equivalent regex behavior."""
    expected: dict[str, Any] = {
        "source_string": "x12y",
        "regex": r"\d+",
        "replacement": "#",
    }
    equivalent = {
        "source_string": "x12y",
        "regex": "[0-9]+",
        "replacement": "#",
    }
    assert semantically_matches_regex(equivalent, expected) is True

    wrong_source = {**equivalent, "source_string": "x13y"}
    wrong_replacement = {**equivalent, "replacement": "!"}
    wrong_behavior = {**equivalent, "regex": "\\w+"}
    invalid_regex = {**equivalent, "regex": "["}
    assert (
        semantically_matches_regex(wrong_source, expected) is False
    )
    assert (
        semantically_matches_regex(wrong_replacement, expected) is False
    )
    assert semantically_matches_regex(wrong_behavior, expected) is False
    assert semantically_matches_regex(invalid_regex, expected) is False


@pytest.mark.parametrize("split, accepted", [
    ("original", False), ("dev", True), ("heldout", True),
])
def test_runtime_acceptance_only_applies_to_original_inputs(
    tmp_path: Path, split: str, accepted: bool,
) -> None:
    """Larger accuracy sets do not inherit the original-input time limit."""
    functions = get_functions()
    prompt = PromptItem(prompt="Greet Ada")
    response = result_for(prompt.prompt, "fn_greet")
    caller = FakeCaller(responses=[response], functions=functions)
    summary = run_style(
        "native", [prompt], {prompt.prompt: response}, functions,
        caller, tmp_path, split, "cpu", RunSettings(device="cpu"),
        None, 301.0, reproducibility={"unit_test": True},
    )
    assert summary["thresholds"]["elapsed_under_300_seconds"] is False
    assert summary["runtime_acceptance"]["applies"] == (split == "original")
    assert summary["thresholds"]["all_passed"] is accepted


def test_run_style_separates_exact_regex_from_semantic_and_function_match(
    tmp_path: Path,
) -> None:
    """Equivalent regexes match semantically but not exactly."""
    functions = get_functions()
    prompt = PromptItem(prompt="Replace digits in 'ID42' with '#'.")
    expected_parameters: dict[str, Any] = {
        "source_string": "ID42",
        "regex": r"\d+",
        "replacement": "#",
    }
    expected = {
        prompt.prompt: {
            "prompt": prompt.prompt,
            "name": "fn_substitute_string_with_regex",
            "parameters": expected_parameters,
        }
    }
    equivalent_parameters = {
        **expected_parameters,
        "regex": "[0-9]+",
    }
    caller = FakeCaller(
        responses=[{
            "prompt": prompt.prompt,
            "name": "fn_substitute_string_with_regex",
            "parameters": equivalent_parameters,
        }],
        functions=functions,
    )

    summary = run_style(
        "native", [prompt], expected, functions, caller, tmp_path,
        "dev", "cpu", RunSettings(device="cpu"), None, 0.0,
    )
    regex_result = json_load(
        tmp_path / "dev" / "native" / "results.json"
    )[0]
    assert regex_result["exact_match"] is False
    assert regex_result["selected_correct"] is True
    assert regex_result["semantic_call"] is True
    assert regex_result["complete_call_semantic"] is True
    assert summary["complete_call_semantic"] == {"count": 1, "rate": 1.0}
    assert summary["batch_validated"] is True
    assert json_load(
        tmp_path / "dev" / "native" / "production_batch.json"
    ) == json_load(tmp_path / "dev" / "native" / "outputs.json")

    wrong_function_dir = tmp_path / "wrong-function"
    wrong_function = FakeCaller(
        responses=[{
            "prompt": prompt.prompt,
            "name": "fn_greet",
            "parameters": {"name": "ID42"},
        }],
        functions=functions,
    )
    run_style(
        "native", [prompt], expected, functions, wrong_function,
        wrong_function_dir, "dev", "cpu", RunSettings(device="cpu"),
        None, 0.0,
    )
    wrong_result = json_load(
        wrong_function_dir / "dev" / "native" / "results.json"
    )[0]
    assert wrong_result["selected_correct"] is False
    assert wrong_result["semantic_call"] is False


def test_swapped_addition_counts_semantic_but_not_exact(
    tmp_path: Path,
) -> None:
    """Swapping addends is task equivalent without an exact argument match."""
    functions = get_functions()
    prompt = PromptItem(prompt="Add 2 and 3.")
    expected = {
        prompt.prompt: {
            "prompt": prompt.prompt,
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        }
    }
    caller = FakeCaller(
        responses=[{
            "prompt": prompt.prompt,
            "name": "fn_add_numbers",
            "parameters": {"a": 3, "b": 2},
        }],
        functions=functions,
    )

    summary = run_style(
        "native", [prompt], expected, functions, caller, tmp_path,
        "dev", "cpu", RunSettings(device="cpu"), None, 0.0,
    )

    result = json_load(tmp_path / "dev" / "native" / "results.json")[0]
    assert result["exact_match"] is False
    assert result["complete_call_semantic"] is True
    assert summary["exact_call"] == {"count": 0, "rate": 0.0}
    assert summary["complete_call_semantic"] == {"count": 1, "rate": 1.0}
    assert summary["batch_validated"] is True


def test_run_style_keeps_failed_cases_in_denominators_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    """A failed case remains in rates and all successful outputs are saved."""
    functions = get_functions()
    prompts = [PromptItem(prompt=f"Greet guest{index}") for index in range(10)]
    expected = {
        prompt.prompt: {
            "prompt": prompt.prompt,
            "name": "fn_greet",
            "parameters": {"name": prompt.prompt.removeprefix("Greet ")},
        }
        for prompt in prompts
    }
    responses: list[dict[str, Any] | Exception] = [
        result_for(prompt.prompt, "fn_greet") for prompt in prompts[:9]
    ]
    responses.append(RuntimeError("planned model failure"))
    caller = FakeCaller(responses=responses, functions=functions)
    settings = RunSettings(device="cpu")

    summary = run_style(
        "native", prompts, expected, functions, caller, tmp_path,
        "dev", "cpu", settings, None, 301.0,
    )

    run_dir = tmp_path / "dev" / "native"
    results = json_load(run_dir / "results.json")
    outputs = json_load(run_dir / "outputs.json")
    contexts = json_load(run_dir / "contexts.json")
    assert json_load(run_dir / "functions.json") == [
        definition.model_dump(exclude_none=True)
        for definition in functions.values()
    ]
    assert summary["examples"] == 10
    assert summary["failures"] == 1
    assert summary["batch_validated"] is False
    assert summary["selection"] == {"count": 9, "rate": 0.9}
    assert summary["thresholds"]["selection_above_90_percent"] is False
    assert (
        summary["elapsed_seconds"]["total_including_initialization"]
        >= 301.0
    )
    assert summary["thresholds"]["elapsed_under_300_seconds"] is False
    assert results[-1]["error"] == "RuntimeError: planned model failure"
    assert results[-1]["selected_correct"] is False
    assert results[-1]["exact_match"] is False
    assert results[-1]["schema_valid"] is False
    assert results[0]["logits_call_count_delta"] == 2
    assert results[0]["generated_token_count_estimate"] == 2
    assert len(outputs) == 9
    assert not (run_dir / "production_batch.json").exists()
    assert len(contexts) == 10
    assert contexts[0]["selection_context_tokens"] > 0
    assert contexts[0]["arguments_context_tokens"] > 0
    assert contexts[0]["actual_decoding_contexts"] == [
        {"phase": "name", "text": "name decoding: Greet guest0"},
        {
            "phase": "parameter:name",
            "text": "argument decoding: Greet guest0",
        },
    ]
    assert caller.context_observer is None
    assert "src/pipeline/function_caller.py" in (
        summary["reproducibility"]["source_sha256"]
    )
    assert "benchmarks/qwen/dev_expected.json" in (
        summary["reproducibility"]["fixture_sha256"]
    )
    assert "numpy" in summary["reproducibility"]["package_versions"]


def test_run_style_exception_keeps_prior_case_artifacts(
    tmp_path: Path,
) -> None:
    """A later instrumentation error leaves completed cases saved."""
    functions = get_functions()
    prompts = [
        PromptItem(prompt="Greet first"),
        PromptItem(prompt="Greet second"),
    ]
    expected = {
        prompt.prompt: {
            "prompt": prompt.prompt,
            "name": "fn_greet",
            "parameters": {"name": prompt.prompt.removeprefix("Greet ")},
        }
        for prompt in prompts
    }
    caller = FakeCaller(
        responses=[
            result_for(prompt.prompt, "fn_greet") for prompt in prompts
        ],
        functions=functions,
        fail_encode_for="Greet second",
    )
    with pytest.raises(RuntimeError, match="planned context encoding failure"):
        run_style(
            "native", prompts, expected, functions, caller, tmp_path,
            "dev", "cpu", RunSettings(device="cpu"), None, 0.0,
        )

    run_dir = tmp_path / "dev" / "native"
    assert len(json_load(run_dir / "results.json")) == 1
    assert len(json_load(run_dir / "outputs.json")) == 1
    assert len(json_load(run_dir / "contexts.json")) == 1
    assert caller.context_observer is None


def test_trace_serializes_raw_bytes_and_indexed_selection_context(
    tmp_path: Path,
) -> None:
    """Trace bytes use hex and indexed context metadata matches the caller."""
    functions = get_functions()
    prompt = PromptItem(prompt="Greet indexed guest")
    expected = {
        prompt.prompt: {
            "prompt": prompt.prompt,
            "name": "fn_greet",
            "parameters": {"name": "indexed guest"},
        }
    }
    decoder = get_decoder()
    decoder.cache.put("stale", np.array([0], dtype=np.int32))
    old_events: list[TraceEvent] = []
    decoder.trace_observer = old_events.append
    caller = FakeCaller(
        responses=[result_for(prompt.prompt, "fn_greet")],
        functions=functions,
        decoder=decoder,
    )

    summary = run_style(
        "indexed", [prompt], expected, functions, caller, tmp_path,
        "dev", "cpu", RunSettings(device="cpu"), None, 0.0, trace=True,
    )

    run_dir = tmp_path / "dev" / "indexed"
    result = json_load(run_dir / "results.json")[0]
    contexts = json_load(run_dir / "contexts.json")[0]
    expected_context, _ = indexed_context(
        list(functions.values()), prompt.prompt
    )
    assert result["token_trace"] == [{
        "step": 0,
        "token_id": 0,
        "allowed_count": 2,
        "complete": False,
        "state": "partial",
        "byte_count": 1,
        "max_steps": 8,
        "raw_hex": "c3",
    }]
    assert "raw" not in result["token_trace"][0]
    assert contexts["selection_context"] == expected_context
    assert contexts["selection_context_tokens"] == len(
        caller.encode(expected_context)
    )
    assert decoder.trace_observer == old_events.append
    assert old_events == []
    assert summary["allowed_set_cache"] == {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "entries": 0,
        "resident_bytes": 0,
    }


def json_load(path: Path) -> Any:
    """Load one JSON artifact written by the evaluator."""
    return json.loads(path.read_text(encoding="utf-8"))
