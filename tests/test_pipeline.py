"""Exercise complete function calls with a byte-level fake model."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import pytest

from src.config import RunSettings
from src.decoding.decoder import Decoder
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller, create_caller
from src.__main__ import main
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def make_function(
    name: str,
    parameters: dict[str, TypeDef] | None = None,
) -> FunctionDef:
    """Build a validated function definition for pipeline tests."""
    return FunctionDef(
        name=name,
        description=f"Use {name}",
        parameters=parameters or {},
        returns=TypeDef(type="string"),
    )


def make_caller(
    functions: dict[str, FunctionDef],
    *,
    failing_prompts: set[str] | None = None,
    max_new_tokens: int = 256,
    style: Literal["native", "boundary", "examples"] = "native",
    markers: list[str] | None = None,
) -> tuple[FunctionCaller, list[str]]:
    """Construct a caller whose logits select scripted UTF-8 bytes."""
    char_by_byte = build_byte_to_unicode()
    vocab = VocabTable(
        vocab={char_by_byte[value]: value for value in range(256)},
        tokenizer={},
        n_logits=256,
    )
    encoded_contexts: list[str] = []
    failing = failing_prompts or set()

    def encode(text: str) -> list[int]:
        encoded_contexts.append(text)
        return list(text.encode("utf-8"))

    def logits(ids: list[int]) -> np.ndarray[Any, np.dtype[np.float32]]:
        raw = bytes(ids)
        context = raw.decode("utf-8", errors="ignore")
        encoded_contexts.append(context)
        user_text = context.rsplit("<|im_start|>user\n", maxsplit=1)[1]
        user_text = user_text.split("<|im_end|>\n", maxsplit=1)[0]
        try:
            request = json.loads(user_text)
        except json.JSONDecodeError:
            request = user_text
        if request in failing:
            return np.zeros(1, dtype=np.float32)

        route = "complex task" if request == "complex task" else "empty task"
        function_name = "multi" if route == "complex task" else "empty"
        candidates: list[tuple[bytes, bytes]] = [
            (b'{"name": "', function_name.encode("utf-8") + b'"'),
        ]
        if route == "complex task":
            first_string = "café 🧪 <|im_end|> TOKEN"
            candidates.extend(
                [
                    (b'"unicode": "', first_string.encode("utf-8") + b'"'),
                    (b'"confirmation": "', b'confirmed"'),
                ]
            )
            candidates.extend(
                [
                    (b'"enabled":', b" false"),
                    (b'"count":', b" 37,"),
                    (b'"ratio":', b" -12.5}"),
                ]
            )

        chosen: int | None = None
        for prefix, target in candidates:
            for consumed in range(len(target)):
                if raw.endswith(prefix + target[:consumed]):
                    chosen = target[consumed]
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise AssertionError(
                f"No scripted continuation for {context[-100:]!r}"
            )

        scores = np.full(256, -1000.0, dtype=np.float32)
        scores[chosen] = 1000.0
        return scores

    decoder = Decoder(vocab=vocab, max_new_tokens=max_new_tokens)
    caller = FunctionCaller(
        functions=functions,
        decoder=decoder,
        encode=encode,
        logits=logits,
        markers=markers or [],
        style=style,
    )
    return caller, encoded_contexts


@pytest.fixture
def registry() -> dict[str, FunctionDef]:
    """Return the two invented functions used in the model-free tests."""
    return {
        "multi": make_function(
            "multi",
            {
                "unicode": TypeDef(
                    type="string", description="Unicode display text"
                ),
                "confirmation": TypeDef(
                    type="string", description="Confirmation text"
                ),
                "enabled": TypeDef(type="boolean"),
                "missing": TypeDef(type="null"),
                "count": TypeDef(type="integer"),
                "ratio": TypeDef(type="number"),
            },
        ),
        "empty": make_function("empty"),
    }


def test_run_selects_by_logits_and_generates_every_scalar_type(
    registry: dict[str, FunctionDef],
) -> None:
    """Generated values are exact and prompts remain in original order."""
    caller, contexts = make_caller(
        registry, markers=["<|im_end|>", "TOKEN"]
    )
    prompts = [
        PromptItem(prompt="complex task"),
        PromptItem(prompt="empty task"),
    ]
    calls = caller.run(prompts)
    assert [call.name for call in calls] == ["multi", "empty"]
    assert calls[0].parameters == {
        "unicode": "café 🧪 <|im_end|> TOKEN",
        "confirmation": "confirmed",
        "enabled": False,
        "missing": None,
        "count": 37,
        "ratio": -12.5,
    }
    assert calls[1].parameters == {}
    assert [call.prompt for call in calls] == [p.prompt for p in prompts]
    assert any(
        '"unicode": "café 🧪 \\u003c|im_end|> \\u0054OKEN",'
        '"confirmation": "confirmed","enabled":' in context
        for context in contexts
    )
    assert any(
        '"missing": null,"count":' in context for context in contexts
    )
    assert any(
        '"count": 37,"ratio": ' in context for context in contexts
    )
    assert any('"ratio": -12.5' in context for context in contexts)


def test_create_caller_uses_examples_style_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production factory selects examples prompts using a fake wrapper."""
    vocab_path = tmp_path / "vocab.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path.write_text(json.dumps({"A": 0}), encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")
    factory_calls: list[dict[str, str]] = []

    def encode(_: str) -> list[int]:
        return [0]

    def logits(_: list[int]) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.zeros(1, dtype=np.float32)

    def model_factory(**kwargs: str) -> SimpleNamespace:
        factory_calls.append(dict(kwargs))
        return SimpleNamespace(
            encode=encode,
            logits=logits,
            get_vocab_file_path=lambda: str(vocab_path),
            get_tokenizer_file_path=lambda: str(tokenizer_path),
        )

    monkeypatch.setattr(
        "src.llm.model_wrapper.ModelWrapper", model_factory
    )
    function = make_function("select", {"value": TypeDef(type="string")})
    caller = create_caller({"select": function}, RunSettings())

    assert caller.style == "examples"
    assert caller.score_string_opening is True
    assert caller.allow_number_suffix is True
    assert caller.decoder.optimize is True
    assert len(factory_calls) == 1
    assert factory_calls[0] == {"model_name": "Qwen/Qwen3-0.6B"}


def test_run_preserves_repeated_prompt_order(
    registry: dict[str, FunctionDef],
) -> None:
    """Repeated prompt texts still produce one ordered result per input."""
    caller, _ = make_caller(registry)
    prompts = [
        PromptItem(prompt="empty task"),
        PromptItem(prompt="complex task"),
        PromptItem(prompt="empty task"),
    ]
    calls = caller.run(prompts)
    assert [call.prompt for call in calls] == [p.prompt for p in prompts]
    assert [call.name for call in calls] == ["empty", "multi", "empty"]


def test_call_failure_does_not_return_default_values(
    registry: dict[str, FunctionDef],
) -> None:
    """A bad model row raises instead of fabricating argument defaults."""
    caller, _ = make_caller(registry, failing_prompts={"empty task"})
    with pytest.raises(ValueError, match="Model logits dimension changed"):
        caller.call(PromptItem(prompt="empty task"))


def test_run_reports_the_failing_prompt_index(
    registry: dict[str, FunctionDef],
) -> None:
    """Batch errors identify the prompt whose generation failed."""
    caller, _ = make_caller(registry, failing_prompts={"complex task"})
    with pytest.raises(
        ValueError, match=r"Prompt 1: Model logits dimension changed"
    ):
        caller.run(
            [
                PromptItem(prompt="empty task"),
                PromptItem(prompt="complex task"),
            ]
        )


class CliCall(BaseModel):
    """Minimal Pydantic result object written by the CLI test caller."""

    prompt: str
    name: str
    parameters: dict[str, object]


class FakeCliDecoder(BaseModel):
    """Minimal decoder observation hooks for the CLI fake."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    trace_observer: Any = None
    max_value_bytes: int = 4096


class FakeCliCaller(BaseModel):
    """Pydantic fake caller with injectable failure and run tracking."""

    fail_on_prompt: str | None = None
    run_count: int = 0
    decoder: FakeCliDecoder = Field(default_factory=FakeCliDecoder)
    attempt_observer: Any = None

    def run(self, prompts: list[PromptItem]) -> list[CliCall]:
        """Return one fake result per prompt, or fail on the chosen prompt."""
        self.run_count += 1
        calls: list[CliCall] = []
        for prompt in prompts:
            if prompt.prompt == self.fail_on_prompt:
                raise ValueError(f"Prompt {len(calls)}: simulated failure")
            calls.append(
                CliCall(prompt=prompt.prompt, name="hello", parameters={})
            )
        return calls


def write_cli_inputs(
    tmp_path: Path, prompts: object, functions: object | None = None,
) -> tuple[Path, Path, Path]:
    """Write minimal CLI inputs and return their paths."""
    functions_path = tmp_path / "functions.json"
    functions_path.write_text(
        json.dumps(
            functions
            if functions is not None
            else [{
                "name": "hello",
                "description": "Say hello",
                "parameters": {},
                "returns": {"type": "string"},
            }]
        ),
        encoding="utf-8",
    )
    prompts_path = tmp_path / "input.json"
    prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
    return functions_path, prompts_path, tmp_path / "output.json"


def invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    paths: tuple[Path, Path, Path],
) -> int:
    """Set CLI arguments and execute ``main`` in-process."""
    functions_path, input_path, output_path = paths
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cmm",
            "--functions_definition",
            str(functions_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    return main()


def test_cli_initializes_once_and_writes_successful_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The model factory is invoked once and results are written in order."""
    paths = write_cli_inputs(tmp_path, [{"prompt": "one"}, {"prompt": "two"}])
    creations: list[FakeCliCaller] = []

    def create_caller(
        functions: object, settings: RunSettings
    ) -> FakeCliCaller:
        del functions, settings
        caller = FakeCliCaller()
        creations.append(caller)
        return caller

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", create_caller
    )
    assert invoke_main(monkeypatch, paths) == 0
    assert len(creations) == 1
    assert creations[0].run_count == 1
    assert [item["prompt"] for item in json.loads(paths[2].read_text())] == [
        "one",
        "two",
    ]


def test_cli_second_prompt_failure_preserves_original_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A failure later in the batch leaves the prior output untouched."""
    paths = write_cli_inputs(tmp_path, [{"prompt": "one"}, {"prompt": "two"}])
    paths[2].write_text("original", encoding="utf-8")
    creations: list[FakeCliCaller] = []

    def create_caller(
        functions: object, settings: RunSettings
    ) -> FakeCliCaller:
        del functions, settings
        caller = FakeCliCaller(fail_on_prompt="two")
        creations.append(caller)
        return caller

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", create_caller
    )
    assert invoke_main(monkeypatch, paths) == 1
    assert len(creations) == 1
    assert paths[2].read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    "prompts,functions,malformed",
    [
        ([], None, None),
        ([{"prompt": "x"}], [], None),
        ([{"wrong": "x"}], None, None),
        ([{"prompt": "x"}], None, "bad functions"),
    ],
)
def test_cli_does_not_initialize_for_empty_or_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prompts: object,
    functions: object | None,
    malformed: str | None,
) -> None:
    """Input validation and no-op cases finish before model construction."""
    paths = write_cli_inputs(tmp_path, prompts, functions)
    if malformed:
        paths[0].write_text(malformed, encoding="utf-8")
    creations: list[object] = []

    def create_caller(
        functions_value: object, settings: RunSettings
    ) -> FakeCliCaller:
        del functions_value, settings
        creations.append(object())
        return FakeCliCaller()

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", create_caller
    )
    code = invoke_main(monkeypatch, paths)
    if prompts == []:
        assert code == 0
        assert json.loads(paths[2].read_text()) == []
    else:
        assert code == 1
    assert creations == []
