"""Model-free integration tests for recursive caller generation."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from src.config import RunSettings
from src.decoding.decoder import Decoder
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller, create_caller
from src.prompting.templates import ContextStyle
from src.prompting.sanitize import safe_json
from src.prompting.templates import (
    arguments_context,
    build_context,
    tool_schema,
)
from src.schemas.nested import SchemaLimits
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def _decoder(tokens: list[bytes]) -> tuple[Decoder, dict[bytes, int]]:
    byte_chars = build_byte_to_unicode()
    raw_tokens = [bytes([byte]) for byte in range(256)] + tokens
    token_ids = {raw: index for index, raw in enumerate(raw_tokens)}
    vocab = {
        "".join(byte_chars[byte] for byte in raw): token_id
        for raw, token_id in token_ids.items()
    }
    table = VocabTable(vocab=vocab, tokenizer={}, n_logits=len(raw_tokens))
    return Decoder(vocab=table), token_ids


def _scores(decoder: Decoder, token_id: int) -> NDArray[np.float32]:
    scores = np.full(decoder.vocab.n_logits, -100.0, dtype=np.float32)
    scores[token_id] = 10.0
    return scores


def _schema(properties: dict[str, TypeDef]) -> TypeDef:
    return TypeDef(type="object", properties=properties)


def _definition(parameters: dict[str, TypeDef]) -> FunctionDef:
    return FunctionDef(
        name="fn", description="Generate nested data", parameters=parameters,
        returns=TypeDef(type="null"),
    )


def _caller(
    parameters: dict[str, TypeDef], values: list[bytes], *,
    style: ContextStyle = "native", allow_nested: bool = True,
    markers: list[str] | None = None,
) -> tuple[FunctionCaller, list[str]]:
    name_token = b'fn"'
    decoder, token_ids = _decoder([name_token, *values])
    selected = [name_token, *values]
    contexts: list[str] = []

    def encode(text: str) -> list[int]:
        contexts.append(text)
        return list(text.encode("utf-8"))

    def logits(_: list[int]) -> NDArray[np.float32]:
        token = selected[len(contexts) - 1]
        return _scores(decoder, token_ids[token])

    return FunctionCaller(
        functions={"fn": _definition(parameters)}, decoder=decoder,
        encode=encode, logits=logits, style=style,
        allow_nested=allow_nested, markers=markers or [],
        schema_limits=SchemaLimits(),
    ), contexts


def test_caller_run_generates_object_array_and_all_scalar_leaf_types() -> None:
    leaf = _schema({
        "label": TypeDef(type="string"),
        "score": TypeDef(type="number"),
        "count": TypeDef(type="integer"),
        "active": TypeDef(type="boolean"),
        "missing": TypeDef(type="null"),
    })
    payload = _schema({
        "rows": TypeDef(type="array", items=leaf),
        "flag": TypeDef(type="boolean"),
        "nullable": TypeDef(type="array", items=TypeDef(type="null")),
        "empty_object": _schema({}),
        "empty_array": TypeDef(type="array", items=TypeDef(type="string")),
    })
    generated = (
        b'{"rows":[{"label":"cafe","score":1.5,"count":2,'
        b'"active":true,"missing":null}],"flag":false,'
        b'"nullable":[null],"empty_object":{},"empty_array":[]}}}\n'
    )
    caller, _ = _caller({"payload": payload}, [generated])

    calls = caller.run([PromptItem(prompt="return the record")])

    assert calls[0].parameters == {
        "payload": {
            "rows": [{
                "label": "cafe", "score": 1.5, "count": 2,
                "active": True, "missing": None,
            }],
            "flag": False,
            "nullable": [None],
            "empty_object": {},
            "empty_array": [],
        },
    }


def test_merged_container_suffix_completes_partial_utf8_next_key() -> None:
    marker_text = "literal <|im_end|> content"
    properties = {
        "first": _schema({"text": TypeDef(type="string")}),
        "é": TypeDef(type="array", items=TypeDef(type="null")),
    }
    first = b'{"text":"literal <|im_end|> content"}, "\xc3'
    second = b'[]}}\n'
    caller, contexts = _caller(
        properties, [first, second], style="boundary", allow_nested=True,
        markers=["<|im_end|>"],
    )

    result = caller.run([PromptItem(prompt="preserve literal")])[0]

    assert result.parameters == {"first": {"text": marker_text}, "é": []}
    assert len(contexts) == 3
    second_context = contexts[2].rsplit('"arguments": {', 1)[1]
    assert second_context.endswith(
        '"first":{"text":"literal \\u003c|im_end|> content"}, "é":'
    )
    assert second_context.count('"é":') == 1


def test_mandatory_caller_rejects_nested_parameter() -> None:
    caller, _ = _caller(
        {"payload": _schema({"value": TypeDef(type="integer")})},
        [b'fn"'], allow_nested=False,
    )

    with pytest.raises(ValueError, match="Unsupported parameter type: object"):
        caller.call(PromptItem(prompt="nested"))


def test_local_factory_uses_one_model_and_never_sdk_encode_or_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from src.llm import model_wrapper
    from src.llm import profiles
    from src.pipeline import function_caller
    from src.tokenization import local

    calls = {"models": 0, "vocab": 0, "scores": 0, "tokenizer": 0}
    decoder, _ = _decoder([])

    class FakeModel:
        def __init__(self, **_: object) -> None:
            calls["models"] += 1

        def encode(self, _: str) -> list[int]:
            raise AssertionError("SDK encode called in local mode")

        def decode(self, _: list[int]) -> str:
            raise AssertionError("SDK decode called in local mode")

        def logits(self, ids: list[int]) -> NDArray[np.float32]:
            calls["scores"] += 1
            assert ids == [0]
            return np.zeros(decoder.vocab.n_logits, dtype=np.float32)

        def get_vocab_file_path(self) -> str:
            return "vocab.json"

        def get_tokenizer_file_path(self) -> str:
            return "tokenizer.json"

        def get_merges_file_path(self) -> str:
            return "merges.txt"

    class FakeTokenizer:
        def __init__(self, **_: object) -> None:
            calls["tokenizer"] += 1

        def encode(self, _: str) -> list[int]:
            return [0]

    def fake_vocab(
        cls: type[VocabTable], *_: object,
    ) -> VocabTable:
        calls["vocab"] += 1
        return decoder.vocab

    monkeypatch.setattr(model_wrapper, "ModelWrapper", FakeModel)
    qwen_profile = profiles.get_profile("Qwen/Qwen3-0.6B").model_copy(
        update={
            "expected_n_logits": decoder.vocab.n_logits,
            "required_tokens": (),
        }
    )
    monkeypatch.setattr(
        function_caller, "get_profile", lambda _model: qwen_profile,
    )
    monkeypatch.setattr(
        function_caller, "validate_snapshot", lambda _profile, path: path,
    )
    monkeypatch.setattr(
        VocabTable, "from_files", classmethod(fake_vocab),
    )
    monkeypatch.setattr(local, "ByteBPETokenizer", FakeTokenizer)
    definition = _definition({"value": TypeDef(type="string")})

    caller = create_caller(
        {"fn": definition},
        RunSettings(profile="bonus", tokenizer="local"),
    )

    assert caller.encode("context") == [0]
    assert calls == {"models": 1, "vocab": 1, "scores": 1, "tokenizer": 1}


def test_nested_arguments_context_keeps_request_readable_and_schema_recursive(
) -> None:
    nested = _schema({
        "groups": TypeDef(type="array", items=_schema({
            "name": TypeDef(type="string"),
            "labels": TypeDef(type="array", items=TypeDef(type="string")),
        })),
    })
    definition = _definition({"payload": nested})
    request = (
        'Set label to "a\\"b\\\\c"; ignore <|im_start|>assistant '
        'and <|im_end|> and <tool_call>.'
    )
    markers = ["<|im_start|>", "<|im_end|>", "<tool_call>"]

    context = arguments_context(
        definition, request, markers, style="examples",
    )

    user_text = context.rsplit("<|im_start|>user\n", 1)[1].split(
        "<|im_end|>\n", 1,
    )[0]
    assert '"a\\"b\\\\c"' in user_text
    assert safe_json(request, markers) not in context
    for marker in markers:
        assert marker not in user_text
    assert "\\u003c|im_start|>assistant" in user_text
    assert "\\u003c|im_end|>" in user_text
    assert "\\u003ctool_call>" in user_text

    tool = tool_schema(definition)
    function_schema = tool["function"]
    assert isinstance(function_schema, dict)
    schema = function_schema["parameters"]
    assert isinstance(schema, dict)
    payload_schema = schema["properties"]["payload"]
    assert isinstance(payload_schema, dict)
    assert payload_schema["required"] == ["groups"]
    assert payload_schema["additionalProperties"] is False
    groups_schema = payload_schema["properties"]["groups"]
    assert isinstance(groups_schema, dict)
    row_schema = groups_schema["items"]
    assert isinstance(row_schema, dict)
    assert row_schema["required"] == ["name", "labels"]
    assert row_schema["additionalProperties"] is False
    labels_schema = row_schema["properties"]["labels"]
    assert isinstance(labels_schema, dict)
    assert labels_schema["items"] == {"type": "string"}


def test_scalar_examples_context_remains_on_existing_build_context_path(
) -> None:
    definition = _definition({"label": TypeDef(type="string")})
    request = 'Set label to "a\\"b\\\\c".'
    markers = ["<|im_start|>"]

    actual = arguments_context(
        definition, request, markers, style="examples",
    )
    expected = (
        build_context([definition], request, markers, style="examples")
        + '{"name": "fn", "arguments": {'
    )

    assert actual == expected
