"""Model-free boundary tests spanning string generation and the caller."""

import numpy as np
from numpy.typing import NDArray

from src.decoding.decoder import Decoder
from src.decoding.generators import generate_string
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def make_decoder(tokens: list[bytes]) -> tuple[Decoder, dict[bytes, int]]:
    """Build byte-alphabet token mappings with extra merged tokens."""
    byte_chars = build_byte_to_unicode()
    raw_tokens = [bytes([byte]) for byte in range(256)] + tokens
    token_ids = {raw: index for index, raw in enumerate(raw_tokens)}
    vocab = {
        "".join(byte_chars[byte] for byte in raw): token_id
        for raw, token_id in token_ids.items()
    }
    table = VocabTable(
        vocab=vocab,
        tokenizer={},
        n_logits=len(raw_tokens),
    )
    return Decoder(vocab=table), token_ids


def fixed_scores(decoder: Decoder, selected: int) -> NDArray[np.float32]:
    """Return logits with one selected vocabulary token highest."""
    scores = np.full(decoder.vocab.n_logits, -100.0, dtype=np.float32)
    scores[selected] = 10.0
    return scores


def test_string_generator_retains_consumed_ascii_boundary_suffix() -> None:
    """Generation returns raw token bytes and its exactly matched suffix."""
    following = b', "next": "'
    combined = b'hello", "next": "'
    invalid = b'hello", "wrong'
    decoder, token_ids = make_decoder([combined, invalid])

    def logits(_: list[int]) -> NDArray[np.float32]:
        scores = fixed_scores(decoder, token_ids[combined])
        scores[token_ids[invalid]] = 1000.0
        return scores

    result = generate_string(decoder, [0], logits, following=following)
    assert result.value == "hello"
    assert result.fragment == combined
    assert result.consumed_suffix == following


def test_scored_opening_masks_wrong_suffix_and_keeps_full_value() -> None:
    """The quote is scored and an invalid high-score boundary is excluded."""
    following = b"}}\n"
    valid = b' "[aeiou]"}}\n'
    invalid = b' "[aeiou]"x}}\n'
    decoder, token_ids = make_decoder([valid, invalid])

    def logits(_: list[int]) -> NDArray[np.float32]:
        scores = fixed_scores(decoder, token_ids[valid])
        scores[token_ids[invalid]] = 1000.0
        return scores

    result = generate_string(
        decoder, [6], logits, following=following, include_opening=True
    )
    assert result.value == "[aeiou]"
    assert result.fragment == valid
    assert result.context_ids == [6, token_ids[valid]]
    assert result.consumed_suffix == following


def test_scored_opening_keeps_a_leading_star_literal() -> None:
    """A generated opening quote and star do not gain an artificial space."""
    token = b'"*"'
    decoder, token_ids = make_decoder([token])
    result = generate_string(
        decoder,
        [2],
        lambda _: fixed_scores(decoder, token_ids[token]),
        include_opening=True,
    )
    assert result.value == "*"
    assert result.fragment == token
    assert result.context_ids == [2, token_ids[token]]


def make_boundary_caller(
    first_key: str,
    second_key: str,
    first_token: bytes,
    second_token: bytes,
) -> tuple[FunctionCaller, list[str]]:
    """Build a two-string boundary caller with scripted token choices."""
    name_token = b'fn"'
    decoder, token_ids = make_decoder(
        [name_token, first_token, second_token]
    )
    definition = FunctionDef(
        name="fn",
        description="Return two strings",
        parameters={
            first_key: TypeDef(type="string"),
            second_key: TypeDef(type="string"),
        },
        returns=TypeDef(type="null"),
    )
    contexts: list[str] = []
    desired = [name_token, first_token, second_token]
    calls = 0

    def encode(text: str) -> list[int]:
        contexts.append(text)
        return list(text.encode("utf-8"))

    def logits(_: list[int]) -> NDArray[np.float32]:
        nonlocal calls
        selected = token_ids[desired[calls]]
        calls += 1
        return fixed_scores(decoder, selected)

    caller = FunctionCaller(
        functions={"fn": definition},
        decoder=decoder,
        encode=encode,
        logits=logits,
        style="boundary",
    )
    return caller, contexts


def test_pipeline_reuses_merged_ascii_next_key_exactly_once() -> None:
    """A token may consume comma, spacing, and the next key's prefix."""
    first = b'alpha", "next": "'
    second = b'x"}}\n'
    caller, contexts = make_boundary_caller("first", "next", first, second)
    result = caller.call(PromptItem(prompt="provide the values"))

    assert result.parameters == {"first": "alpha", "next": "x"}
    assert len(contexts) == 3
    generated_args = contexts[2].rsplit('"arguments": {', 1)[1]
    assert generated_args.endswith('"first": "alpha", "next": "')
    assert generated_args.count('"next": "') == 1


def test_pipeline_holds_partial_utf8_key_suffix_until_full_key() -> None:
    """A split multibyte key prefix is assembled once in encoded context."""
    first = b'x", "\xc3'
    second = b'y"}}\n'
    caller, contexts = make_boundary_caller("first", "é", first, second)
    result = caller.call(PromptItem(prompt="provide the values"))

    assert result.parameters == {"first": "x", "é": "y"}
    assert len(contexts) == 3
    generated_args = contexts[2].rsplit('"arguments": {', 1)[1]
    assert generated_args.endswith('"first": "x", "é": "')
    assert generated_args.count('"é": "') == 1


def test_scored_opening_completes_partial_utf8_key_before_next_value() -> None:
    """The next key prefix completes without adding or losing its bytes."""
    first = b' "x", "\xc3'
    second = b' "y"}}\n'
    caller, contexts = make_boundary_caller("first", "é", first, second)
    caller.score_string_opening = True
    result = caller.call(PromptItem(prompt="provide the values"))

    assert result.parameters == {"first": "x", "é": "y"}
    assert len(contexts) == 3
    assert contexts[1].endswith('"first":')
    assert contexts[2].endswith('"first": "x", "é":')
    assert not contexts[2].endswith('"é": "')
    generated_args = contexts[2].rsplit('"arguments": {', 1)[1]
    assert generated_args.count('"é":') == 1
