"""Exercise optional number-boundary decoding without a real model."""

import numpy as np
from numpy.typing import NDArray

from src.constraints.number_boundary import NumberBoundaryDFA
from src.constraints.number_dfa import NumberDFA
from src.decoding.decoder import Decoder
from src.decoding.generators import generate_number
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def decoder_for(tokens: list[bytes]) -> tuple[Decoder, dict[bytes, int]]:
    """Build a decoder containing byte tokens and selected merged tokens."""
    byte_map = build_byte_to_unicode()
    token_bytes = [bytes([value]) for value in range(256)] + tokens
    vocabulary = {
        "".join(byte_map[byte] for byte in raw): token_id
        for token_id, raw in enumerate(token_bytes)
    }
    decoder = Decoder(
        vocab=VocabTable(
            vocab=vocabulary, tokenizer={}, n_logits=len(token_bytes),
        ),
    )
    return decoder, {raw: index for index, raw in enumerate(token_bytes)}


def constant_scores(
    count: int, preferred: int,
    secondary: tuple[int, float] | None = None,
) -> NDArray[np.float32]:
    """Return scores that prefer one token and optionally an invalid one."""
    scores = np.zeros(count, dtype=np.float32)
    scores[preferred] = 10.0
    if secondary is not None:
        index, score = secondary
        scores[index] = score
    return scores


def test_generate_number_accepts_merged_suffix_and_masks_invalid() -> None:
    valid = b" 1}}\n"
    invalid = b" 1}garbage"
    decoder, ids = decoder_for([valid, invalid])
    contexts: list[list[int]] = []

    def logits(context: list[int]) -> NDArray[np.float32]:
        contexts.append(list(context))
        return constant_scores(
            decoder.vocab.n_logits, ids[valid], (ids[invalid], 100.0),
        )

    result = generate_number(
        decoder, [7], logits, following=b"}\n",
    )

    assert result.value == 1.0
    assert result.fragment == valid
    assert result.context_ids == [7, ids[valid]]
    assert result.consumed_delimiter == "}"
    assert result.consumed_suffix == b"}\n"
    assert contexts == [[7]]


def test_integer_boundary_keeps_exact_large_integer_and_history() -> None:
    merged = b" 12345678901234567890}}\n"
    decoder, ids = decoder_for([merged])
    result = generate_number(
        decoder, [4], lambda _: constant_scores(
            decoder.vocab.n_logits, ids[merged],
        ),
        integer=True, following=b"}\n",
    )
    assert result.value == 12345678901234567890
    assert result.fragment == merged
    assert result.context_ids == [4, ids[merged]]
    assert result.consumed_delimiter == "}"
    assert result.consumed_suffix == b"}\n"


def test_pipeline_completes_final_number_in_one_scoring_step() -> None:
    merged = b" 1}}\n"
    decoder, ids = decoder_for([merged])
    definition = FunctionDef(
        name="single", description="Return a numeric value.",
        parameters={"value": TypeDef(type="number")},
        returns=TypeDef(type="number"),
    )
    model_calls: list[str] = []

    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def logits(context: list[int]) -> NDArray[np.float32]:
        raw = bytes(value for value in context if value < 256).decode(
            "utf-8", errors="ignore",
        )
        if raw.endswith('{"name": "'):
            chosen = ord("s")
        elif raw.endswith('{"name": "s'):
            chosen = ord("i")
        elif raw.endswith('{"name": "si'):
            chosen = ord("n")
        elif raw.endswith('{"name": "sin'):
            chosen = ord("g")
        elif raw.endswith('{"name": "sing'):
            chosen = ord("l")
        elif raw.endswith('{"name": "singl'):
            chosen = ord("e")
        elif raw.endswith('{"name": "single'):
            chosen = ord('"')
        elif raw.endswith('"value":'):
            model_calls.append("number")
            return constant_scores(decoder.vocab.n_logits, ids[merged])
        else:
            raise AssertionError(
                f"Unexpected fake-model context tail: {raw[-50:]!r}"
            )
        scores = np.zeros(decoder.vocab.n_logits, dtype=np.float32)
        scores[chosen] = 10.0
        return scores

    caller = FunctionCaller(
        functions={"single": definition}, decoder=decoder, encode=encode,
        logits=logits, allow_number_suffix=True,
    )
    result = caller.call(PromptItem(prompt="Return one"))

    assert result.name == "single"
    assert result.parameters == {"value": 1.0}
    assert model_calls == ["number"]


def test_number_boundary_uses_the_wrapped_number_completion() -> None:
    dfa = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    state = dfa.advance(dfa.initial_state(), b"1}}\n")
    assert state is not None
    assert dfa.is_complete(state)
