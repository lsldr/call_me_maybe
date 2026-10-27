"""Model-free end-to-end checks for empty JSON string generation."""

import numpy as np
from numpy.typing import NDArray
from src.decoding.decoder import Logits

from src.decoding.decoder import Decoder
from src.decoding.generators import generate_string
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def decoder_for(tokens: list[bytes]) -> tuple[Decoder, dict[bytes, int]]:
    """Build a decoder with byte tokens and selected merged tokens."""
    byte_map = build_byte_to_unicode()
    raw_tokens = [bytes([value]) for value in range(256)] + tokens
    vocab = {
        "".join(byte_map[byte] for byte in raw): token_id
        for token_id, raw in enumerate(raw_tokens)
    }
    return Decoder(vocab=VocabTable(
        vocab=vocab, tokenizer={}, n_logits=len(raw_tokens),
    )), {raw: index for index, raw in enumerate(raw_tokens)}


def prefer(decoder: Decoder, token_id: int) -> Logits:
    """Return scores that select one token at every decoding step."""
    def logits(_: list[int]) -> NDArray[np.float32]:
        scores = np.zeros(decoder.vocab.n_logits, dtype=np.float32)
        scores[token_id] = 1.0
        return scores
    return logits


def test_empty_string_after_prefilled_opening_quote() -> None:
    quote = b'"'
    decoder, ids = decoder_for([quote])

    result = generate_string(decoder, [7], prefer(decoder, ids[quote]))

    assert result.value == ""
    assert result.fragment == quote
    assert result.context_ids == [7, ids[quote]]


def test_empty_string_with_scored_opening_quote_and_merged_close() -> None:
    empty = b'""'
    decoder, ids = decoder_for([empty])

    result = generate_string(
        decoder, [7], prefer(decoder, ids[empty]), include_opening=True,
    )

    assert result.value == ""
    assert result.fragment == empty
    assert result.context_ids == [7, ids[empty]]


def test_empty_string_consumes_merged_boundary_suffix() -> None:
    merged = b'"", '
    decoder, ids = decoder_for([merged])

    result = generate_string(
        decoder, [7], prefer(decoder, ids[merged]), following=b", ",
        include_opening=True,
    )

    assert result.value == ""
    assert result.fragment == merged
    assert result.consumed_suffix == b", "
    assert result.context_ids == [7, ids[merged]]


def test_scored_empty_string_boundary_handles_json_escapes() -> None:
    spelling = b'"line\\n\\u0041"'
    decoder, ids = decoder_for([spelling])

    result = generate_string(
        decoder, [9], prefer(decoder, ids[spelling]), include_opening=True,
    )

    assert result.value == "line\nA"
    assert result.fragment == spelling
