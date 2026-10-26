"""Tests for score-based indexed name selection."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.decoding.decoder import Decoder
from src.decoding.indexed import generate_indexed_name
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def make_decoder(raw_tokens: list[bytes]) -> tuple[Decoder, dict[bytes, int]]:
    """Build a tiny byte-token vocabulary for label selection."""
    alphabet = build_byte_to_unicode()
    token_ids = {raw: index for index, raw in enumerate(raw_tokens)}
    vocab = {
        "".join(alphabet[byte] for byte in raw): token_id
        for raw, token_id in token_ids.items()
    }
    table = VocabTable(
        vocab=vocab,
        tokenizer={},
        n_logits=len(raw_tokens),
    )
    return Decoder(vocab=table), token_ids


def scores_for(decoder: Decoder, token_id: int) -> NDArray[np.float32]:
    """Make one token highest and all other logits lower."""
    scores = np.full(decoder.vocab.n_logits, -10.0, dtype=np.float32)
    scores[token_id] = 10.0
    return scores


def test_illegal_highest_score_is_masked_and_valid_label_maps_name() -> None:
    """Only mapped labels can win, even when another token scores higher."""
    decoder, token_ids = make_decoder([b"A", b"B", b"z", b"?"])
    calls: list[list[int]] = []

    def logits(context: list[int]) -> NDArray[np.float32]:
        calls.append(list(context))
        result = scores_for(decoder, token_ids[b"B"])
        result[token_ids[b"?"]] = 1000.0
        return result

    result = generate_indexed_name(
        decoder, [8, 9], logits, {"A": "zebra", "B": "aardvark"}
    )
    assert result.value == "aardvark"
    assert result.fragment == b"B"
    assert result.context_ids == [8, 9, token_ids[b"B"]]
    assert calls == [[8, 9]]


def test_choice_uses_logits_not_original_name_order_or_lexical_order() -> None:
    """The original function names only label the scored choices."""
    decoder, token_ids = make_decoder([b"A", b"B"])

    def logits(_: list[int]) -> NDArray[np.float32]:
        return scores_for(decoder, token_ids[b"A"])

    result = generate_indexed_name(
        decoder, [0], logits, {"A": "zebra", "B": "aardvark"}
    )
    assert result.value == "zebra"
    assert result.fragment == b"A"


def test_cache_is_scoped_to_the_complete_label_mapping() -> None:
    """Reusing one decoder with a new map builds its own legal label set."""
    decoder, token_ids = make_decoder([b"A", b"B", b"C", b"D"])
    first = generate_indexed_name(
        decoder,
        [1],
        lambda _: scores_for(decoder, token_ids[b"A"]),
        {"A": "first", "B": "second"},
    )
    second = generate_indexed_name(
        decoder,
        [1],
        lambda _: scores_for(decoder, token_ids[b"C"]),
        {"C": "third", "D": "fourth"},
    )
    assert first.value == "first"
    assert second.value == "third"
    assert second.fragment == b"C"


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"a": "lowercase"},
        {"AA": "multiple"},
        {"É": "non-ascii"},
        {"A": ""},
    ],
)
def test_invalid_label_maps_are_rejected(labels: dict[str, str]) -> None:
    """Choices must be nonempty uppercase ASCII single-letter labels."""
    decoder, token_ids = make_decoder([b"A"])
    with pytest.raises(ValueError, match="label"):
        generate_indexed_name(
            decoder,
            [0],
            lambda _: scores_for(decoder, token_ids[b"A"]),
            labels,
        )
