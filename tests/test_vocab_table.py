"""Regression coverage for validated vocabulary lookup construction."""

from typing import Any

import numpy as np
from pydantic import ValidationError
import pytest

from src.vocab.vocab_table import VocabTable


def test_out_of_range_id_is_a_validation_error() -> None:
    """Bounds fail before indexing the derived byte table."""
    with pytest.raises(ValidationError, match="exceeds n_logits"):
        VocabTable(vocab={"a": 2}, tokenizer={}, n_logits=2)


def test_lookup_tables_are_always_derived() -> None:
    """Injected lookup data cannot replace the actual vocabulary mapping."""
    table = VocabTable(
        vocab={"a": 0, "ab": 2, "b": 3}, tokenizer={}, n_logits=5,
        id_to_bytes=[b"wrong"],
        first_byte_index={9: np.array([99], dtype=np.int32)},
    )
    assert table.id_to_bytes == [b"a", None, b"ab", b"b", None]
    assert set(table.first_byte_index) == {ord("a"), ord("b")}
    assert table.first_byte_index[ord("a")].tolist() == [0, 2]
    assert table.first_byte_index[ord("b")].tolist() == [3]
    assert table.first_byte_index[ord("a")].dtype == np.int32


@pytest.mark.parametrize("entry", [
    {"id": 1}, {"content": "special"},
    {"id": -1, "content": "special"},
    {"id": 2, "content": "special"},
    {"id": True, "content": "special"},
    {"id": "1", "content": "special"},
    {"id": 1, "content": 12},
])
@pytest.mark.parametrize("source", ["tokenizer", "added_tokens"])
def test_invalid_added_tokens(entry: dict[str, Any], source: str) -> None:
    """Added-token identities are validated at both input locations."""
    kwargs: dict[str, Any] = {
        "vocab": {"a": 0}, "tokenizer": {}, "n_logits": 2,
    }
    if source == "tokenizer":
        kwargs["tokenizer"] = {"added_tokens": [entry]}
    else:
        kwargs["added_tokens"] = [entry]
    with pytest.raises(ValidationError):
        VocabTable(**kwargs)


def test_added_token_metadata_and_padding() -> None:
    """Special-token metadata is preserved without populating byte slots."""
    table = VocabTable(
        vocab={"a": 0}, n_logits=3,
        tokenizer={"added_tokens": [
            {"id": 1, "content": "<special>", "special": True},
        ]},
    )
    assert table.added_tokens[0].content == "<special>"
    assert table.added_tokens[0].model_dump()["special"] is True
    assert table.id_to_bytes == [b"a", None, None]
