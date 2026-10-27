"""Tests for local byte-BPE encode/decode behavior."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.tokenization.local import ByteBPETokenizer
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import AddedToken, VocabTable


PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501


def fixture(
    tmp_path: Path,
    *,
    pattern: str = PATTERN,
    added: list[dict[str, Any]] | None = None,
    unsupported: bool = False,
) -> ByteBPETokenizer:
    bytechars = build_byte_to_unicode()
    vocab = {char: byte for byte, char in bytechars.items()}
    vocab[bytechars[ord("a")] + bytechars[ord("b")]] = 256
    additions = added or []
    metadata: dict[str, Any] = {
        "version": "1.0",
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern},
                "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False,
                "trim_offsets": False, "use_regex": False},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},  # noqa: E501
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},  # noqa: E501
        "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]], "dropout": None,  # noqa: E501
                  "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",  # noqa: E501
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False},  # noqa: E501
        "added_tokens": additions,
    }
    if unsupported:
        metadata["decoder"]["use_regex"] = True
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(metadata), encoding="utf-8")
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("#version: 0.2\na b\n", encoding="utf-8")
    table = VocabTable(
        vocab=vocab, tokenizer={}, n_logits=260,
        added_tokens=[
            AddedToken(id=item["id"], content=item["content"])
            for item in additions
        ],
    )
    return ByteBPETokenizer(vocab=table, tokenizer_path=tokenizer_path, merges_path=merges_path)  # noqa: E501


def added(content: str, token_id: int, special: bool) -> dict[str, Any]:
    return {"id": token_id, "content": content, "single_word": False, "lstrip": False,  # noqa: E501
            "rstrip": False, "normalized": False, "special": special}


def test_merges_and_unicode_normalization(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    assert tokenizer.encode("ab") == [256]
    assert tokenizer.encode("e\u0301") == tokenizer.encode("é")
    assert tokenizer.decode(tokenizer.encode("é")) == "é"


def test_merge_rank_precedence_and_leftmost_tie(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    alphabet = build_byte_to_unicode()
    tokenizer.vocab.vocab[alphabet[ord("b")] + alphabet[ord("c")]] = 257
    tokenizer.vocab.vocab[alphabet[ord(
        "a")] + alphabet[ord("b")] + alphabet[ord("c")]] = 258
    tokenizer.vocab.n_logits = 260
    tokenizer.vocab.id_to_bytes = VocabTable.build_id_to_bytes(
        tokenizer.vocab.vocab, tokenizer.vocab.n_logits
    )
    tokenizer.merges_path.write_text(
        "#version: 0.2\na b\nb c\nab c\n", encoding="utf-8"
    )
    tokenizer._ranks = {
        ("a", "b"): 0, ("b", "c"): 1, ("ab", "c"): 2,
    }
    assert tokenizer.encode("abc") == [258]
    assert tokenizer.encode("abab") == [256, 256]


def test_heap_merges_long_pieces(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    alphabet = build_byte_to_unicode()
    tokenizer.vocab.vocab[alphabet[ord("a")] * 2] = 257
    tokenizer.vocab.n_logits = 260
    tokenizer.vocab.id_to_bytes = VocabTable.build_id_to_bytes(
        tokenizer.vocab.vocab, tokenizer.vocab.n_logits
    )
    tokenizer.merges_path.write_text(
        "#version: 0.2\na a\n", encoding="utf-8"
    )
    tokenizer._ranks = {("a", "a"): 0}
    encoded = tokenizer.encode("a" * 4096)
    assert encoded == [257] * 2048


def test_utf8_bytes_can_span_token_boundaries(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    encoded = tokenizer.encode("🙂")
    assert len(encoded) == 4
    assert tokenizer.decode(encoded) == "🙂"
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode(encoded[:1])
    assert tokenizer.decode(encoded[:1], errors="replace") == "�"


def test_added_tokens_match_before_normalization_and_longest_first(
    tmp_path: Path,
) -> None:
    additions = [added("<", 257, False), added("<x>", 258, True)]
    tokenizer = fixture(tmp_path, added=additions)
    assert tokenizer.encode("<x>e\u0301") == [258, *tokenizer.encode("é")]
    assert tokenizer.decode([258]) == "<x>"
    assert tokenizer.decode([258], skip_special_tokens=True) == ""
    assert tokenizer.decode([257], skip_special_tokens=True) == "<"


def test_unsupported_bytelevel_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported tokenizer option"):
        fixture(tmp_path, unsupported=True)


def test_added_token_requires_metadata_identity_and_boolean_flags(
    tmp_path: Path,
) -> None:
    descriptor = added("<x>", 257, True)
    tokenizer = fixture(tmp_path, added=[descriptor])
    tokenizer.vocab.added_tokens = []
    tokenizer_path = tmp_path / "tokenizer.json"
    with pytest.raises(ValueError, match="does not match VocabTable"):
        ByteBPETokenizer(
            vocab=tokenizer.vocab, tokenizer_path=tokenizer_path,
            merges_path=tmp_path / "merges.txt",
        )
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    data["added_tokens"][0]["single_word"] = 0
    tokenizer_path.write_text(json.dumps(data), encoding="utf-8")
    table = VocabTable(
        vocab={char: byte for byte, char in build_byte_to_unicode().items()},
        tokenizer={}, n_logits=260,
        added_tokens=[AddedToken(id=257, content="<x>")],
    )
    with pytest.raises(ValueError, match="added-token option"):
        ByteBPETokenizer(
            vocab=table, tokenizer_path=tokenizer_path,
            merges_path=tmp_path / "merges.txt",
        )


def test_model_vocab_must_match_shared_vocab(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
    tokenizer_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="model vocab does not match"):
        ByteBPETokenizer(
            vocab=tokenizer.vocab, tokenizer_path=tokenizer_path,
            merges_path=tmp_path / "merges.txt",
        )


def test_model_merge_order_must_match_merges_file(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    alphabet = build_byte_to_unicode()
    token_ba = alphabet[ord("b")] + alphabet[ord("a")]
    tokenizer.vocab.vocab[token_ba] = 257
    tokenizer.vocab.n_logits = 260
    tokenizer.vocab.id_to_bytes = VocabTable.build_id_to_bytes(
        tokenizer.vocab.vocab, tokenizer.vocab.n_logits
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    data["model"]["vocab"] = tokenizer.vocab.vocab
    data["model"]["merges"] = [["a", "b"], ["b", "a"]]
    tokenizer_path.write_text(json.dumps(data), encoding="utf-8")
    tokenizer.merges_path.write_text(
        "#version: 0.2\nb a\na b\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rank order does not match"):
        ByteBPETokenizer(
            vocab=tokenizer.vocab, tokenizer_path=tokenizer_path,
            merges_path=tokenizer.merges_path,
        )


def test_decode_rejects_bool_ids_and_non_list(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path)
    with pytest.raises(TypeError, match="list"):
        tokenizer.decode(cast(list[int], (1,)))
    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.decode([True])


def test_isolated_split_preserves_unmatched_gaps(tmp_path: Path) -> None:
    tokenizer = fixture(tmp_path, pattern=r"\p{L}+")
    assert tokenizer.decode(tokenizer.encode("!ab?")) == "!ab?"
