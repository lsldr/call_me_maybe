"""Vocabulary and token lookup tables for constrained decoding."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy import int32
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    TypeAdapter,
)
from pydantic_core import from_json

from src.vocab.byte_table import unicode_to_bytes


class AddedToken(BaseModel):
    """Validated added-token identity, preserving tokenizer metadata."""

    model_config = ConfigDict(strict=True, extra="allow")

    id: int = Field(ge=0)
    content: str = Field(min_length=1)


class VocabTable(BaseModel):
    """Store vocabulary mappings and precomputed byte lookup tables."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
    )

    vocab: dict[str, int] = Field(
        ...,
        min_length=1,
        description="Mapping from token spelling to token ID.",
    )
    tokenizer: dict[str, Any] = Field(
        ...,
        description="Raw parsed tokenizer configuration.",
    )
    n_logits: int = Field(
        ...,
        gt=0,
        description="Total logits dimension / token capacity.",
    )
    added_tokens: list[AddedToken] = Field(
        default_factory=list,
        description="Special or added tokens from tokenizer.json.",
    )
    id_to_bytes: list[bytes | None] = Field(
        default_factory=list,
        description="Map of token ID to byte representation.",
    )
    first_byte_index: dict[int, NDArray[int32]] = Field(
        default_factory=dict,
        description="Grouped token IDs indexed by their first byte.",
    )

    @field_validator("vocab")
    @classmethod
    def validate_vocab(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate token spellings, byte alphabet, and ID uniqueness.

        Raises:
            ValueError: If vocab is empty, contains duplicate IDs, or contains
                characters outside the GPT-2 byte alphabet.
        """
        if not v:
            raise ValueError("Vocabulary mapping cannot be empty.")

        alphabet = unicode_to_bytes()
        seen_ids: dict[int, str] = {}

        for token, token_id in v.items():
            if not isinstance(token, str):
                raise TypeError(
                    f"Token spelling must be str, got {type(token).__name__}."
                )
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(
                    f"Token ID for '{token}' must be a non-negative integer, "
                    f"got {token_id}."
                )
            if token_id in seen_ids:
                raise ValueError(
                    f"Duplicate token ID {token_id}: mapped by both "
                    f"'{seen_ids[token_id]}' and '{token}'."
                )
            seen_ids[token_id] = token

            for char in token:
                if char not in alphabet:
                    raise ValueError(
                        f"Token spelling '{token}' contains character "
                        f"{char!r} (U+{ord(char):04X}) outside the byte "
                        "alphabet."
                    )
        return v

    @field_validator("tokenizer")
    @classmethod
    def validate_tokenizer(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate tokenizer configuration structure."""
        TypeAdapter(list[AddedToken]).validate_python(
            v.get("added_tokens", []), strict=True
        )
        return v

    @model_validator(mode="after")
    def validate_ids_within_logits(self) -> "VocabTable":
        """Validate IDs before deriving both lookup tables from the vocab.

        Raises:
            ValueError: If any token ID exceeds or equals n_logits.
        """
        for token, token_id in self.vocab.items():
            if token_id >= self.n_logits:
                raise ValueError(
                    f"Token ID {token_id} for '{token}' exceeds "
                    f"n_logits ({self.n_logits})."
                )
        tokenizer_tokens = TypeAdapter(list[AddedToken]).validate_python(
            self.tokenizer.get("added_tokens", []), strict=True
        )
        if not self.added_tokens:
            self.added_tokens = tokenizer_tokens
        for added_token in [*self.added_tokens, *tokenizer_tokens]:
            if added_token.id >= self.n_logits:
                raise ValueError(
                    f"Added token ID {added_token.id} exceeds "
                    f"n_logits ({self.n_logits})."
                )
        self.id_to_bytes = self.build_id_to_bytes(
            self.vocab, self.n_logits
        )
        self.first_byte_index = self.build_first_byte_index(
            self.id_to_bytes
        )
        return self

    @staticmethod
    def build_id_to_bytes(
        vocab: dict[str, int], n_logits: int
    ) -> list[bytes | None]:
        """Build mapping from token IDs to their raw byte representations.

        Args:
            vocab: Mapping from token string to token ID.
            n_logits: Size of logits array (total IDs).

        Returns:
            List indexed by token ID containing decoded bytes or None.
        """
        inverse = unicode_to_bytes()
        id_to_bytes: list[bytes | None] = [None] * n_logits

        for spelling, token_id in vocab.items():
            id_to_bytes[token_id] = bytes(inverse[char] for char in spelling)

        return id_to_bytes

    @staticmethod
    def build_first_byte_index(
        id_to_bytes: list[bytes | None],
    ) -> dict[int, NDArray[int32]]:
        """Group token IDs by their initial byte.

        Args:
            id_to_bytes: Mapping of token ID to byte sequence.

        Returns:
            Dictionary mapping initial byte to sorted int32 numpy array.
        """
        groups: dict[int, list[int]] = {}
        for token_id, byte_seq in enumerate(id_to_bytes):
            if byte_seq:
                groups.setdefault(byte_seq[0], []).append(token_id)

        return {
            first_byte: np.array(token_ids, dtype=int32)
            for first_byte, token_ids in groups.items()
        }

    @classmethod
    def from_files(
        cls,
        vocab_file: str | Path,
        tokenizer_file: str | Path,
        n_logits: int,
    ) -> "VocabTable":
        """Load files, extract metadata, and instantiate VocabTable.

        Args:
            vocab_file: Path to `vocab.json`.
            tokenizer_file: Path to `tokenizer.json`.
            n_logits: Total logit count.

        Returns:
            A fully validated VocabTable instance with built tables.

        Raises:
            FileNotFoundError: If a file path does not exist.
            ValueError: If JSON data or structure is invalid.
        """
        vocab_path = Path(vocab_file)
        tokenizer_path = Path(tokenizer_file)

        if not vocab_path.is_file():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        vocab_data = from_json(vocab_path.read_bytes())
        tokenizer_data = from_json(tokenizer_path.read_bytes())

        if not isinstance(vocab_data, dict):
            raise ValueError(
                f"Expected JSON object in {vocab_path}, got {type(vocab_data)}"
            )
        if not isinstance(tokenizer_data, dict):
            raise ValueError(
                f"Expected JSON object in {tokenizer_path}, "
                f"got {type(tokenizer_data)}"
            )

        added_tokens = tokenizer_data.get("added_tokens", [])
        if not isinstance(added_tokens, list):
            raise ValueError("'added_tokens' in tokenizer must be a list.")

        # Keep metadata, not a second copy of the large BPE vocab/merge table.
        tokenizer_data.pop("model", None)

        # The model validator checks bounds before constructing lookup tables.
        return cls(
            vocab=vocab_data,
            tokenizer=tokenizer_data,
            n_logits=n_logits,
            added_tokens=added_tokens,
        )
