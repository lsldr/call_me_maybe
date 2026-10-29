"""Run greedy generation and constrained decoding with the public SDK."""

import os
from pathlib import Path
from typing import Any, Literal

from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)


def _default_device() -> Literal["cpu", "cuda", "mps"]:
    """Resolve default device from CMM_DEVICE or fallback to cpu."""
    dev = os.getenv("CMM_DEVICE", "cpu").lower()
    if dev in ("cpu", "cuda", "mps"):
        return dev  # type: ignore[return-value]
    return "cpu"


class ModelWrapper(BaseModel):
    """Validate model settings and manage one loaded SDK instance."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    model_name: str = Field(
        default="Qwen/Qwen3-0.6B",
        min_length=1,
        description="Name or Hugging Face hub ID of the target model.",
    )
    snapshot_path: Path | None = None
    device: Literal["cpu", "cuda", "mps"] = Field(
        default_factory=_default_device,
        description="Execution device for model inference.",
    )
    call_count: int = Field(
        default=0,
        ge=0,
        description="Number of forward logits passes executed.",
    )

    _sdk: Small_LLM_Model = PrivateAttr()
    _vocab_file: str = PrivateAttr(default="")
    _merges_file: str = PrivateAttr(default="")
    _tokenizer_file: str = PrivateAttr(default="")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name is non-empty and stripped."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Model name cannot be empty or whitespace.")
        return stripped

    def model_post_init(self, __context: Any) -> None:
        """Instantiate the SDK model and cache asset paths once.

        Args:
            __context: Optional initialization context from Pydantic.
        """
        if self.snapshot_path is None:
            self._sdk = Small_LLM_Model(
                model_name=self.model_name,
                device=self.device,
            )
            self._vocab_file = self._sdk.get_path_to_vocab_file()
            self._merges_file = self._sdk.get_path_to_merges_file()
            self._tokenizer_file = self._sdk.get_path_to_tokenizer_file()
        else:
            snapshot = self.snapshot_path.resolve()
            for filename in (
                "config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors",
            ):
                if not (snapshot / filename).is_file():
                    raise ValueError(
                        f"Missing model asset: {snapshot / filename}"
                    )
            # The SDK accepts a local model directory publicly. Its Hub-only
            # path helpers do not, so use those same referenced local assets.
            self._sdk = Small_LLM_Model(
                model_name=str(snapshot),
                device=self.device,
                trust_remote_code=False,
            )
            self._vocab_file = str(snapshot / "vocab.json")
            self._merges_file = str(snapshot / "merges.txt")
            self._tokenizer_file = str(snapshot / "tokenizer.json")

    def encode(self, text: str) -> list[int]:
        """Convert text into a flat list of token IDs.

        Args:
            text: Text to tokenize, including prompt framing.

        Returns:
            Token IDs in original sequence order.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Expected str for text, got {type(text).__name__}"
            )
        ids: list[int] = self._sdk.encode(text)[0].tolist()
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of token IDs, stripping special tokens.

        Args:
            token_ids: Sequence of token IDs to decode.

        Returns:
            Decoded text string.
        """
        if not isinstance(token_ids, list) or any(
            type(token) is not int or token < 0 for token in token_ids
        ):
            raise TypeError("token_ids must be a list of integers.")
        return self._sdk.decode(token_ids)

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> NDArray[np.float32]:
        """Get one next-token logits row for candidate token positions.

        Args:
            input_ids: Full context sequence so far.

        Returns:
            Float32 numpy array indexed by token ID.

        Raises:
            ValueError: If input_ids is empty or contains negative IDs.
        """
        if not input_ids:
            raise ValueError("input_ids must contain at least one token ID.")
        if any(type(tok) is not int or tok < 0 for tok in input_ids):
            raise ValueError("input_ids must contain non-negative integers.")

        self.call_count += 1
        raw_logits = self._sdk.get_logits_from_input_ids(input_ids)
        scores = np.asarray(raw_logits, dtype=np.float32)
        if scores.ndim != 1 or not scores.size:
            raise ValueError(
                "SDK logits must be a nonempty one-dimensional row."
            )
        return scores

    def logits(self, input_ids: list[int]) -> NDArray[np.float32]:
        """Alias for get_logits_from_input_ids per plan spec."""
        return self.get_logits_from_input_ids(input_ids)

    def get_vocab_file_path(self) -> str:
        """Get the absolute path to vocab.json."""
        return self._vocab_file

    def get_merges_file_path(self) -> str:
        """Get the absolute path to merges.txt."""
        return self._merges_file

    def get_tokenizer_file_path(self) -> str:
        """Get the absolute path to tokenizer.json."""
        return self._tokenizer_file
