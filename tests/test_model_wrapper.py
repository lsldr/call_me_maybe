"""SDK interface regression tests that never load model weights."""

import numpy as np
from numpy.typing import NDArray
import pytest
from typing import cast

from src.llm import model_wrapper


class FakeSDK:
    """Implement only the public SDK signatures used by the wrapper."""

    def __init__(self, model_name: str, device: str) -> None:
        """Accept the SDK's model-loading arguments without loading weights."""

    def get_path_to_vocab_file(self) -> str:
        """Return a dummy vocabulary path."""
        return "vocab.json"

    def get_path_to_merges_file(self) -> str:
        """Return a dummy merges path."""
        return "merges.txt"

    def get_path_to_tokenizer_file(self) -> str:
        """Return a dummy tokenizer path."""
        return "tokenizer.json"

    def encode(self, text: str) -> NDArray[np.int64]:
        """Return a batched sequence with a tensor-compatible interface."""
        return np.array([[0]], dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        """Match the SDK signature, which accepts no keyword options."""
        return "".join(str(token) for token in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> object:
        """Always select token one."""
        return [0.0, 1.0]


class TwoDimensionalLogitsSDK(FakeSDK):
    """Return malformed logits to exercise wrapper shape validation."""

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[list[float]]:
        """Return a matrix instead of the required one-dimensional row."""
        return [[0.0, 1.0]]


def test_decode_and_generate_use_public_sdk_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final generation decoding works without unsupported SDK arguments."""
    monkeypatch.setattr(model_wrapper, "Small_LLM_Model", FakeSDK)
    model = model_wrapper.ModelWrapper()
    assert model.decode([1, 0]) == "10"
    assert model.generate("prompt", max_new_tokens=2) == "11"
    assert model.call_count == 2
    assert model.generate("prompt", stop_token_ids=[1]) == ""
    assert model.call_count == 3


@pytest.mark.parametrize("token_ids", [[True], cast(list[int], [1.5]), [-1]])
def test_decode_rejects_non_integer_or_negative_ids(
    monkeypatch: pytest.MonkeyPatch, token_ids: list[int],
) -> None:
    """Decode accepts only non-negative values whose exact type is int."""
    monkeypatch.setattr(model_wrapper, "Small_LLM_Model", FakeSDK)
    model = model_wrapper.ModelWrapper()
    with pytest.raises(TypeError, match="list of integers"):
        model.decode(token_ids)


def test_logits_reject_boolean_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boolean IDs are rejected even though bool subclasses int."""
    monkeypatch.setattr(model_wrapper, "Small_LLM_Model", FakeSDK)
    model = model_wrapper.ModelWrapper()
    with pytest.raises(ValueError, match="non-negative integers"):
        model.get_logits_from_input_ids([True])


def test_logits_reject_non_vector_sdk_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK output must be one-dimensional before constrained decoding."""
    monkeypatch.setattr(
        model_wrapper, "Small_LLM_Model", TwoDimensionalLogitsSDK
    )
    model = model_wrapper.ModelWrapper()
    with pytest.raises(ValueError, match="one-dimensional row"):
        model.get_logits_from_input_ids([0])
