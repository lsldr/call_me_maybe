"""Model-free tests for profile-aware loading and factory wiring."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.config import RunSettings
from src.llm import model_wrapper
from src.llm.profiles import ModelProfile, get_profile
from src.models.function_def import FunctionDef, TypeDef
from src.pipeline import function_caller as factory
from src.vocab.vocab_table import VocabTable


def _definition() -> FunctionDef:
    return FunctionDef(
        name="echo",
        description="Copy text",
        parameters={"value": TypeDef(type="string")},
        returns=TypeDef(type="string"),
    )


def _toy_profile(
    *, expected_n_logits: int = 4,
    required_tokens: tuple[tuple[str, int], ...] = (
        ("<|im_start|>", 1), ("<|im_end|>", 2),
    ),
) -> ModelProfile:
    return get_profile(
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    ).model_copy(update={
        "expected_n_logits": expected_n_logits,
        "required_tokens": required_tokens,
    })


def _toy_vocab(
    *, n_logits: int = 4,
    marker_tokens: tuple[tuple[str, int], ...] = (
        ("<|im_start|>", 1), ("<|im_end|>", 2),
    ),
) -> VocabTable:
    return VocabTable(
        vocab={"a": 0, "b": 1, "c": 2, "d": 3} if n_logits == 4 else {
            chr(97 + index): index for index in range(n_logits)
        },
        tokenizer={"added_tokens": [
            {"content": content, "id": token_id}
            for content, token_id in marker_tokens
        ]},
        n_logits=n_logits,
    )


class _FactoryModel:
    """Tiny model facade with explicit observable SDK calls."""

    def __init__(self, n_logits: int, calls: dict[str, Any], **kwargs: object):
        self.n_logits = n_logits
        self.calls = calls
        self.calls["model_kwargs"] = kwargs

    def encode(self, text: str) -> list[int]:
        self.calls.setdefault("encodes", []).append(text)
        return [0]

    def logits(self, ids: list[int]) -> NDArray[np.float32]:
        self.calls.setdefault("logit_inputs", []).append(ids)
        return np.zeros(self.n_logits, dtype=np.float32)

    def get_vocab_file_path(self) -> str:
        return "vocab.json"

    def get_tokenizer_file_path(self) -> str:
        return "tokenizer.json"

    def get_merges_file_path(self) -> str:
        return "merges.txt"


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    profile: ModelProfile | None = None,
    vocab: VocabTable | None = None,
    n_logits: int = 4,
    calls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.llm import model_wrapper as wrapper_module

    state = {} if calls is None else calls
    selected_profile = _toy_profile() if profile is None else profile
    monkeypatch.setattr(
        factory, "get_profile", lambda _model: selected_profile,
    )
    monkeypatch.setattr(
        factory, "cached_snapshot", lambda _profile: tmp_path / "cached"
    )
    monkeypatch.setattr(
        factory, "validate_snapshot", lambda _profile, path: Path(path)
    )
    monkeypatch.setattr(
        wrapper_module,
        "ModelWrapper",
        lambda **kwargs: _FactoryModel(n_logits, state, **kwargs),
    )
    table = _toy_vocab() if vocab is None else vocab
    monkeypatch.setattr(
        VocabTable,
        "from_files",
        classmethod(lambda _cls, *_args: table),
    )
    return state


def test_bonus_factory_uses_profile_and_forwards_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = _patch_factory(monkeypatch, tmp_path)
    caller = factory.create_caller(
        {"echo": _definition()},
        RunSettings(
            model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            profile="bonus",
            recovery="retry",
        ),
    )

    assert caller.context_family == "smollm2"
    assert caller.recovery == "retry"
    assert calls["encodes"] == ["Warmup"]
    assert calls["logit_inputs"] == [[0]]
    assert calls["model_kwargs"]["snapshot_path"] == tmp_path / "cached"
    assert calls["model_kwargs"]["model_name"] == (
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    selection = caller._selection_context([_definition()], "copy this")
    arguments = caller._arguments_context(_definition(), "copy this")
    assert selection.endswith('<|im_start|>assistant\n{"name": "')
    assert arguments.endswith(
        '<|im_start|>assistant\n{"name": "echo", "arguments": {'
    )
    for context in (selection, arguments):
        assert "<think>" not in context
        assert "</think>" not in context
        assert "<tool_call>" not in context


def test_bonus_factory_rejects_wrong_logits_and_marker_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_factory(
        monkeypatch, tmp_path, profile=_toy_profile(expected_n_logits=5),
    )
    with pytest.raises(ValueError, match="Logits dimension"):
        factory.create_caller(
            {"echo": _definition()},
            RunSettings(
                model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                profile="bonus",
            ),
        )

    wrong_marker_profile = _toy_profile()
    wrong_marker_vocab = _toy_vocab(
        marker_tokens=(("<|im_start|>", 1), ("<|im_end|>", 3))
    )
    _patch_factory(
        monkeypatch,
        tmp_path,
        profile=wrong_marker_profile,
        vocab=wrong_marker_vocab,
    )
    with pytest.raises(ValueError, match="Model marker differs"):
        factory.create_caller(
            {"echo": _definition()},
            RunSettings(
                model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                profile="bonus",
            ),
        )


def test_custom_snapshot_validation_happens_before_model_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {"model_created": False}
    selected = _toy_profile()
    monkeypatch.setattr(factory, "get_profile", lambda _model: selected)
    custom = tmp_path / "custom-snapshot"

    def reject_snapshot(_profile: ModelProfile, path: Path) -> Path:
        assert path == custom
        raise ValueError("snapshot metadata rejected")

    monkeypatch.setattr(factory, "validate_snapshot", reject_snapshot)

    def forbidden_model(**_: object) -> object:
        calls["model_created"] = True
        raise AssertionError("model should not be constructed")

    monkeypatch.setattr(model_wrapper, "ModelWrapper", forbidden_model)
    with pytest.raises(ValueError, match="snapshot metadata rejected"):
        factory.create_caller(
            {"echo": _definition()},
            RunSettings(
                model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                profile="bonus",
                model_snapshot=custom,
            ),
        )
    assert not calls["model_created"]


@pytest.mark.parametrize(
    "settings, message",
    [
        (
            {"model_name": "HuggingFaceTB/SmolLM2-360M-Instruct"},
            "alternate models require bonus",
        ),
        (
            {
                "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "profile": "bonus",
                "tokenizer": "local",
            },
            "unsupported for",
        ),
        ({"model_name": "unknown/model"}, "Unsupported model ID"),
        (
            {
                "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "profile": "bonus",
                "max_context_tokens": 8193,
            },
            "Context limit exceeds",
        ),
    ],
)
def test_run_settings_rejects_unsupported_profile_combinations(
    settings: dict[str, Any], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RunSettings(**settings)


def test_vocab_hides_reserved_added_tokens_from_byte_grammar() -> None:
    table = VocabTable(
        vocab={"a": 0, "b": 1, "c": 2},
        tokenizer={"added_tokens": [
            {"content": "<|im_start|>", "id": 1},
        ]},
        n_logits=3,
    )
    assert table.id_to_bytes[1] is None
    assert 1 not in table.first_byte_index.get(ord("b"), np.array([]))


def test_mandatory_factory_keeps_legacy_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch_factory(monkeypatch, Path("/tmp/unused"), calls=calls)
    qwen = get_profile("Qwen/Qwen3-0.6B")
    monkeypatch.setattr(factory, "get_profile", lambda _model: qwen)

    def forbidden_snapshot(*_: object) -> Path:
        raise AssertionError("mandatory factory must not validate snapshots")

    monkeypatch.setattr(factory, "validate_snapshot", forbidden_snapshot)
    caller = factory.create_caller(
        {"echo": _definition()}, RunSettings(profile="mandatory")
    )
    assert caller.context_family == "qwen3"
    assert caller.recovery == "none"
    assert "snapshot_path" not in calls["model_kwargs"]


def test_wrapper_uses_local_snapshot_without_hub_path_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for filename in (
        "config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "model.safetensors",
    ):
        (snapshot / filename).write_text("metadata", encoding="utf-8")
    calls: dict[str, Any] = {}

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            calls["constructor"] = kwargs

        def get_path_to_vocab_file(self) -> str:
            raise AssertionError("Hub-only path helper must not be called")

        def get_path_to_merges_file(self) -> str:
            raise AssertionError("Hub-only path helper must not be called")

        def get_path_to_tokenizer_file(self) -> str:
            raise AssertionError("Hub-only path helper must not be called")

    monkeypatch.setattr(model_wrapper, "Small_LLM_Model", FakeSDK)
    model = model_wrapper.ModelWrapper(
        model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        snapshot_path=snapshot,
    )
    resolved = snapshot.resolve()
    assert calls["constructor"] == {
        "model_name": str(resolved),
        "device": "cpu",
        "trust_remote_code": False,
    }
    assert model.get_vocab_file_path() == str(resolved / "vocab.json")
    assert model.get_merges_file_path() == str(resolved / "merges.txt")
    assert model.get_tokenizer_file_path() == str(resolved / "tokenizer.json")


def test_wrapper_preflights_snapshot_before_sdk_constructor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    called = False

    class FakeSDK:
        def __init__(self, **_: object) -> None:
            nonlocal called
            called = True

    monkeypatch.setattr(model_wrapper, "Small_LLM_Model", FakeSDK)
    snapshot = tmp_path / "incomplete"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="Missing model asset"):
        model_wrapper.ModelWrapper(
            model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            snapshot_path=snapshot,
        )
    assert not called
