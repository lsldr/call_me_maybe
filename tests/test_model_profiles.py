"""Unit tests for pinned model metadata profiles."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.llm.profiles import (
    ModelProfile,
    cached_snapshot,
    get_profile,
    validate_snapshot,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_profile(files: dict[str, bytes]) -> ModelProfile:
    return ModelProfile(
        model_id="org/model",
        revision="a" * 40,
        context_family="smollm2",
        expected_n_logits=17,
        max_context_tokens=64,
        supported_tokenizers=("sdk",),
        required_tokens=(("<|im_start|>", 1), ("<|im_end|>", 2)),
        asset_sha256=tuple(
            (name, _sha256(files[name])) for name in files
        ),
    )


def _write_snapshot(path: Path, files: dict[str, bytes]) -> None:
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).write_bytes(content)
    (path / "model.safetensors").write_bytes(b"weights-placeholder")


def _snapshot_files(vocab_size: int = 17) -> dict[str, bytes]:
    return {
        "config.json": json.dumps({"vocab_size": vocab_size}).encode(),
        "tokenizer.json": b"tokenizer metadata",
        "tokenizer_config.json": b"tokenizer config",
        "vocab.json": b"vocabulary",
        "merges.txt": b"merges",
    }


def test_profiles_pin_metadata_and_smollm2_sdk_only() -> None:
    qwen = get_profile("Qwen/Qwen3-0.6B")
    smol = get_profile("HuggingFaceTB/SmolLM2-360M-Instruct")

    assert qwen.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert qwen.expected_n_logits == 151936
    assert qwen.max_context_tokens == 40960
    assert ("<think>", 151667) in qwen.required_tokens
    assert ("</think>", 151668) in qwen.required_tokens
    assert smol.revision == "a10cc1512eabd3dde888204e902eca88bddb4951"
    assert smol.expected_n_logits == 49152
    assert smol.max_context_tokens == 8192
    assert smol.required_tokens == (("<|im_start|>", 1), ("<|im_end|>", 2))
    assert smol.supported_tokenizers == ("sdk",)

    with pytest.raises(ValidationError):
        smol.model_id = "other/model"
    with pytest.raises(ValueError, match="Unsupported model ID"):
        get_profile("unknown/model")


def test_profile_rejects_malformed_revision() -> None:
    with pytest.raises(ValidationError):
        ModelProfile(
            model_id="org/model",
            revision="not-a-revision",
            context_family="smollm2",
            expected_n_logits=4,
            max_context_tokens=8,
            supported_tokenizers=("sdk",),
            required_tokens=(("<|im_start|>", 1),),
            asset_sha256=(("config.json", "0" * 64),),
        )


def test_cached_snapshot_uses_hf_cache_override(tmp_path: Path) -> None:
    profile = get_profile("HuggingFaceTB/SmolLM2-360M-Instruct")
    path = cached_snapshot(profile, tmp_path)
    assert path == (
        tmp_path / "models--HuggingFaceTB--SmolLM2-360M-Instruct"
        / "snapshots" / profile.revision
    )


def test_validate_snapshot_accepts_matching_metadata(tmp_path: Path) -> None:
    files = _snapshot_files()
    profile = _synthetic_profile(files)
    snapshot = tmp_path / profile.revision
    _write_snapshot(snapshot, files)
    assert validate_snapshot(profile, snapshot) == snapshot.resolve()


def test_validate_snapshot_checks_revision_and_files(tmp_path: Path) -> None:
    files = _snapshot_files()
    profile = _synthetic_profile(files)
    wrong_revision = tmp_path / ("b" * 40)
    _write_snapshot(wrong_revision, files)
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_snapshot(profile, wrong_revision)

    missing = tmp_path / profile.revision
    missing.mkdir()
    with pytest.raises(ValueError, match="asset is missing"):
        validate_snapshot(profile, missing)


def test_validate_snapshot_checks_asset_hashes(tmp_path: Path) -> None:
    files = _snapshot_files()
    profile = _synthetic_profile(files)
    snapshot = tmp_path / profile.revision
    _write_snapshot(snapshot, files)
    (snapshot / "vocab.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch for vocab.json"):
        validate_snapshot(profile, snapshot)


def test_validate_snapshot_checks_config_and_nonempty_weights(
    tmp_path: Path,
) -> None:
    files = _snapshot_files(vocab_size=18)
    profile = _synthetic_profile(files)
    snapshot = tmp_path / profile.revision
    _write_snapshot(snapshot, files)
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        validate_snapshot(profile, snapshot)

    # Refresh the hash to get past metadata hash verification and exercise the
    # explicit nonempty weight check.
    empty_weight_snapshot = tmp_path / ("c" * 40)
    empty_weight_snapshot.mkdir()
    for name, content in files.items():
        (empty_weight_snapshot / name).write_bytes(content)
    (empty_weight_snapshot / "model.safetensors").touch()
    empty_profile = profile.model_copy(update={"revision": "c" * 40})
    with pytest.raises(ValueError, match="Nonempty model.safetensors"):
        validate_snapshot(empty_profile, empty_weight_snapshot)


def test_cached_snapshot_honors_cache_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_profile("Qwen/Qwen3-0.6B")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cached_snapshot(profile) == (
        tmp_path / "hub-cache" / "models--Qwen--Qwen3-0.6B"
        / "snapshots" / profile.revision
    )
