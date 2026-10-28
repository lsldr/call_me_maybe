"""Pinned metadata profiles for supported local language models."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

TokenizerName = Literal["sdk", "local"]
ContextFamily = Literal["qwen3", "smollm2"]


class ModelProfile(BaseModel):
    """Immutable expected metadata for one pinned model revision."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    context_family: ContextFamily
    expected_n_logits: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)
    supported_tokenizers: tuple[TokenizerName, ...] = Field(min_length=1)
    required_tokens: tuple[tuple[str, int], ...] = Field(min_length=1)
    asset_sha256: tuple[tuple[str, str], ...] = Field(min_length=1)


_QWEN_ASSET_HASHES = (
    (
        "config.json",
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    ),
    (
        "tokenizer.json",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    (
        "vocab.json",
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
)
_SMOL_SHARED_ASSET_HASHES = (
    (
        "tokenizer.json",
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    ),
    (
        "vocab.json",
        "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    ),
    (
        "merges.txt",
        "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    ),
)


def _smol_asset_hashes(
    config_hash: str,
    tokenizer_config_hash: str = (
        "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"
    ),
) -> tuple[tuple[str, str], ...]:
    """Combine model-specific metadata with shared SmolLM tokenizer assets."""
    return (
        ("config.json", config_hash),
        *_SMOL_SHARED_ASSET_HASHES[:1],
        ("tokenizer_config.json", tokenizer_config_hash),
        *_SMOL_SHARED_ASSET_HASHES[1:],
    )


_SMOL_ASSET_HASHES = _smol_asset_hashes(
    "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee"
)

PROFILES: Mapping[str, ModelProfile] = MappingProxyType({
    "Qwen/Qwen3-0.6B": ModelProfile(
        model_id="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        context_family="qwen3",
        expected_n_logits=151936,
        max_context_tokens=40960,
        supported_tokenizers=("sdk", "local"),
        required_tokens=(
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
            ("<tool_call>", 151657),
            ("</tool_call>", 151658),
            ("<think>", 151667),
            ("</think>", 151668),
        ),
        asset_sha256=_QWEN_ASSET_HASHES,
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ModelProfile(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        revision="a10cc1512eabd3dde888204e902eca88bddb4951",
        context_family="smollm2",
        expected_n_logits=49152,
        max_context_tokens=8192,
        supported_tokenizers=("sdk",),
        required_tokens=(
            ("<|im_start|>", 1),
            ("<|im_end|>", 2),
        ),
        asset_sha256=_SMOL_ASSET_HASHES,
    ),
    "HuggingFaceTB/SmolLM2-135M-Instruct": ModelProfile(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        context_family="smollm2",
        expected_n_logits=49152,
        max_context_tokens=8192,
        supported_tokenizers=("sdk",),
        required_tokens=(
            ("<|im_start|>", 1),
            ("<|im_end|>", 2),
        ),
        asset_sha256=_smol_asset_hashes(
            "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"
        ),
    ),
    "HuggingFaceTB/SmolLM-135M-Instruct": ModelProfile(
        model_id="HuggingFaceTB/SmolLM-135M-Instruct",
        revision="fcc320f490e08fdb4b99d935b2c58d40bf35b0d0",
        context_family="smollm2",
        expected_n_logits=49152,
        max_context_tokens=2048,
        supported_tokenizers=("sdk",),
        required_tokens=(
            ("<|im_start|>", 1),
            ("<|im_end|>", 2),
        ),
        asset_sha256=_smol_asset_hashes(
            "d89dd5a56c63ac3d2f6cea098ad8aebd94e5fc4e4060fb61cd7e0e8bd0b678bb",
            "75668bda11eb0d5ece137b4eb3fb2ad679f847fd19846b726369a26d92f9f28e",
        ),
    ),
})

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def get_profile(model_id: str) -> ModelProfile:
    """Return the registered profile for an exact model ID."""
    try:
        return PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model ID: {model_id!r}") from exc


def _default_cache_root() -> Path:
    """Resolve the conventional Hugging Face Hub cache directory."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_snapshot(
    profile: ModelProfile, cache_root: Path | None = None,
) -> Path:
    """Return the conventional local snapshot path for a profile revision."""
    root = _default_cache_root() if cache_root is None else cache_root
    repo_directory = "models--" + profile.model_id.replace("/", "--")
    return root / repo_directory / "snapshots" / profile.revision


def _sha256(path: Path) -> str:
    """Hash a metadata file without loading its entire contents into memory.

    Args:
        path: File whose bytes are checked against the pinned profile.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(profile: ModelProfile, path: Path) -> Path:
    """Validate cached metadata and weights without importing model libs."""
    snapshot = path.resolve()
    if not snapshot.is_dir():
        raise ValueError(f"Model snapshot directory is missing: {snapshot}")
    if not _REVISION_PATTERN.fullmatch(snapshot.name):
        raise ValueError(
            "Snapshot directory name must be a 40-character revision: "
            f"{snapshot}"
        )
    if snapshot.name != profile.revision:
        raise ValueError(
            f"Snapshot revision mismatch: expected {profile.revision}, "
            f"got {snapshot.name}"
        )

    for asset_name, expected_hash in profile.asset_sha256:
        asset = snapshot / asset_name
        if not asset.is_file():
            raise ValueError(f"Required snapshot asset is missing: {asset}")
        actual_hash = _sha256(asset)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Snapshot asset hash mismatch for {asset_name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    weights = snapshot / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError(f"Nonempty model.safetensors is required: {weights}")

    import json

    config_path = snapshot / "config.json"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model config: {config_path}") from exc
    vocab_size = config.get("vocab_size") if isinstance(config, dict) else None
    if vocab_size != profile.expected_n_logits:
        raise ValueError(
            f"Model config vocab_size mismatch: expected "
            f"{profile.expected_n_logits}, got {vocab_size!r}"
        )
    return snapshot
