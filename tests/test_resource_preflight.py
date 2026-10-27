"""Tests for the stdlib-only local resource preflight utility."""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def preflight() -> Any:
    """Load the ignored helper only for tests that exercise it."""
    return pytest.importorskip("scripts.resource_preflight")


def make_snapshot(
    root: Path, asset_names: tuple[str, ...], revision: str = "rev-a",
) -> Path:
    """Create a tiny fake snapshot with representative asset files."""
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in asset_names:
        (snapshot / name).write_bytes(name.encode("ascii"))
    return snapshot


def test_snapshot_prefers_refs_main_and_falls_back_to_available_config(
    tmp_path: Path, preflight: Any,
) -> None:
    root = tmp_path / "hub"
    preferred = make_snapshot(root, preflight.ASSET_NAMES, "preferred")
    make_snapshot(root, preflight.ASSET_NAMES, "other")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("preferred\n", encoding="utf-8")
    assert preflight.find_snapshot(cache_root=root) == preferred
    (root / "refs" / "main").unlink()
    assert preflight.find_snapshot(cache_root=root) in {
        root / "snapshots" / "other",
        preferred,
    }
    assert preflight.find_snapshot(
        tmp_path / "missing", cache_root=root
    ) is None


def test_asset_links_are_idempotent_and_preserve_collisions(
    tmp_path: Path, preflight: Any,
) -> None:
    snapshot = make_snapshot(tmp_path, preflight.ASSET_NAMES, "source")
    output = tmp_path / "out"
    assets = output / "assets"
    assets.mkdir(parents=True)
    collision = assets / "vocab.json"
    collision.write_text("keep me", encoding="utf-8")

    result = preflight.link_assets(snapshot, output)
    assert result["vocab.json"]["status"] == "collision"
    assert collision.read_text(encoding="utf-8") == "keep me"
    for name in preflight.ASSET_NAMES[1:]:
        assert result[name]["status"] == "linked"
        link = assets / name
        assert link.is_symlink()
        assert link.resolve() == (snapshot / name).resolve()
        assert link.stat().st_ino == (snapshot / name).stat().st_ino

    second = preflight.link_assets(snapshot, output)
    assert second["vocab.json"]["status"] == "collision"
    assert all(
        second[name]["status"] == "already_linked"
        for name in preflight.ASSET_NAMES[1:]
    )
    assert (snapshot / "model.safetensors").read_bytes() == (
        b"model.safetensors"
    )


def test_link_assets_reports_missing_source_without_creating_placeholder(
    tmp_path: Path, preflight: Any,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    result = preflight.link_assets(snapshot, tmp_path / "out")
    assert result["config.json"]["status"] == "missing"
    assert not (tmp_path / "out" / "assets" / "config.json").exists()


def test_meminfo_path_report_and_atomic_json_output(
    tmp_path: Path, preflight: Any,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       1024 kB\n"
        "MemAvailable:    512 kB\n"
        "SwapTotal:      2048 kB\n"
        "SwapFree:        256 kB\n",
        encoding="ascii",
    )
    assert preflight.memory_report(meminfo) == {
        "total_bytes": 1024 * 1024,
        "available_bytes": 512 * 1024,
        "swap_total_bytes": 2048 * 1024,
        "swap_free_bytes": 256 * 1024,
    }

    target = tmp_path / "source.bin"
    target.write_bytes(b"abcd")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(target)
    report = preflight.path_report(alias)
    assert report["bytes"] == target.stat().st_size
    assert report["is_symlink"] is True
    assert report["resolved"] == str(target.resolve())

    destination = preflight.write_report(
        {"paths": [report]}, tmp_path / "report"
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "paths": [report]
    }
    assert sorted(item.name for item in destination.parent.iterdir()) == [
        "environment.json"
    ]


def test_build_report_warns_about_low_memory_and_tmpfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    preflight: Any,
) -> None:
    snapshot = make_snapshot(tmp_path, preflight.ASSET_NAMES, "report")
    monkeypatch.setattr(
        preflight,
        "memory_report",
        lambda: {
            "total_bytes": 4 * 1024**3,
            "available_bytes": 1 * 1024**3,
            "swap_total_bytes": 0,
            "swap_free_bytes": 0,
        },
    )
    monkeypatch.setattr(
        preflight,
        "_mount_type",
        lambda path: "tmpfs" if path == tmp_path else "ext4",
    )
    report = preflight.build_report(
        tmp_path, tmp_path, snapshot, include_nvidia=False
    )
    assert report["snapshot"] == str(snapshot)
    assert report["assets"]["config.json"]["bytes"] == len(b"config.json")
    assert any("below 2 GiB" in warning for warning in report["warnings"])
    assert any("tmpfs" in warning for warning in report["warnings"])
    assert any(
        "inference still needs RAM" in warning
        for warning in report["warnings"]
    )
