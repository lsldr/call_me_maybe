"""Check atomic replacement and failure preservation without model loading."""

import json
from pathlib import Path

import pytest

from src.io import writer


def test_replace_and_unicode(tmp_path: Path) -> None:
    """Replace an existing output with readable UTF-8 JSON."""
    path = tmp_path / "result.json"
    path.write_text("old", encoding="utf-8")
    data = {"message": "Zażółć 🌍"}
    writer.write_json_file(path, data)
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert "Zażółć 🌍" in path.read_text(encoding="utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_bare_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filename without a directory writes in the working directory."""
    monkeypatch.chdir(tmp_path)
    writer.write_json_file("result.json", [])
    assert json.loads(Path("result.json").read_text()) == []


def test_create_directories(tmp_path: Path) -> None:
    """Create nested output directories as needed."""
    path = tmp_path / "nested" / "output" / "result.json"
    writer.write_json_file(path, [])
    assert json.loads(path.read_text()) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_preserves_destination(tmp_path: Path, value: float) -> None:
    """Reject nonfinite values and remove partially serialized files."""
    path = tmp_path / "result.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        writer.write_json_file(path, [1, value])
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rename failure keeps the original and removes the temporary file."""
    path = tmp_path / "result.json"
    path.write_text("original")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("src.io.writer.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        writer.write_json_file(path, [1])
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_creation_failure_preserves_error(tmp_path: Path) -> None:
    """A parent that is a file fails without a cleanup exception."""
    parent = tmp_path / "file"
    parent.write_text("original")
    with pytest.raises(FileExistsError):
        writer.write_json_file(parent / "result.json", [])
    assert parent.read_text() == "original"


def test_write_failure_and_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup cannot hide a write error, even if removing the temp fails."""
    path = tmp_path / "result.json"
    path.write_text("original")

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr("src.io.writer.json.dump", fail_dump)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="disk full"):
        writer.write_json_file(path, [])
    assert path.read_text() == "original"


def test_temp_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening a temporary file can fail before a cleanup path exists."""
    path = tmp_path / "result.json"
    path.write_text("original")

    def fail_open(*args: object, **kwargs: object) -> None:
        raise PermissionError("cannot create temp")

    monkeypatch.setattr(writer, "NamedTemporaryFile", fail_open)
    with pytest.raises(PermissionError, match="cannot create temp"):
        writer.write_json_file(path, [])
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_interrupt_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted serialization preserves output and removes its temp."""
    path = tmp_path / "result.json"
    path.write_text("original")

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("src.io.writer.json.dump", interrupt)
    with pytest.raises(KeyboardInterrupt):
        writer.write_json_file(path, [])
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
