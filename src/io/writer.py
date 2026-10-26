"""Atomic JSON file writing utilities."""

from contextlib import suppress
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json_file(file_path: str | Path, data: object) -> None:
    """Write UTF-8 JSON atomically, preserving the destination on failure.

    Create missing parent directories and a unique temporary file beside the
    destination. Reject nonfinite numbers. Propagate serialization, filesystem,
    and interrupt errors without letting temporary-file cleanup mask them.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                data, temp_file, ensure_ascii=False, allow_nan=False, indent=4,
            )
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
