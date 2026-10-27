"""Allocate distinct, ordered names for generated run traces."""

from datetime import date
import os
from pathlib import Path
import re


def allocate_trace(
    label: str, directory: Path = Path("data/traces"),
) -> Path:
    """Reserve a numbered trace filename without overwriting another run.

    Args:
        label: Short description of the workload, sanitized for filenames.
        directory: Directory in which generated traces are kept.

    Returns:
        The path of an exclusively created empty file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:64]
    safe_label = safe_label or "run"
    numbers = [
        int(match.group(1))
        for path in directory.glob("*.jsonl")
        if (match := re.match(r"^(\d+)_", path.name)) is not None
    ]
    number = max(numbers, default=0) + 1
    while True:
        name = f"{number}_{date.today().isoformat()}_{safe_label}.jsonl"
        path = directory / name
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                 0o600)
        except FileExistsError:
            number += 1
            continue
        os.close(descriptor)
        return path
