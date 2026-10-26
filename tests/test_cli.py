"""Exercise the CLI in subprocesses without loading model weights."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the module using the current test environment."""
    return subprocess.run(
        [sys.executable, "-m", "src", *args], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )


def write_inputs(tmp_path: Path, prompts: object) -> tuple[Path, Path, Path]:
    """Prepare a minimal valid registry and the requested prompts."""
    functions = tmp_path / "functions.json"
    functions.write_text(json.dumps([{
        "name": "hello", "description": "Say hello", "parameters": {},
        "returns": {"type": "string"},
    }]))
    inputs = tmp_path / "input.json"
    inputs.write_text(json.dumps(prompts))
    return functions, inputs, tmp_path / "output.json"


def invoke(paths: tuple[Path, Path, Path]) -> subprocess.CompletedProcess[str]:
    """Pass the three file paths to the command-line flags."""
    functions, inputs, output = paths
    return run_cli(
        "--functions_definition", str(functions),
        "--input", str(inputs), "--output", str(output),
    )


def test_help() -> None:
    """Help works without importing or initializing the model SDK."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--functions_definition" in result.stdout
    assert result.stderr == ""


def test_empty_prompts(tmp_path: Path) -> None:
    """Empty input writes a valid empty output and exits successfully."""
    paths = write_inputs(tmp_path, [])
    result = invoke(paths)
    assert result.returncode == 0, result.stderr
    assert json.loads(paths[2].read_text()) == []
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("file_index", [0, 1])
@pytest.mark.parametrize("failure", ["missing", "broken", "schema"])
def test_bad_inputs(
    tmp_path: Path, file_index: int, failure: str,
) -> None:
    """Both input files fail cleanly without overwriting existing output."""
    paths = write_inputs(tmp_path, [])
    if failure == "missing":
        paths[file_index].unlink()
    else:
        paths[file_index].write_text("{" if failure == "broken" else "{}")
    paths[2].write_text("original")
    result = invoke(paths)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "[call_me_maybe] error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert paths[2].read_text() == "original"


def test_nonempty_inputs_without_functions(tmp_path: Path) -> None:
    """Report absent functions without attempting real model generation."""
    paths = write_inputs(tmp_path, [{"prompt": "Hello"}])
    paths[0].write_text("[]")
    paths[2].write_text("original")
    result = invoke(paths)
    assert result.returncode == 1
    assert "No available functions" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert paths[2].read_text() == "original"


def test_output_failure(tmp_path: Path) -> None:
    """Writer failures stay inside the application error boundary."""
    paths = write_inputs(tmp_path, [])
    paths[2].mkdir()
    result = invoke(paths)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "[call_me_maybe] error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert paths[2].is_dir()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "functions.json", "input.json", "output.json",
    ]
