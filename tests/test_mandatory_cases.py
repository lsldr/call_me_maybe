"""Check the copied original prompts and separate demonstration inputs."""

import json
import re
from pathlib import Path

from src.evaluation.cases import load_cases
from src.io.reader import load_function_definitions
from src.models.prompt import FunctionCallResult


def test_original_prompts_are_an_exact_copy_of_input() -> None:
    """The baseline copy keeps its original bytes and prompt order."""
    original = Path("data/evaluation/original_prompts.json")
    source = Path("data/input/function_calling_tests.json")
    assert original.read_bytes() == source.read_bytes()
    assert len(json.loads(original.read_text())) == 11


def test_demo_suite_has_valid_independent_expected_calls() -> None:
    """Each of the ten varied demo requests has a valid function contract."""
    cases = load_cases("benchmark")
    originals = {row["prompt"] for row in json.loads(
        Path("data/evaluation/original_prompts.json").read_text())}
    functions = load_function_definitions(
        "data/input/functions_definition.json",
    )
    assert len(cases) == 10
    assert len({case.prompt for case in cases}) == 10
    assert not originals.intersection(case.prompt for case in cases)
    assert sum(bool(case.probes) for case in cases) == 3
    for case in cases:
        FunctionCallResult.model_validate({
            "prompt": case.prompt, "name": case.expected.name,
            "parameters": case.expected.parameters,
        }, context={"functions": functions})
        if case.probes:
            re.compile(str(case.expected.parameters["regex"]))


def test_empty_and_substring_examples_state_intended_values() -> None:
    """Empty input and substring regex are authored explicitly, not guessed."""
    cases = {case.id: case for case in load_cases("benchmark")}
    assert cases["reverse-empty-string"].expected.parameters["s"] == ""
    substring = cases["regex-substring"]
    assert substring.expected.parameters["regex"] == "sun"
    assert re.sub("sun", "moon", "sunshine and sunset") == (
        "moonshine and moonset"
    )
