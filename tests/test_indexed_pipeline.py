"""Model-free integration tests for indexed function selection."""

import numpy as np
from numpy.typing import NDArray

from src.decoding.decoder import Decoder
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def make_function(name: str, *, with_text: bool = False) -> FunctionDef:
    """Return an invented function definition for selection tests."""
    parameters = (
        {"text": TypeDef(type="string", description="Input text")}
        if with_text else {}
    )
    return FunctionDef(
        name=name, description=f"Arbitrary description for {name}",
        parameters=parameters, returns=TypeDef(type="string"),
    )


def make_caller(
    functions: dict[str, FunctionDef],
    *,
    max_new_tokens: int = 128,
    max_context_tokens: int = 1536,
) -> tuple[FunctionCaller, list[tuple[str, str]]]:
    """Create an indexed-style caller driven by scripted byte scores."""
    byte_map = build_byte_to_unicode()
    vocab = VocabTable(
        vocab={byte_map[value]: value for value in range(256)},
        tokenizer={}, n_logits=256,
    )
    contexts: list[tuple[str, str]] = []

    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def logits(ids: list[int]) -> NDArray[np.float32]:
        raw = bytes(ids).decode("utf-8", errors="ignore")
        if "Choose exactly one function for the request." in raw:
            chosen = ord("B")
        elif raw.endswith('"text": "'):
            chosen = ord("o")
        else:
            target = b'ok"}}\n'
            prefix = '"text": "'
            for consumed in range(len(target)):
                if raw.endswith(prefix + target[:consumed].decode("utf-8")):
                    chosen = target[consumed]
                    break
            else:
                raise AssertionError(f"No scripted route for {raw[-100:]!r}")
        scores = np.full(256, -1000, dtype=np.float32)
        scores[chosen] = 1000
        return scores

    caller = FunctionCaller(
        functions=functions,
        decoder=Decoder(
            vocab=vocab, max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
        ),
        encode=encode,
        logits=logits,
        style="compact",
        context_observer=lambda phase, text: contexts.append((phase, text)),
    )
    # The optional mode is currently a pipeline experiment, so opt into it
    # without changing the validated public style literals in this test.
    setattr(caller, "style", "indexed")
    return caller, contexts


def test_indexed_mode_selects_label_then_extracts_from_one_definition(
) -> None:
    functions = {
        "alpha": make_function("alpha"),
        "beta": make_function("beta", with_text=True),
    }
    caller, contexts = make_caller(functions)

    result = caller.call(PromptItem(prompt="Words do not name the function"))

    assert result.name == "beta"
    assert result.parameters == {"text": "ok"}
    name_contexts = [text for phase, text in contexts if phase == "name"]
    assert len(name_contexts) == 1
    assert "A {\"name\":\"alpha\"" in name_contexts[0]
    assert "B {\"name\":\"beta\"" in name_contexts[0]
    argument_contexts = [
        text for phase, text in contexts if phase == "parameter:text"
    ]
    assert len(argument_contexts) == 1
    assert '"name":"beta"' in argument_contexts[0]
    assert '"name":"alpha"' not in argument_contexts[0]


def test_more_than_26_functions_falls_back_to_name_trie() -> None:
    functions = {
        f"fn_{index:02d}": make_function(f"fn_{index:02d}")
        for index in range(27)
    }
    caller, contexts = make_caller(functions, max_context_tokens=5000)

    def trie_logits(ids: list[int]) -> NDArray[np.float32]:
        raw = bytes(ids).decode("utf-8", errors="ignore")
        prefix = '{"name": "'
        target = b'fn_26"'
        for consumed in range(len(target)):
            if raw.endswith(prefix + target[:consumed].decode("ascii")):
                chosen = target[consumed]
                break
        else:
            raise AssertionError(f"No trie route for {raw[-100:]!r}")
        scores = np.full(256, -1000, dtype=np.float32)
        scores[chosen] = 1000
        return scores

    caller.logits = trie_logits
    result = caller.call(PromptItem(prompt="Choose the final registry entry"))

    assert result.name == "fn_26"
    name_contexts = [text for phase, text in contexts if phase == "name"]
    assert len(name_contexts) == 1
    assert '"name": "' in name_contexts[0]
    assert "Choose exactly one function for the request." not in (
        name_contexts[0]
    )
