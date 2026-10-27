"""Typed adapters around the shared constrained decoding loop."""

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from src.constraints.closed_set_trie import FunctionSetTrie, TrieNode
from src.constraints.container import JSONContainerDFA
from src.constraints.number_dfa import NumberDFA
from src.constraints.number_boundary import NumberBoundaryDFA
from src.constraints.string_dfa import StringDFA
from src.constraints.string_boundary import StringBoundaryDFA
from src.constraints.string_value import StringValueDFA
from src.decoding.decoder import Decoder, Logits
from src.models.function_def import TypeDef
from src.schemas.nested import SchemaLimits, validate_value

Scalar = str | int | float | bool | None


class GeneratedValue(BaseModel):
    """Keep a typed value separate from its model-context spelling."""

    model_config = ConfigDict(strict=True, extra="forbid")
    value: JsonValue
    fragment: bytes
    context_ids: list[int]
    consumed_delimiter: str = ""
    consumed_suffix: bytes = b""
    forced_suffix: bytes = b""


def generate_container(
    decoder: Decoder, ids: list[int], logits: Logits,
    schema: TypeDef, *, limits: SchemaLimits,
    following: bytes = b"", markers: tuple[str, ...] = (),
) -> GeneratedValue:
    """Generate one recursive value with the shared scored decoding loop."""
    grammar = JSONContainerDFA(
        schema=schema, limits=limits, following=following, markers=markers,
    )
    fragment, final_ids, state = decoder.decode(
        ids, logits, grammar.initial_state(), grammar.advance,
        grammar.next_bytes, grammar.is_complete,
        namespace=("container", grammar.identity),
    )
    content = fragment[:-len(state.suffix)] if state.suffix else fragment
    try:
        value = json.loads(content.decode("utf-8"))
        compact = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise decoder.failure(
            "invalid_value", str(error),
            generated_tokens=len(final_ids) - len(ids),
            generated_bytes=len(fragment), context_tokens=len(final_ids),
            logits_calls=len(final_ids) - len(ids),
        ) from error
    if len(compact) > limits.max_bytes:
        raise decoder.failure(
            "byte_limit", "value exceeds maximum serialized byte size",
            generated_tokens=len(final_ids) - len(ids),
            generated_bytes=len(fragment), context_tokens=len(final_ids),
            logits_calls=len(final_ids) - len(ids),
        )
    try:
        validate_value(value, schema, limits)
    except ValueError as error:
        raise decoder.failure(
            "invalid_value", str(error),
            generated_tokens=len(final_ids) - len(ids),
            generated_bytes=len(fragment), context_tokens=len(final_ids),
            logits_calls=len(final_ids) - len(ids),
        ) from error
    return GeneratedValue(
        value=value, fragment=fragment, context_ids=final_ids,
        consumed_suffix=state.suffix,
    )


def generate_name(
    decoder: Decoder, ids: list[int], logits: Logits,
    names: list[str],
    *, finish_unique: bool = False,
) -> GeneratedValue:
    """Select an available name using only constrained model scores."""
    trie = FunctionSetTrie()
    trie.set_targets(names)
    fragment, final_ids, state = decoder.decode(
        ids, logits, 0, trie.advance, trie.next_bytes,
        lambda node: (
            trie.completed_name(node) is not None
            or (finish_unique and trie.unique_completion(node) is not None)
        ),
        namespace=("names", tuple(names)),
    )
    completion = trie.unique_completion(state)
    if completion is None:
        raise ValueError("Name decoding did not resolve a unique function.")
    name, suffix = completion
    return GeneratedValue(
        value=name, fragment=fragment, context_ids=final_ids,
        forced_suffix=suffix,
    )


def generate_string(
    decoder: Decoder, ids: list[int], logits: Logits,
    *, following: bytes | None = None, include_opening: bool = False,
    max_steps: int | None = None,
) -> GeneratedValue:
    """Decode a JSON string strictly, including escape interpretation."""
    if include_opening:
        full_value = StringValueDFA(following=following or b"")
        fragment, final_ids, full_state = decoder.decode(
            ids, logits, full_value.initial_state(), full_value.advance,
            full_value.next_bytes, full_value.is_complete,
            namespace=("string-value", following),
            max_steps=max_steps,
        )
        suffix = full_state.inner.suffix
        content = fragment[:-len(suffix)] if suffix else fragment
        return GeneratedValue(
            value=json.loads(content.decode("utf-8")),
            fragment=fragment, context_ids=final_ids,
            consumed_suffix=suffix,
        )
    if following is not None:
        boundary = StringBoundaryDFA(following=following)
        fragment, final_ids, state = decoder.decode(
            ids, logits, boundary.initial_state(), boundary.advance,
            boundary.next_bytes, boundary.is_complete,
            namespace=("string-boundary", following),
            max_steps=max_steps,
        )
        content = fragment[:-len(state.suffix)] if state.suffix else fragment
        return GeneratedValue(
            value=json.loads((b'"' + content).decode("utf-8")),
            fragment=fragment, context_ids=final_ids,
            consumed_suffix=state.suffix,
        )
    dfa = StringDFA()
    fragment, final_ids, _ = decoder.decode(
        ids, logits, dfa.initial_state(), dfa.advance,
        dfa.next_bytes, dfa.is_complete, namespace="string",
        max_steps=max_steps,
    )
    value = json.loads((b'"' + fragment).decode("utf-8"))
    return GeneratedValue(
        value=value, fragment=fragment, context_ids=final_ids,
    )


def generate_number(
    decoder: Decoder, ids: list[int], logits: Logits,
    *, integer: bool = False, delimiter: Literal[",", "}"] = "}",
    following: bytes | None = None,
) -> GeneratedValue:
    """Generate a decimal and its delimiter; reject float overflow."""
    dfa = NumberDFA(
        integer=integer, delimiter=delimiter, allow_leading_space=True,
    )
    suffix = b""
    if following is not None:
        boundary = NumberBoundaryDFA(number=dfa, following=following)
        fragment, final_ids, state = decoder.decode(
            ids, logits, boundary.initial_state(), boundary.advance,
            boundary.next_bytes, boundary.is_complete,
            namespace=("number-boundary", integer, delimiter, following),
        )
        suffix = state.suffix
    else:
        fragment, final_ids, _ = decoder.decode(
            ids, logits, dfa.initial_state(), dfa.advance,
            dfa.next_bytes, dfa.is_complete,
            namespace=("number", integer, delimiter, "optional-space"),
        )
    content = fragment[:-len(suffix)] if suffix else fragment
    spelling = content[:-1].decode("ascii")
    value = int(spelling) if integer else float(spelling)
    if not integer and not math.isfinite(value):
        raise decoder.failure(
            "nonfinite_number", "Generated number is not finite.",
            generated_tokens=len(final_ids) - len(ids),
            generated_bytes=len(fragment), context_tokens=len(final_ids),
            logits_calls=len(final_ids) - len(ids),
        )
    return GeneratedValue(
        value=value, fragment=fragment, context_ids=final_ids,
        consumed_delimiter=delimiter,
        consumed_suffix=suffix,
    )


def generate_boolean(
    decoder: Decoder, ids: list[int], logits: Logits,
) -> GeneratedValue:
    """Score both boolean literals, including their initial space."""
    trie = FunctionSetTrie()
    for literal in (" true", " false"):
        node = 0
        for byte in literal.encode("ascii"):
            if byte not in trie.nodes[node].children:
                trie.nodes[node].children[byte] = len(trie.nodes)
                trie.nodes.append(TrieNode())
            node = trie.nodes[node].children[byte]
        trie.nodes[node].terminal_name = literal
    fragment, final_ids, _ = decoder.decode(
        ids, logits, 0, trie.advance, trie.next_bytes,
        lambda node: trie.completed_name(node) is not None,
        namespace="boolean",
    )
    return GeneratedValue(
        value=json.loads(fragment), fragment=fragment, context_ids=final_ids,
    )
