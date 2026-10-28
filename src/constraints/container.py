"""Incremental byte DFA for schema-constrained nested JSON containers."""

import hashlib
import json
from copy import deepcopy
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.constraints.number_dfa import NumberDFA, NumberState
from src.constraints.string_dfa import StringDFA, StringState
from src.models.function_def import TypeDef
from src.prompting.sanitize import safe_json
from src.schemas.nested import SchemaLimits, array_bounds, validate_schema


class ContainerFrame(BaseModel):
    """One immutable parser frame; only bounded syntax state is retained.

    Attributes:
        node: Compiled schema node ID for this container.
        phase: Current parser phase for this container.
        index: Current property index for objects, or 0 for arrays.
        count: Number of items consumed for arrays, or properties for objects.
        key_offset: Current byte offset into the property key for objects.
        string: Current string DFA state, or None if not parsing a string.
        number: Current number DFA state, or None if not parsing a number.
        literal_offset: Current byte offset into the literal for booleans/null.
        literal: The literal bytes for booleans/null, or None if not parsing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node: int
    phase: str
    index: int = 0
    count: int = 0
    key_offset: int = 0
    string: StringState | None = None
    number: NumberState | None = None
    literal_offset: int = 0
    literal: bytes | None = None


class ContainerState(BaseModel):
    """Hashable continuation state for a nested JSON value.

    This state represents the parser's position within a nested JSON structure,
    including the current container, its properties, and the bytes consumed.

    Attributes:
        stack: A tuple of ContainerFrame objects representing the current
            stack of nested containers.
        started: A boolean indicating whether the root container was started.
        done: A boolean indicating whether the root container was fully parsed.
        suffix: A bytes object representing the suffix of the input that was
            consumed but not yet validated against the following bytes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stack: tuple[ContainerFrame, ...] = ()
    started: bool = False
    done: bool = False
    suffix: bytes = b""


class JSONContainerDFA(BaseModel):
    """Validate a complete array/object value incrementally by bytes.

    Attributes:
        root_schema: The root schema node for the container.
        limits: The schema limits for validation.
        following: The bytes that follow the current input.
        markers: A tuple of strings that mark the positions of the input.
        _compiled_nodes: A tuple of compiled schema nodes for the DFA.
        _compiled_children: A tuple of tuples representing the child node IDs
            for each compiled node.
        _compiled_keys: A tuple of tuples representing the property keys for
            each compiled object node.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    root_schema: TypeDef = Field(alias="schema")
    limits: SchemaLimits = Field(default_factory=SchemaLimits)
    following: bytes = b""
    markers: tuple[str, ...] = ()
    _compiled_nodes: tuple[TypeDef, ...] = PrivateAttr(default=())
    _compiled_children: tuple[tuple[int, ...], ...] = PrivateAttr(default=())
    _compiled_keys: tuple[tuple[bytes, ...], ...] = PrivateAttr(default=())

    def __init__(self, **data: object) -> None:
        """Initialize the DFA and compile its validated schema tree.

        Args:
            **data: Model fields, including the root schema and DFA options.

        Raises:
            ValueError: If the root is not a container or its schema
                is invalid.
        """
        super().__init__(**data)
        if self.root_schema.type not in {"array", "object"}:
            raise ValueError("container DFA root must be an array or object")
        validate_schema(self.root_schema, self.limits)
        snapshot = deepcopy(self.root_schema)
        validate_schema(snapshot, self.limits)
        object.__setattr__(self, "root_schema", snapshot)
        nodes: list[TypeDef] = []
        children: list[tuple[int, ...]] = []
        keys: list[tuple[bytes, ...]] = []

        def compile_node(node: TypeDef) -> int:
            """Append a schema node and its descendants to the compiled tables.

            Args:
                node: Schema node to compile.

            Returns:
                The node's index in the compiled tables.
            """
            node_id = len(nodes)
            nodes.append(node)
            children.append(())
            properties = node.properties or {}
            keys.append(
                tuple(
                    safe_json(key, self.markers).encode("utf-8")
                    for key in properties
                )
            )
            if node.type == "array":
                assert node.items is not None
                child_ids: tuple[int, ...] = (compile_node(node.items),)
            elif node.type == "object":
                child_ids = tuple(
                    compile_node(child) for child in properties.values()
                )
            else:
                child_ids = ()
            children[node_id] = child_ids
            return node_id

        compile_node(snapshot)
        object.__setattr__(self, "_compiled_nodes", tuple(nodes))
        object.__setattr__(self, "_compiled_children", tuple(children))
        object.__setattr__(self, "_compiled_keys", tuple(keys))

    @property
    def identity(self) -> str:
        """Return a stable cache identity that preserves property order."""
        payload = {
            "schema": self.root_schema.model_dump(
                mode="json", exclude_none=False
            ),
            "limits": self.limits.model_dump(mode="json"),
            "following": self.following.hex(),
            "markers": self.markers,
        }
        canonical = json.dumps(payload, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _child(self, node_id: int, index: int) -> int:
        """Return a compiled child ID by array item or object property index.

        Args:
            node_id: Parent node's compiled ID.
            index: Child position; arrays always use their single item schema.

        Returns:
            The compiled ID of the selected child schema.
        """
        return self._compiled_children[node_id][
            0 if self._compiled_nodes[node_id].type == "array" else index
        ]

    def initial_state(self) -> ContainerState:
        """Return the immutable initial parser state.

        Returns:
            A state ready to consume the first JSON byte.
        """
        return ContainerState()

    def advance(
        self, state: ContainerState, raw: bytes
    ) -> ContainerState | None:
        """Consume raw bytes and return the next state, or reject the prefix.

        Args:
            state: The parser state before this token.
            raw: Bytes to validate and consume as one token.

        Returns:
            The immutable advanced state, or ``None`` if any byte is invalid.
        """
        if not raw or state.done:
            return None
        nodes = self._compiled_nodes
        stack = list(state.stack)
        started: bool = state.started
        done: bool = state.done
        suffix = state.suffix
        i = 0
        while i < len(raw):
            byte = raw[i]
            if done:
                candidate = suffix + raw[i:]
                if len(candidate) > len(
                    self.following
                ) or not self.following.startswith(candidate):
                    return None
                suffix = candidate
                i = len(raw)
                break
            if not started:
                if byte in b" \t\r\n":
                    i += 1
                    continue
                wrong_root = (
                    byte == ord("[") and self.root_schema.type != "array"
                ) or (byte == ord("{") and self.root_schema.type != "object")
                if byte not in (ord("["), ord("{")) or wrong_root:
                    return None
                started = True
                stack.append(
                    ContainerFrame(
                        node=0,
                        phase="ARRAY_FIRST" if byte == 91 else "OBJECT_FIRST",
                    )
                )
                i += 1
                continue

            frame = stack[-1]
            node = nodes[frame.node]
            if frame.phase == "STRING":
                assert frame.string is not None
                # Hand the rest of this token to StringDFA in one pass.
                inner = StringDFA().advance(frame.string, raw[i:])
                if inner is not None:
                    if StringDFA().is_complete(inner):
                        stack.pop()
                    else:
                        stack[-1] = frame.model_copy(update={"string": inner})
                    i = len(raw)
                    continue
                inner = StringDFA().advance(frame.string, raw[i:i + 1])
                if inner is None:
                    return None
                if StringDFA().is_complete(inner):
                    stack.pop()
                else:
                    stack[-1] = frame.model_copy(update={"string": inner})
                i += 1
                continue
            if frame.phase == "NUMBER":
                assert frame.number is not None
                if byte in b" \t\r\n,]}" and frame.number.phase in {
                    "ZERO",
                    "INT",
                    "FRAC",
                }:
                    stack.pop()
                    continue
                number = NumberDFA(integer=node.type == "integer").advance(
                    frame.number, bytes([byte])
                )
                if number is None:
                    return None
                stack[-1] = frame.model_copy(update={"number": number})
                i += 1
                continue
            if frame.phase == "LITERAL":
                assert frame.literal is not None
                offset = frame.literal_offset
                if frame.literal[offset:offset + 1] != bytes([byte]):
                    return None
                next_offset = offset + 1
                complete = len(frame.literal) == next_offset
                if complete:
                    stack.pop()
                else:
                    stack[-1] = frame.model_copy(
                        update={"literal_offset": next_offset}
                    )
                i += 1
                continue

            if frame.phase == "OBJECT_KEY":
                if frame.key_offset == 0 and byte in b" \t\r\n":
                    i += 1
                    continue
                if frame.index >= len(self._compiled_keys[frame.node]):
                    return None
                key = self._compiled_keys[frame.node][frame.index]
                offset = frame.key_offset
                if byte != key[offset]:
                    return None
                offset += 1
                if offset == len(key):
                    stack[-1] = frame.model_copy(
                        update={"phase": "OBJECT_COLON", "key_offset": 0}
                    )
                else:
                    stack[-1] = frame.model_copy(update={"key_offset": offset})
                i += 1
                continue

            if frame.phase == "OBJECT_FIRST":
                if byte in b" \t\r\n":
                    i += 1
                    continue
                if byte == ord("}"):
                    if nodes[frame.node].properties or {}:
                        return None
                    stack.pop()
                    i += 1
                    if not stack:
                        done = True
                    continue
                stack[-1] = frame.model_copy(update={"phase": "OBJECT_KEY"})
                continue

            if byte in b" \t\r\n":
                i += 1
                continue
            if frame.phase == "OBJECT_COLON":
                if byte != ord(":"):
                    return None
                stack[-1] = frame.model_copy(update={"phase": "OBJECT_VALUE"})
                i += 1
                continue
            if frame.phase in {
                "ARRAY_FIRST",
                "ARRAY_NEXT",
                "OBJECT_FIRST",
                "OBJECT_VALUE",
            }:
                is_array = node.type == "array"
                if frame.phase in {"ARRAY_FIRST", "OBJECT_FIRST"} and byte == (
                    ord("]") if is_array else ord("}")
                ):
                    minimum, _ = (
                        array_bounds(node, self.limits) if is_array else (0, 0)
                    )
                    if is_array and frame.count < minimum:
                        return None
                    stack.pop()
                    i += 1
                    if not stack:
                        done = True
                    continue
                if is_array:
                    _, maximum = array_bounds(node, self.limits)
                    if frame.count >= maximum:
                        return None
                child_id = self._child(
                    frame.node, frame.count if is_array else frame.index
                )
                # mark container parent as awaiting its separator before
                # pushing child
                waiting = "ARRAY_SEPARATOR" if is_array else "OBJECT_SEPARATOR"
                count = frame.count + 1 if is_array else frame.count
                stack[-1] = frame.model_copy(
                    update={"phase": waiting, "count": count}
                )
                child = nodes[child_id]
                opener = byte
                if child.type in {"array", "object"}:
                    wanted = ord("[") if child.type == "array" else ord("{")
                    if opener != wanted:
                        return None
                    stack.append(
                        ContainerFrame(
                            node=child_id,
                            phase=(
                                "ARRAY_FIRST"
                                if wanted == 91
                                else "OBJECT_FIRST"
                            ),
                        )
                    )
                elif child.type == "string":
                    if opener != ord('"'):
                        return None
                    stack.append(
                        ContainerFrame(
                            node=child_id, phase="STRING", string=StringState()
                        )
                    )
                elif child.type in {"integer", "number"}:
                    stack.append(
                        ContainerFrame(
                            node=child_id, phase="NUMBER", number=NumberState()
                        )
                    )
                    continue
                else:
                    if child.type == "boolean":
                        literal = b"true" if byte == ord("t") else b"false"
                    else:
                        literal = b"null"
                    if byte != literal[0]:
                        return None
                    stack.append(
                        ContainerFrame(
                            node=child_id,
                            phase="LITERAL",
                            literal_offset=1,
                            literal=literal,
                        )
                    )
                i += 1
                continue
            if frame.phase in {"ARRAY_SEPARATOR", "OBJECT_SEPARATOR"}:
                is_array = frame.phase == "ARRAY_SEPARATOR"
                close = ord("]") if is_array else ord("}")
                if byte == close:
                    if is_array:
                        minimum, _ = array_bounds(node, self.limits)
                        if frame.count < minimum:
                            return None
                    elif frame.index + 1 != len(node.properties or {}):
                        return None
                    stack.pop()
                    i += 1
                    if not stack:
                        done = True
                    continue
                if byte != ord(","):
                    return None
                phase = "ARRAY_NEXT" if is_array else "OBJECT_KEY"
                stack[-1] = frame.model_copy(
                    update={
                        "phase": phase,
                        "index": frame.index + (0 if is_array else 1),
                    }
                )
                i += 1
                continue
            return None
        return ContainerState(
            stack=tuple(stack), started=started, done=done, suffix=suffix
        )

    def next_bytes(self, state: ContainerState) -> set[int]:
        """Return byte values whose one-byte extension remains valid.

        Args:
            state: The parser state to inspect.

        Returns:
            The set of legal next byte values.
        """
        return {
            b
            for b in range(256)
            if self.advance(state, bytes([b])) is not None
        }

    def is_complete(self, state: ContainerState) -> bool:
        """Return whether the root value has closed.

        Args:
            state: The parser state to inspect.

        Returns:
            ``True`` once the root container's closing byte was consumed.
        """
        return state.done
