"""Trie over the JSON spellings of a closed set of function names."""

import json

from pydantic import BaseModel, ConfigDict, Field


class TrieNode(BaseModel):
    """A trie node addressed by its index in the parent trie."""

    model_config = ConfigDict(extra="forbid")

    children: dict[int, int] = Field(default_factory=dict)
    terminal_name: str | None = None


class FunctionSetTrie(BaseModel):
    """A replaceable set of function names represented as a byte trie."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[TrieNode] = Field(default_factory=lambda: [TrieNode()])
    root: int = 0

    def set_targets(self, names: list[str]) -> None:
        """Replace targets with JSON UTF-8 paths after the opening quote."""
        self.nodes = [TrieNode()]
        self.root = 0
        for name in names:
            node_index = self.root
            spelling = json.dumps(name, ensure_ascii=False).encode("utf-8")[1:]
            for byte in spelling:
                child_index = self.nodes[node_index].children.get(byte)
                if child_index is None:
                    child_index = len(self.nodes)
                    self.nodes[node_index].children[byte] = child_index
                    self.nodes.append(TrieNode())
                node_index = child_index
            self.nodes[node_index].terminal_name = name

    def search(self, name: str) -> bool:
        """Return whether *name* is one of the current targets."""
        spelling = json.dumps(name, ensure_ascii=False).encode("utf-8")[1:]
        node = self.advance(self.root, spelling)
        return node is not None and self.completed_name(node) == name

    def advance(self, node: int, raw_bytes: bytes) -> int | None:
        """Return the node reached by all bytes without changing trie state."""
        if node < 0 or node >= len(self.nodes):
            return None
        for byte in raw_bytes:
            child = self.nodes[node].children.get(byte)
            if child is None:
                return None
            node = child
        return node

    def next_bytes(self, node: int) -> set[int]:
        """Return valid next bytes at *node*."""
        if node < 0 or node >= len(self.nodes):
            raise ValueError("node index is out of range")
        return set(self.nodes[node].children)

    def unique_completion(self, node: int) -> tuple[str, bytes] | None:
        """Find a sole terminal below ``node`` and its remaining byte suffix.

        The traversal is iterative so long names do not depend on Python's
        recursion limit. It stops as soon as a second terminal is found.
        """
        if node < 0 or node >= len(self.nodes):
            raise ValueError("node index is out of range")
        stack: list[tuple[int, bytes]] = [(node, b"")]
        found: tuple[str, bytes] | None = None
        while stack:
            current, suffix = stack.pop()
            name = self.nodes[current].terminal_name
            if name is not None:
                if found is not None:
                    return None
                found = (name, suffix)
            for byte, child in self.nodes[current].children.items():
                stack.append((child, suffix + bytes([byte])))
        return found

    def completed_name(self, node: int) -> str | None:
        """Return the original function name if *node* is terminal."""
        if node < 0 or node >= len(self.nodes):
            raise ValueError("node index is out of range")
        return self.nodes[node].terminal_name
