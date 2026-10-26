"""Tests for byte-wise traversal of JSON function names."""

import json

import pytest

from src.constraints.closed_set_trie import FunctionSetTrie


def test_shared_prefix_and_completion_compete_with_continuation() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["fn_get", "fn_get_area"])
    prefix = trie.advance(trie.root, b"fn_get")
    assert prefix is not None
    assert trie.completed_name(prefix) is None
    assert trie.next_bytes(prefix) == {ord('"'), ord("_")}
    assert trie.advance(prefix, b"_area\"") is not None
    assert trie.advance(trie.root, b'fn_get"') is not None


def test_single_name_empty_name_and_overshoot() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["a"])
    root = trie.root
    end = trie.advance(root, b'a"')
    assert end is not None
    assert trie.completed_name(end) == "a"
    assert trie.advance(root, b'a",') is None
    trie.set_targets([""])
    end = trie.advance(trie.root, b'"')
    assert end is not None
    assert trie.completed_name(end) == ""


def test_targets_replace_previous_tree_and_search_compatibility() -> None:
    trie = FunctionSetTrie()
    assert not trie.search("fn_greet")
    trie.set_targets(["fn_greet", 'say"hello', "zażółć"])
    assert trie.search("fn_greet")
    assert trie.search('say"hello')
    assert trie.search("zażółć")
    assert not trie.search("fn_")
    trie.set_targets(["replacement"])
    assert not trie.search("fn_greet")
    assert trie.search("replacement")


def test_json_escaping_and_utf8_paths() -> None:
    names = ['say"hello', r"back\\slash", "zażółć"]
    trie = FunctionSetTrie()
    trie.set_targets(names)
    for name in names:
        path = json.dumps(name, ensure_ascii=False).encode("utf-8")[1:]
        terminal = trie.advance(trie.root, path)
        assert terminal is not None
        assert trie.completed_name(terminal) == name


def test_token_can_cross_branch_and_trials_do_not_mutate_tree() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["fn_get", "fn_get_area"])
    prefix = trie.advance(trie.root, b"fn_get")
    assert prefix is not None
    original_children = dict(trie.nodes[prefix].children)
    terminal = trie.advance(prefix, b"_area\"")
    assert terminal is not None
    assert trie.completed_name(terminal) == "fn_get_area"
    assert trie.advance(prefix, b"_missing") is None
    assert trie.advance(prefix, b"\"junk") is None
    assert trie.nodes[prefix].children == original_children
    assert trie.completed_name(prefix) is None


def test_unique_completion_stops_only_after_one_terminal_remains() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["a", "ab"])

    assert trie.unique_completion(trie.root) is None
    prefix = trie.advance(trie.root, b"a")
    assert prefix is not None
    assert trie.unique_completion(prefix) is None
    after_b = trie.advance(prefix, b"b")
    assert after_b is not None
    assert trie.unique_completion(after_b) == ("ab", b'"')
    terminal = trie.advance(after_b, b'"')
    assert terminal is not None
    assert trie.unique_completion(terminal) == ("ab", b"")


def test_unique_completion_returns_remaining_json_suffix() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["only-name"])
    prefix = trie.advance(trie.root, b"on")
    assert prefix is not None
    assert trie.unique_completion(prefix) == ("only-name", b"ly-name\"")


def test_unique_completion_handles_unicode_json_name_bytes() -> None:
    trie = FunctionSetTrie()
    name = 'météo"'
    trie.set_targets([name])
    full_path = json.dumps(name, ensure_ascii=False).encode("utf-8")[1:]
    prefix = trie.advance(trie.root, full_path[:3])
    assert prefix is not None
    assert trie.unique_completion(prefix) == (name, full_path[3:])


def test_unique_completion_is_iterative_for_long_names() -> None:
    trie = FunctionSetTrie()
    name = "n" * 3000
    trie.set_targets([name])
    found = trie.unique_completion(trie.root)
    assert found == (name, b"n" * 3000 + b'"')


def test_unique_completion_does_not_mutate_and_tracks_replacement() -> None:
    trie = FunctionSetTrie()
    trie.set_targets(["first"])
    before = [
        (dict(node.children), node.terminal_name) for node in trie.nodes
    ]
    assert trie.unique_completion(trie.root) == ("first", b"first\"")
    assert [
        (dict(node.children), node.terminal_name) for node in trie.nodes
    ] == before

    trie.set_targets(["next"])
    assert trie.unique_completion(trie.root) == ("next", b"next\"")
    assert trie.unique_completion(0) == ("next", b"next\"")


@pytest.mark.parametrize("node", [-1, 1])
def test_unique_completion_rejects_invalid_nodes(node: int) -> None:
    trie = FunctionSetTrie()
    with pytest.raises(ValueError, match="out of range"):
        trie.unique_completion(node)
