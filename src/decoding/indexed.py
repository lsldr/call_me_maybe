"""Scored selection from indexed labels mapped to original names."""

from src.constraints.closed_set_trie import FunctionSetTrie, TrieNode
from src.decoding.decoder import Decoder, Logits
from src.decoding.generators import GeneratedValue


def generate_indexed_name(
    decoder: Decoder,
    ids: list[int],
    logits: Logits,
    labels: dict[str, str],
) -> GeneratedValue:
    """Select a single-letter label by model scores and return its name.

    Args:
        decoder: Shared constrained decoder and vocabulary.
        ids: Encoded context for the current selection.
        logits: Callable that scores the full current token history.
        labels: Mapping from uppercase ASCII labels to original names.

    Returns:
        The selected original name, label fragment, and exact token history.

    Raises:
        ValueError: If the mapping is empty or contains invalid labels.
    """
    if not labels or len(labels) > 26:
        raise ValueError(
            "Indexed labels must contain between 1 and 26 choices."
        )

    trie = FunctionSetTrie()
    for label, name in labels.items():
        if (
            not isinstance(label, str)
            or len(label) != 1
            or not "A" <= label <= "Z"
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError(
                "Each indexed label must be one uppercase ASCII letter "
                "mapped to a nonempty name."
            )
        byte = ord(label)
        child = len(trie.nodes)
        trie.nodes[trie.root].children[byte] = child
        trie.nodes.append(TrieNode(terminal_name=name))

    fragment, final_ids, state = decoder.decode(
        ids,
        logits,
        trie.root,
        trie.advance,
        trie.next_bytes,
        lambda node: trie.completed_name(node) is not None,
        namespace=("indexed", tuple(labels.items())),
    )
    return GeneratedValue(
        value=trie.completed_name(state),
        fragment=fragment,
        context_ids=final_ids,
    )
