def build_byte_to_unicode() -> dict[int, str]:
    """
    Build a GPT-2 byte alphabet to unicode mapping.

    Returns:
        Dictionary mapping byte values (0-255) to unicode characters
        that map to GPT-2's byte-level BPE tokenizer.
    """
    preserved_bytes = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )

    table = {byte: chr(byte) for byte in preserved_bytes}
    next_codepoint = 256
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(next_codepoint)
            next_codepoint += 1

    return table


def unicode_to_bytes() -> dict[str, int]:
    """Build a unicode to GPT-2 byte alphabet mapping."""
    return {char: byte for byte, char in build_byte_to_unicode().items()}
