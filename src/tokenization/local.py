"""A small local implementation of the pinned byte-level BPE profile."""

import json
import heapq
import unicodedata
from pathlib import Path
from typing import Literal, cast

import regex
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


class ByteBPETokenizer(BaseModel):
    """Byte BPE tokenizer driven by a validated tokenizer.json and merges file."""  # noqa: E501

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)

    vocab: VocabTable
    tokenizer_path: Path
    merges_path: Path

    _pattern: regex.Pattern = PrivateAttr()
    _ranks: dict[tuple[str, str], int] = PrivateAttr()
    _added: list[dict[str, object]] = PrivateAttr()
    _bytes_to_unicode: dict[int, str] = PrivateAttr(
        default_factory=build_byte_to_unicode
    )
    _unicode_to_bytes: dict[str, int] = PrivateAttr()

    @model_validator(mode="after")
    def _load_profile(self) -> "ByteBPETokenizer":
        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file not found: {self.tokenizer_path}"
            )
        if not self.merges_path.is_file():
            raise FileNotFoundError(
                f"Merges file not found: {self.merges_path}"
            )
        with self.tokenizer_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError("tokenizer.json must contain an object")
        self._validate_profile(metadata, self.vocab.n_logits)
        model = metadata["model"]
        if not isinstance(model, dict):
            raise ValueError("Unsupported BPE model component")
        model_vocab = model.get("vocab")
        if (
            not isinstance(model_vocab, dict)
            or any(
                type(token_id) is not int for token_id in model_vocab.values()
            )
            or model_vocab != self.vocab.vocab
        ):
            raise ValueError(
                "tokenizer.json model vocab does not match VocabTable"
            )
        self._ranks = self._read_merges(model.get("merges"))
        pattern = metadata["pre_tokenizer"]["pretokenizers"][0]["pattern"][
            "Regex"
        ]  # noqa: E501
        self._pattern = regex.compile(pattern)
        self._added = metadata.get("added_tokens", [])
        # Discard the duplicate model vocab and merge tables after validation.
        metadata.pop("model", None)
        inverse = build_byte_to_unicode()
        self._unicode_to_bytes = {char: byte for byte, char in inverse.items()}
        return self

    def _validate_profile(
        self, data: dict[str, object], n_logits: int
    ) -> None:
        def require(
            obj: object, key: str, expected: object, where: str
        ) -> dict[str, object]:  # noqa: E501
            if not isinstance(obj, dict):
                raise ValueError(f"Unsupported tokenizer component at {where}")
            value = obj.get(key)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"Unsupported tokenizer option {where}.{key}: {value!r}"
                )
            return cast(dict[str, object], obj)

        if data.get("version") != "1.0":
            raise ValueError("Unsupported tokenizer version")
        if (
            data.get("truncation") is not None
            or data.get("padding") is not None
        ):  # noqa: E501
            raise ValueError(
                "Tokenizer truncation and padding must be disabled"
            )
        norm = require(data.get("normalizer"), "type", "NFC", "normalizer")
        if set(norm) != {"type"}:
            raise ValueError("Unsupported normalizer options")
        pre = require(
            data.get("pre_tokenizer"), "type", "Sequence", "pre_tokenizer"
        )
        if set(pre) != {"type", "pretokenizers"}:
            raise ValueError("Unsupported Sequence options")
        children = pre.get("pretokenizers")
        if not isinstance(children, list) or len(children) != 2:
            raise ValueError("Unsupported pre_tokenizer sequence")
        split = require(
            children[0], "type", "Split", "pre_tokenizer.pretokenizers[0]"
        )
        if set(split) != {"type", "pattern", "behavior", "invert"}:
            raise ValueError("Unsupported Split options")
        require(split, "behavior", "Isolated", "pre_tokenizer.Split")
        require(split, "invert", False, "pre_tokenizer.Split")
        pattern = split.get("pattern")
        if (
            not isinstance(pattern, dict)
            or set(pattern) != {"Regex"}
            or not isinstance(pattern["Regex"], str)
        ):  # noqa: E501
            raise ValueError("Unsupported pre_tokenizer split pattern")
        flags = {
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        for section in ("pre_tokenizer", "decoder", "post_processor"):
            value = data.get(section)
            if section == "pre_tokenizer":
                if (
                    not isinstance(value, dict)
                    or value.get("type") != "Sequence"
                ):  # noqa: E501
                    raise ValueError(f"Unsupported {section}")
                bytelevel = (
                    value.get("pretokenizers", [])[1]
                    if len(value.get("pretokenizers", [])) == 2
                    else None
                )
            else:
                bytelevel = value
            if (
                not isinstance(bytelevel, dict)
                or bytelevel.get("type") != "ByteLevel"
            ):  # noqa: E501
                raise ValueError(f"Unsupported {section} component")
            if section == "pre_tokenizer" and set(bytelevel) != {
                "type",
                *flags,
            }:
                raise ValueError("Unsupported ByteLevel pre_tokenizer options")
            for key, expected in flags.items():
                require(bytelevel, key, expected, f"{section}.ByteLevel")
            if set(bytelevel) != {"type", *flags}:
                raise ValueError(f"Unsupported {section} ByteLevel options")
        model = require(data.get("model"), "type", "BPE", "model")
        expected_model: dict[str, object] = {
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }
        for key, option_value in expected_model.items():
            require(model, key, option_value, "model.BPE")
        if set(model) != {"type", "vocab", "merges", *expected_model}:
            raise ValueError("Unsupported BPE model options")
        if set(data) - {
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        }:  # noqa: E501
            raise ValueError("Unknown tokenizer.json component")
        tokens = data.get("added_tokens", [])
        if not isinstance(tokens, list):
            raise ValueError("added_tokens must be a list")
        seen_ids: set[int] = set()
        seen_contents: set[str] = set()
        for token in tokens:
            if not isinstance(token, dict):
                raise ValueError("Invalid added token")
            expected_keys = {
                "id",
                "content",
                "special",
                "single_word",
                "lstrip",
                "rstrip",
                "normalized",
            }
            if set(token) != expected_keys:
                raise ValueError("Unknown or missing added-token options")
            token_id = token.get("id")
            content = token.get("content")
            if type(token_id) is not int or not 0 <= token_id < n_logits:
                raise ValueError("Added token ID is invalid or out of range")
            if not isinstance(content, str) or not content:
                raise ValueError(
                    "Added token content must be a nonempty string"
                )
            if token_id in seen_ids or content in seen_contents:
                raise ValueError("Duplicate added-token ID or content")
            seen_ids.add(token_id)
            seen_contents.add(content)
            added_defaults = {
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
            for key, val in added_defaults.items():
                if (
                    type(token.get(key)) is not bool
                    or token.get(key) is not val
                ):  # noqa: E501
                    raise ValueError(f"Unsupported added-token option {key}")
            if not isinstance(token.get("special"), bool):
                raise ValueError(
                    "added token must declare special as a boolean"
                )
        supplied = {
            (item.id, item.content) for item in self.vocab.added_tokens
        }
        declared = {(int(item["id"]), str(item["content"])) for item in tokens}
        if supplied != declared:
            raise ValueError("Added-token metadata does not match VocabTable")

    def _read_merges(self, model_merges: object) -> dict[tuple[str, str], int]:
        if not isinstance(model_merges, list):
            raise ValueError(
                "BPE model merges must be an ordered list of pairs"
            )
        expected: list[tuple[str, str]] = []
        for pair in model_merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(part, str) for part in pair)
            ):
                raise ValueError(
                    "BPE model merges must use ordered two-string lists"
                )
            expected.append((pair[0], pair[1]))
        ranks: dict[tuple[str, str], int] = {}
        with self.merges_path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream):
                line = line.rstrip("\r\n")
                if line_no == 0 and line.startswith("#version:"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2 or not all(parts):
                    raise ValueError(f"Invalid merge line {line_no + 1}")
                pair = (parts[0], parts[1])
                output = "".join(pair)
                if len(ranks) >= len(expected) or expected[len(ranks)] != pair:
                    raise ValueError(
                        "merges.txt rank order does not match tokenizer.json"
                    )
                if (
                    pair in ranks
                    or parts[0] not in self.vocab.vocab
                    or parts[1] not in self.vocab.vocab
                    or output not in self.vocab.vocab
                ):  # noqa: E501
                    raise ValueError(
                        f"Invalid or duplicate merge at line {line_no + 1}"
                    )
                ranks[pair] = len(ranks)
        if len(ranks) != len(expected):
            raise ValueError(
                "merges.txt entry count does not match tokenizer.json"
            )
        return ranks

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Input text. Added tokens are matched before NFC normalization.  # noqa: E501

        Returns:
            Token IDs in encoded order.
        """
        if not isinstance(text, str):
            raise TypeError("encode expects a string")
        result: list[int] = []
        by_content: dict[str, dict[str, object]] = {}
        for item in self._added:
            by_content[str(item["content"])] = item
        # Match literal added tokens before normalization, choosing the longest at each position.  # noqa: E501
        cursor = 0
        while cursor < len(text):
            match = max(
                (s for s in by_content if text.startswith(s, cursor)),
                key=len,
                default=None,
            )
            if match is not None:
                result.append(int(cast(int, by_content[match]["id"])))
                cursor += len(match)
                continue
            next_pos = min(
                (
                    text.find(s, cursor)
                    for s in by_content
                    if text.find(s, cursor) >= 0  # noqa: E501
                ),
                default=len(text),
            )
            result.extend(self._encode_ordinary(text[cursor:next_pos]))
            cursor = next_pos
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        pieces: list[str] = []
        last = 0
        for match in self._pattern.finditer(text):
            if match.start() > last:
                pieces.append(text[last:match.start()])
            pieces.append(match.group())
            last = match.end()
        if last < len(text):
            pieces.append(text[last:])
        ids: list[int] = []
        alphabet = self._bytes_to_unicode
        for piece in pieces:
            symbols = [alphabet[b] for b in piece.encode("utf-8")]
            for token in self._merge_symbols(symbols):
                try:
                    ids.append(self.vocab.vocab[token])
                except KeyError as exc:
                    raise ValueError(
                        f"Token absent from supplied vocabulary: {token!r}"
                    ) from exc  # noqa: E501
        return ids

    def _merge_symbols(self, symbols: list[str]) -> list[str]:
        """Apply ranked adjacent merges using a stale-safe min-heap."""
        count = len(symbols)
        if count < 2:
            return symbols
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        versions = [0] * count
        queue: list[tuple[int, int, int, int, int, str, str]] = []

        def enqueue(left: int) -> None:
            right = following[left]
            if right < 0:
                return
            left_text, right_text = symbols[left], symbols[right]
            rank = self._ranks.get((left_text, right_text))
            if rank is not None:
                heapq.heappush(
                    queue,
                    (
                        rank,
                        left,
                        right,
                        versions[left],
                        versions[right],
                        left_text,
                        right_text,
                    ),
                )

        for index in range(count - 1):
            enqueue(index)
        while queue:
            _, left, right, lv, rv, left_text, right_text = heapq.heappop(
                queue
            )
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or versions[left] != lv
                or versions[right] != rv
                or symbols[left] != left_text
                or symbols[right] != right_text
            ):
                continue
            symbols[left] += symbols[right]
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            after = following[right]
            following[left] = after
            if after >= 0:
                previous[after] = left
            before = previous[left]
            if before >= 0:
                enqueue(before)
            enqueue(left)
        return [symbols[index] for index in range(count) if alive[index]]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        errors: Literal["strict", "replace"] = "strict",
    ) -> str:  # noqa: E501
        """Decode token IDs into text.

        Args:
            ids: List of token IDs to decode.
            skip_special_tokens: Omit tokens marked special when true.
            errors: UTF-8 error handling for accumulated ordinary token bytes.

        Returns:
            Decoded text, including non-special added-token content.
        """
        if not isinstance(ids, list):
            raise TypeError("ids must be a list of integers")
        if errors not in ("strict", "replace"):
            raise ValueError("errors must be 'strict' or 'replace'")
        added = {cast(int, item["id"]): item for item in self._added}
        raw = bytearray()
        output: list[str] = []
        for token_id in ids:
            if (
                type(token_id) is not int
                or token_id < 0
                or token_id >= self.vocab.n_logits
            ):
                raise ValueError(f"Invalid token ID: {token_id!r}")
            if token_id in added:
                item = added[token_id]
                if not (skip_special_tokens and item.get("special") is True):
                    output.append(bytes(raw).decode("utf-8", errors=errors))
                    raw.clear()
                    output.append(str(item["content"]))
                continue
            if (
                token_id >= len(self.vocab.id_to_bytes)
                or self.vocab.id_to_bytes[token_id] is None
            ):  # noqa: E501
                raise ValueError(f"Unassigned token ID: {token_id}")
            raw.extend(cast(bytes, self.vocab.id_to_bytes[token_id]))
        output.append(bytes(raw).decode("utf-8", errors=errors))
        return "".join(output)
