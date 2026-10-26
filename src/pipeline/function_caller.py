"""Coordinate model-only selection, scalar extraction, and batch validation."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import RunSettings
from src.decoding.decoder import Decoder, Logits
from src.decoding.generators import (
    Scalar, generate_boolean, generate_name, generate_number, generate_string,
)
from src.decoding.indexed import generate_indexed_name
from src.io.reader import validate_function_calls
from src.models.function_def import FunctionDef
from src.models.prompt import FunctionCallResult, PromptItem
from src.prompting.sanitize import safe_json
from src.prompting.indexed import indexed_context
from src.prompting.templates import (
    ContextStyle, arguments_context, selection_context,
    string_argument_context,
)


class FunctionCaller(BaseModel):
    """Keep one decoder and public model callables for a complete run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    functions: dict[str, FunctionDef]
    decoder: Decoder
    encode: Callable[[str], list[int]]
    logits: Logits
    markers: list[str] = Field(default_factory=list)
    style: ContextStyle = "native"
    context_observer: Callable[[str, str], None] | None = None
    finish_unique_names: bool = True
    score_string_opening: bool = False
    allow_number_suffix: bool = False

    def encode_context(self, phase: str, text: str) -> list[int]:
        """Record a completed context prefix, then encode it as a whole."""
        if self.context_observer is not None:
            self.context_observer(phase, text)
        return self.encode(text)

    def parameter_prefix(self, key: str, kind: str) -> str:
        """Return known syntax with an optional prefilled opening quote."""
        return safe_json(key, self.markers) + ":" + (
            ' "' if kind == "string" and not self.score_string_opening
            else ""
        )

    def call(self, prompt: PromptItem) -> FunctionCallResult:
        """Select a function, then extract its arguments in definition order.

        Args:
            prompt: Original request; retained verbatim in the result.

        Returns:
            A validated call with exactly the declared scalar arguments.
        """
        definitions = list(self.functions.values())
        if self.style == "indexed" and len(definitions) <= 26:
            text, labels = indexed_context(
                definitions, prompt.prompt, self.markers,
            )
            generated = generate_indexed_name(
                self.decoder, self.encode_context("name", text),
                self.logits, labels,
            )
        else:
            text = selection_context(
                definitions, prompt.prompt, self.markers, style=self.style,
            )
            generated = generate_name(
                self.decoder, self.encode_context("name", text), self.logits,
                list(self.functions),
                finish_unique=self.finish_unique_names,
            )
        name = generated.value
        if not isinstance(name, str) or name not in self.functions:
            raise ValueError("Name generation did not select a function.")
        definition = self.functions[name]
        text = arguments_context(
            definition, prompt.prompt, self.markers, style=self.style,
        )
        parameters: dict[str, Scalar] = {}
        entries = list(definition.parameters.items())
        pending = b""
        for index, (key, schema) in enumerate(entries):
            delimiter: Literal[",", "}"] = (
                "," if index + 1 < len(entries) else "}"
            )
            prefix = self.parameter_prefix(key, schema.type)
            raw_prefix = prefix.encode("utf-8")
            if not raw_prefix.startswith(pending):
                raise ValueError("Consumed syntax does not match next key.")
            # Pending bytes can end inside a Unicode key. Complete the known
            # prefix before decoding, preserving every selected suffix byte.
            text += (pending + raw_prefix[len(pending):]).decode("utf-8")
            pending = b""
            separator = delimiter + (
                " " if self.style in {
                    "boundary", "indexed", "guided", "examples",
                }
                and delimiter == "," else ""
            )
            if schema.type == "null":
                parameters[key] = None
                text += " null" + separator
                continue
            if schema.type == "string":
                following = None
                if self.style in {
                    "boundary", "indexed", "guided", "examples",
                }:
                    if index + 1 < len(entries):
                        next_key, next_schema = entries[index + 1]
                        tail = separator + self.parameter_prefix(
                            next_key, next_schema.type,
                        )
                    else:
                        tail = "}}\n"
                    following = tail.encode("utf-8")
                value_context = (
                    string_argument_context(
                        definition, prompt.prompt, key,
                        dict(parameters), self.markers,
                    ) if self.style == "focused" else text
                )
                if self.style == "focused" and self.score_string_opening:
                    value_context = value_context[:-1]
                value = generate_string(
                    self.decoder,
                    self.encode_context(f"parameter:{key}", value_context),
                    self.logits, following=following,
                    include_opening=self.score_string_opening,
                )
            elif schema.type in {"number", "integer"}:
                value = generate_number(
                    self.decoder,
                    self.encode_context(f"parameter:{key}", text), self.logits,
                    integer=schema.type == "integer", delimiter=delimiter,
                    following=(
                        b"}\n" if self.allow_number_suffix
                        and delimiter == "}" else None
                    ),
                )
            elif schema.type == "boolean":
                value = generate_boolean(
                    self.decoder,
                    self.encode_context(f"parameter:{key}", text), self.logits,
                )
            else:
                raise ValueError(f"Unsupported parameter type: {schema.type}")
            parameters[key] = value.value
            # Rebuild at completed-value boundaries only. Reserialize string
            # values to prevent generated marker text becoming control IDs.
            if schema.type == "string":
                if self.score_string_opening:
                    if value.fragment.startswith(b" "):
                        text += " "
                else:
                    text = text[:-1]
                text += safe_json(value.value, self.markers)
                suffix = value.consumed_suffix
                if len(suffix) > len(separator):
                    if delimiter == ",":
                        text += separator
                        pending = suffix[len(separator):]
                    else:
                        text += suffix.decode("utf-8")
                else:
                    text += suffix.decode("utf-8") + separator[len(suffix):]
            else:
                text += value.fragment.decode("utf-8")
                if not value.consumed_delimiter:
                    text += separator
                elif delimiter == "," and self.style in {
                    "boundary", "indexed", "guided", "examples",
                }:
                    text += " "
        if not entries:
            text += "}"
        return FunctionCallResult.model_validate(
            {"prompt": prompt.prompt, "name": name, "parameters": parameters},
            context={"functions": self.functions},
        )

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Validate the entire ordered batch, failing with the prompt index."""
        results: list[FunctionCallResult] = []
        for index, prompt in enumerate(prompts):
            try:
                results.append(self.call(prompt))
            except Exception as error:
                raise ValueError(f"Prompt {index}: {error}") from error
        return validate_function_calls(results, self.functions, prompts)


def create_caller(
    functions: dict[str, FunctionDef], settings: RunSettings,
) -> FunctionCaller:
    """Load one SDK model and vocabulary, with no duplicated asset files."""
    from src.llm.model_wrapper import ModelWrapper
    from src.vocab.vocab_table import VocabTable

    if not functions:
        raise ValueError("No available functions for the input prompts.")
    model = (
        ModelWrapper(model_name=settings.model_name, device=settings.device)
        if settings.device is not None
        else ModelWrapper(model_name=settings.model_name)
    )
    scores = model.logits(model.encode("Warmup"))
    vocab = VocabTable.from_files(
        model.get_vocab_file_path(), model.get_tokenizer_file_path(),
        len(scores),
    )
    return FunctionCaller(
        functions=functions,
        decoder=Decoder(
            vocab=vocab, max_new_tokens=settings.max_new_tokens,
            max_value_bytes=settings.max_value_bytes,
            max_context_tokens=settings.max_context_tokens,
        ),
        encode=model.encode, logits=model.logits,
        markers=[token.content for token in vocab.added_tokens],
        style="examples",
        score_string_opening=True,
        allow_number_suffix=True,
    )
