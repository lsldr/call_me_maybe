"""Coordinate model-only selection, scalar extraction, and batch validation."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.config import RunSettings
from src.decoding.decoder import Decoder, Logits
from src.decoding.failures import (
    DecodeFailure, FailureReason, FailureRecord, can_retry,
)
from src.decoding.generators import (
    GeneratedValue,
    generate_boolean, generate_container, generate_name, generate_number,
    generate_string,
)
from src.io.reader import validate_function_calls
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import FunctionCallResult, PromptItem
from src.prompting.sanitize import safe_json
from src.prompting.recovery import inject_instruction
from src.prompting.templates import (
    BOUNDARY_STYLES, ContextStyle, arguments_context, selection_context,
)
from src.schemas.nested import SchemaLimits
from src.llm.profiles import ContextFamily, cached_snapshot, get_profile
from src.llm.profiles import validate_snapshot
from src.prompting import smollm
from src.diagnostics.performance import PerformanceStats, measure


class FunctionCaller(BaseModel):
    """Keep one decoder and public model callables for a complete run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid",
                              validate_assignment=True)
    functions: dict[str, FunctionDef]
    decoder: Decoder
    encode: Callable[[str], list[int]]
    logits: Logits
    markers: list[str] = Field(default_factory=list)
    style: ContextStyle = "native"
    context_observer: Callable[[str, str], None] | None = None
    selection_observer: Callable[[str], None] | None = None
    finish_unique_names: bool = True
    score_string_opening: bool = False
    allow_number_suffix: bool = False
    allow_nested: bool = False
    schema_limits: SchemaLimits = Field(default_factory=SchemaLimits)
    recovery: Literal["none", "retry"] = "none"
    context_family: ContextFamily = "qwen3"
    failure_records: list[FailureRecord] = Field(default_factory=list)
    failure_observer: Callable[[FailureRecord], None] | None = None
    performance: PerformanceStats | None = None
    attempt_observer: Callable[[str, dict[str, JsonValue]], None] | None = None

    def encode_context(self, phase: str, text: str) -> list[int]:
        """Record a completed context prefix, then encode it as a whole."""
        if self.context_observer is not None:
            self.context_observer(phase, text)
        with measure(self.performance, "encode"):
            return self.encode(text)

    def _observe_generation(
        self, generate: Callable[[], GeneratedValue], *,
        prompt_index: int | None, phase: Literal["selection", "parameter"],
        path: tuple[str | int, ...] = (), attempt: int = 1,
        retry_available: bool = False,
    ) -> GeneratedValue:
        """Capture attempt boundaries without changing generation decisions."""
        observer = self.attempt_observer
        if self.performance is not None:
            self.performance.attempt = attempt
        if observer is not None:
            observer("attempt_start", {
                "prompt_index": prompt_index, "phase": phase,
                "path": list(path), "attempt": attempt,
            })
        try:
            result = generate()
        except BaseException as error:
            if observer is not None:
                observer("attempt_end", {
                    "status": "failed", "message": str(error),
                    "reason": error.reason if isinstance(error, DecodeFailure)
                    else None,
                    "usage": error.usage.model_dump(mode="json")
                    if isinstance(error, DecodeFailure) else None,
                    "retry_scheduled": retry_available
                    and isinstance(error, DecodeFailure) and can_retry(error),
                })
            raise
        if observer is not None:
            observer("attempt_end", {
                "status": "completed", "value": result.value,
                "forced_suffix_hex": result.forced_suffix.hex(),
            })
        return result

    def parameter_prefix(self, key: str, kind: str) -> str:
        """Return known syntax with an optional prefilled opening quote."""
        return safe_json(key, self.markers) + ":" + (
            ' "' if kind == "string" and not self.score_string_opening
            else ""
        )

    def _arguments_context(self, definition: FunctionDef, request: str) -> str:
        """Build the initial arguments context for the active profile."""
        if self.context_family == "smollm2":
            return smollm.arguments_context(definition, request, self.markers)
        return arguments_context(
            definition, request, self.markers, style=self.style,
        )

    def _selection_context(
        self, definitions: list[FunctionDef], request: str,
    ) -> str:
        """Build the initial selection context for the active profile."""
        if self.context_family == "smollm2":
            return smollm.selection_context(definitions, request, self.markers)
        return selection_context(
            definitions, request, self.markers, style=self.style,
        )

    def _record_failure(
        self, error: DecodeFailure, *, prompt_index: int | None,
        phase: Literal["selection", "parameter"],
        argument_path: tuple[str | int, ...], attempt: int,
        retry_scheduled: bool,
    ) -> None:
        """Append a typed failure record and notify an optional observer."""
        record = FailureRecord(
            prompt_index=prompt_index, phase=phase,
            argument_path=argument_path, attempt=attempt,
            reason=error.reason, message=str(error), usage=error.usage,
            retry_scheduled=retry_scheduled,
        )
        self.failure_records.append(record)
        if self.failure_observer is not None:
            self.failure_observer(record)

    def _generate_parameter(
        self, *, key: str, schema: TypeDef,
        index: int, entries: list[tuple[str, TypeDef]], text: str,
        separator: str, delimiter: Literal[",", "}"],
        retry_reason: FailureReason | None = None,
    ) -> GeneratedValue:
        """Generate one declared argument with the shared constrained loop."""
        phase = f"parameter:{key}"
        if retry_reason is not None:
            phase += ":attempt:2"
            text = inject_instruction(text, retry_reason)
        ids = self.encode_context(phase, text)
        if index + 1 < len(entries):
            next_key, next_schema = entries[index + 1]
            tail = separator + self.parameter_prefix(
                next_key, next_schema.type,
            )
        else:
            tail = "}}\n"
        if schema.type == "string":
            return generate_string(
                self.decoder, ids, self.logits,
                following=tail.encode("utf-8")
                if self.style in BOUNDARY_STYLES else None,
                include_opening=self.score_string_opening,
            )
        if schema.type in {"array", "object"} and self.allow_nested:
            return generate_container(
                self.decoder, ids, self.logits, schema,
                limits=self.schema_limits, following=tail.encode("utf-8"),
                markers=tuple(self.markers),
            )
        if schema.type in {"number", "integer"}:
            return generate_number(
                self.decoder, ids, self.logits,
                integer=schema.type == "integer", delimiter=delimiter,
                following=b"}\n" if self.allow_number_suffix
                and delimiter == "}" else None,
            )
        if schema.type == "boolean":
            return generate_boolean(self.decoder, ids, self.logits)
        raise ValueError(f"Unsupported parameter type: {schema.type}")

    def call(
        self, prompt: PromptItem, *, prompt_index: int | None = None,
    ) -> FunctionCallResult:
        """Select a function, then extract its arguments in definition order.

        Args:
            prompt: Original request; retained verbatim in the result.

        Returns:
            A validated call with exactly the declared scalar arguments.
        """
        definitions = list(self.functions.values())
        try:
            text = self._selection_context(definitions, prompt.prompt)
            generated = self._observe_generation(
                lambda: generate_name(
                    self.decoder, self.encode_context("name", text),
                    self.logits, list(self.functions),
                    finish_unique=self.finish_unique_names,
                ), prompt_index=prompt_index, phase="selection",
            )
        except DecodeFailure as error:
            self._record_failure(
                error, prompt_index=prompt_index, phase="selection",
                argument_path=(), attempt=1, retry_scheduled=False,
            )
            raise
        name = generated.value
        if not isinstance(name, str) or name not in self.functions:
            raise ValueError("Name generation did not select a function.")
        if self.selection_observer is not None:
            self.selection_observer(name)
        definition = self.functions[name]
        text = self._arguments_context(definition, prompt.prompt)
        parameters: dict[str, JsonValue] = {}
        entries = list(definition.parameters.items())
        pending = b""
        retry_used = False
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
                " " if self.style in BOUNDARY_STYLES
                and delimiter == "," else ""
            )
            if schema.type == "null":
                parameters[key] = None
                text += " null" + separator
                continue
            try:
                value = self._observe_generation(
                    lambda: self._generate_parameter(
                        key=key, schema=schema,
                        index=index, entries=entries, text=text,
                        separator=separator, delimiter=delimiter,
                    ), prompt_index=prompt_index, phase="parameter",
                    path=("parameters", key), retry_available=(
                        self.recovery == "retry" and not retry_used
                    ),
                )
            except DecodeFailure as error:
                schedule_retry = (
                    self.recovery == "retry" and not retry_used
                    and can_retry(error)
                )
                self._record_failure(
                    error, prompt_index=prompt_index, phase="parameter",
                    argument_path=("parameters", key), attempt=1,
                    retry_scheduled=schedule_retry,
                )
                if not schedule_retry:
                    raise
                retry_used = True
                text = self._arguments_context(definition, prompt.prompt)
                for previous_key, _previous_schema in entries[:index]:
                    previous_delimiter = ","
                    previous_separator = previous_delimiter + (
                        " " if self.style in BOUNDARY_STYLES else ""
                    )
                    text += (
                        safe_json(previous_key, self.markers) + ":"
                        + safe_json(parameters[previous_key], self.markers)
                        + previous_separator
                    )
                text += self.parameter_prefix(key, schema.type)
                pending = b""
                retry_reason = error.reason
                try:
                    value = self._observe_generation(
                        lambda: self._generate_parameter(
                            key=key, schema=schema,
                            index=index, entries=entries, text=text,
                            separator=separator, delimiter=delimiter,
                            retry_reason=retry_reason,
                        ), prompt_index=prompt_index, phase="parameter",
                        path=("parameters", key), attempt=2,
                    )
                except DecodeFailure as retry_error:
                    self._record_failure(
                        retry_error, prompt_index=prompt_index,
                        phase="parameter", argument_path=("parameters", key),
                        attempt=2, retry_scheduled=False,
                    )
                    raise
            parameters[key] = value.value
            # Rebuild at completed-value boundaries only. Reserialize string
            # values to prevent generated marker text becoming control IDs.
            if schema.type in {"string", "array", "object"}:
                if schema.type == "string" and self.score_string_opening:
                    if value.fragment.startswith(b" "):
                        text += " "
                elif schema.type == "string":
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
                elif delimiter == "," and self.style in BOUNDARY_STYLES:
                    text += " "
        if not entries:
            text += "}"
        return FunctionCallResult.model_validate(
            {"prompt": prompt.prompt, "name": name, "parameters": parameters},
            context={
                "functions": self.functions, "allow_nested": self.allow_nested,
                "schema_limits": self.schema_limits,
            },
        )

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Validate the entire ordered batch, failing with the prompt index."""
        results: list[FunctionCallResult] = []
        for index, prompt in enumerate(prompts):
            try:
                results.append(self.call(prompt, prompt_index=index))
            except Exception as error:
                raise ValueError(f"Prompt {index}: {error}") from error
        return validate_function_calls(
            results, self.functions, prompts, allow_nested=self.allow_nested,
            schema_limits=self.schema_limits,
        )


def create_caller(
    functions: dict[str, FunctionDef], settings: RunSettings,
) -> FunctionCaller:
    """Initialize one model, optionally recording costs outside generation."""
    stats = PerformanceStats(workload="initialization") if (
        settings.performance_path is not None
    ) else None
    with measure(stats, "total"):
        caller = _create_caller(functions, settings, stats)
    if stats is not None:
        stats.workload = "generation"
    return caller


def _create_caller(
    functions: dict[str, FunctionDef], settings: RunSettings,
    stats: PerformanceStats | None,
) -> FunctionCaller:
    """Load one SDK model and vocabulary, with no duplicated asset files."""
    from src.llm.model_wrapper import ModelWrapper
    from src.vocab.vocab_table import VocabTable

    if not functions:
        raise ValueError("No available functions for the input prompts.")
    profile = get_profile(settings.model_name)
    if settings.profile == "bonus":
        snapshot = validate_snapshot(
            profile, settings.model_snapshot or cached_snapshot(profile),
        )
        model = ModelWrapper(
            model_name=settings.model_name, snapshot_path=snapshot,
            device=settings.device,
        ) if settings.device is not None else ModelWrapper(
            model_name=settings.model_name, snapshot_path=snapshot,
        )
    else:
        model = ModelWrapper(
            model_name=settings.model_name, device=settings.device,
        ) if settings.device is not None else ModelWrapper(
            model_name=settings.model_name,
        )
    # Local mode must not reach the SDK encoder, even during warmup.
    with measure(stats, "warmup"):
        scores = model.logits(
            [0] if settings.tokenizer == "local" else model.encode("Warmup")
        )
    vocab = VocabTable.from_files(
        model.get_vocab_file_path(), model.get_tokenizer_file_path(),
        len(scores),
    )
    if settings.profile == "bonus":
        if vocab.n_logits != profile.expected_n_logits:
            raise ValueError("Logits dimension differs from model profile.")
        added = {token.content: token.id for token in vocab.added_tokens}
        for marker, token_id in profile.required_tokens:
            if added.get(marker) != token_id:
                raise ValueError(
                    f"Model marker differs from profile: {marker}"
                )
    encode = model.encode
    if settings.tokenizer == "local":
        from src.tokenization.local import ByteBPETokenizer

        tokenizer = ByteBPETokenizer(
            vocab=vocab,
            tokenizer_path=Path(model.get_tokenizer_file_path()),
            merges_path=Path(model.get_merges_file_path()),
        )
        encode = tokenizer.encode
    return FunctionCaller(
        functions=functions,
        decoder=Decoder(
            vocab=vocab, max_new_tokens=settings.max_new_tokens,
            max_value_bytes=settings.max_value_bytes,
            max_context_tokens=settings.max_context_tokens,
            optimize=settings.optimize, performance=stats,
        ),
        encode=encode, logits=model.logits,
        markers=[token.content for token in vocab.added_tokens],
        style="examples",
        score_string_opening=True,
        allow_number_suffix=True,
        allow_nested=settings.profile == "bonus",
        schema_limits=settings.schema_limits,
        recovery=settings.recovery,
        context_family=profile.context_family,
        performance=stats,
    )
