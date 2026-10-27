from typing import Literal

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.nested import SchemaLimits
from src.llm.profiles import get_profile


class RunSettings(BaseModel):
    """Validate paths, inference device, and bounded generation resources."""

    model_config = ConfigDict(extra="forbid")

    functions_path: Path = Path("data/input/functions_definition.json")
    input_path: Path = Path("data/input/function_calling_tests.json")
    output_path: Path = Path("data/output/function_calls.json")

    model_name: str = "Qwen/Qwen3-0.6B"
    device: Literal["cpu", "cuda", "mps"] | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    max_value_bytes: int = Field(default=4096, ge=1)
    max_context_tokens: int = Field(default=1536, ge=1)
    profile: Literal["mandatory", "bonus"] = "mandatory"
    tokenizer: Literal["sdk", "local"] = "sdk"
    schema_limits: SchemaLimits = Field(default_factory=SchemaLimits)
    recovery: Literal["none", "retry"] = "none"
    model_snapshot: Path | None = None
    diagnostics_path: Path | None = None
    optimize: bool = True
    performance_path: Path | None = None
    trace_path: Path | None = None
    visual: bool = False

    @model_validator(mode="after")
    def validate_features(self) -> "RunSettings":
        """Reject unsupported bonus switches before loading any model.

        Returns:
            Validated settings with explicit feature boundaries.
        """
        if self.tokenizer == "local" and self.profile != "bonus":
            raise ValueError("Local tokenization requires --profile bonus.")
        model = get_profile(self.model_name)
        if self.profile != "bonus" and (
            self.recovery != "none" or self.model_snapshot is not None
            or self.model_name != "Qwen/Qwen3-0.6B"
        ):
            raise ValueError("Recovery and alternate models require bonus.")
        if self.tokenizer not in model.supported_tokenizers:
            raise ValueError(
                f"Tokenizer {self.tokenizer!r} is unsupported for "
                f"{self.model_name}."
            )
        if self.max_context_tokens > model.max_context_tokens:
            raise ValueError("Context limit exceeds the model profile limit.")
        artifacts = [path for path in (
            self.diagnostics_path, self.performance_path, self.trace_path,
        ) if path is not None]
        protected = [self.output_path, self.functions_path, self.input_path]
        resolved = [path.resolve() for path in artifacts]
        if len(set(resolved)) != len(resolved) or any(
            path in [item.resolve() for item in protected] for path in resolved
        ):
            raise ValueError("Diagnostics must use separate output files.")
        return self
