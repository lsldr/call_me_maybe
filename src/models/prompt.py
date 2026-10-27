"""Schemas for natural-language inputs and generated function calls."""

from pydantic import (
    BaseModel, ConfigDict, JsonValue, ValidationInfo, model_validator,
)

from src.models.function_def import FunctionDef
from src.schemas.nested import validate_value


class PromptItem(BaseModel):
    """Preserve a natural-language request exactly as supplied."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str


class FunctionCallResult(BaseModel):
    """Validate an output object against the selected function's contract."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    prompt: str
    name: str
    parameters: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_function_contract(
        self, info: ValidationInfo
    ) -> "FunctionCallResult":
        """Check the function name, required arguments, and scalar types.

        Args:
            info: Validation context containing a function registry.

        Returns:
            The validated call.

        Raises:
            ValueError: If context is missing or the contract is violated.
        """
        if not info.context or "functions" not in info.context:
            raise ValueError("Validation context must include 'functions'.")

        functions: dict[str, FunctionDef] = info.context["functions"]

        definition = functions.get(self.name)
        if definition is None:
            raise ValueError(
                f"Function '{self.name}' is not defined in the context."
            )

        expected = set(definition.parameters)
        actual = set(self.parameters)

        if missing := expected - actual:
            raise ValueError(
                f"Missing parameters for '{self.name}': {sorted(missing)}"
            )

        if extra := actual - expected:
            raise ValueError(
                f"Extra parameters for '{self.name}': {sorted(extra)}"
            )

        for name, param in definition.parameters.items():
            value = self.parameters[name]

            if info.context.get("allow_nested", False):
                validate_value(value, param, info.context.get("schema_limits"))
                continue

            if param.type == "string":
                valid = isinstance(value, str)
            elif param.type == "number":
                valid = isinstance(value, (int, float)) and not isinstance(
                    value, bool
                )
            elif param.type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif param.type == "null":
                valid = value is None
            elif param.type == "boolean":
                valid = isinstance(value, bool)
            else:
                raise ValueError(f"Unknown parameter type: {param.type}")

            if not valid:
                raise ValueError(
                    f"Parameter '{name}' for function '{self.name}' "
                    f"must be of type '{param.type}', but got value: {value}"
                )

        return self
