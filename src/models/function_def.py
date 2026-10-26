"""Schemas describing the functions available to the language model."""

from pydantic import BaseModel, ConfigDict, Field


class TypeDef(BaseModel):
    """Preserve recursive schemas, including unsupported decoder types."""

    model_config = ConfigDict(extra="allow", strict=True)

    type: str = Field(
        ..., description="The type of the parameter."
    )
    items: "TypeDef | None" = None
    properties: dict[str, "TypeDef"] | None = None
    description: str | None = Field(
        default=None, description="A brief description of the parameter."
    )


class FunctionDef(BaseModel):
    """Describe a function contract independently of generated calls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., description="The name of the function.")
    description: str = Field(
        ..., description="A brief description of the function."
    )
    parameters: dict[str, TypeDef] = Field(
        ...,
        description="A dict mapping names to TypeDef instances.",
    )
    returns: TypeDef = Field(..., description="The function's return type.")
