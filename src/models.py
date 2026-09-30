"""Pydantic data models for function definitions, prompts, and results."""
from typing import Dict, Any
from pydantic import BaseModel, ConfigDict


class FunctionParameter(BaseModel):
    """Represent a single parameter schema for a function definition.

    Attributes:
        type (str): The expected data type of the parameter.
        description (str | None): Optional description of the parameter.
    """
    model_config = ConfigDict(extra="forbid")

    type: str
    description: str | None = None


class FunctionDefinition(BaseModel):
    """Represent a callable function schema and its metadata.

    Attributes:
        name (str): The identifier name of the function.
        description (str): A brief summary of what the function does.
        parameters (Dict[str, FunctionParameter]): Mapping of parameter
            names to their respective type schemas.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, Any] | None = None


class TestPrompt(BaseModel):
    """Represent an input natural language test prompt.

    Attributes:
        prompt (str): The user query to be processed by the model.
    """
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent the final structured function call output.

    Attributes:
        prompt (str): The original user prompt.
        name (str): The selected function name.
        parameters (Dict[str, Any]): The generated argument key-value pairs.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
