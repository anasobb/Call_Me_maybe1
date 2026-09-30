"""Module for loading and parsing JSON input data into Pydantic models."""
import json
from typing import List
from pydantic import ValidationError
from .models import FunctionDefinition, TestPrompt


def load_function_definitions(filepath: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing function schemas.

    Returns:
        List[FunctionDefinition]: A list of validated function objects,
        or an empty list if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(
                    f"❌ Schema Error in '{filepath}': "
                    "Root JSON structure must be a list [...]."
                )
                return []
            functions: List[FunctionDefinition] = []
            for idx, item in enumerate(data):
                try:
                    functions.append(FunctionDefinition(**item))
                except ValidationError as ve:
                    print(
                        f"❌ Invalid schema at function #{idx + 1} "
                        f"in '{filepath}':"
                    )
                    for err in ve.errors():
                        field = " -> ".join(str(loc) for loc in err["loc"])
                        msg = err["msg"]
                        print(f"   - Field '{field}': {msg}")
                    print(
                        "!!!! Tip: Ensure keys are spelled correctly: "
                        "'name', 'description', 'parameters', "
                        "and optional 'returns'."
                    )
                    return []
            return functions

    except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
        print(f"❌ Error loading function definitions: {e}")
        return []


def load_test_prompts(filepath: str) -> List[TestPrompt]:
    """Load and validate test prompts from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing test prompts.

    Returns:
        List[TestPrompt]: A list of validated test prompt objects,
        or an empty list if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(
                    f"❌ Schema Error in '{filepath}': "
                    "Root JSON structure must be a list [...]."
                )
                return []

            prompts: List[TestPrompt] = []
            for idx, item in enumerate(data):
                try:
                    prompts.append(TestPrompt(**item))
                except ValidationError as ve:
                    print(
                        f"❌ Invalid schema at prompt #{idx + 1} "
                        f"in '{filepath}':"
                    )
                    for err in ve.errors():
                        field = " -> ".join(str(loc) for loc in err["loc"])
                        msg = err["msg"]
                        print(f"   - Field '{field}': {msg}")
                    print(
                        "!!!!!! Tip: Each item must contain only the "
                        "'prompt' key with a string value."
                    )
                    return []
            return prompts

    except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
        print(f"❌ Error loading test prompts: {e}")
        return []
