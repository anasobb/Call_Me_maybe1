"""Main entry point for the Call Me Maybe LLM function-calling CLI."""

import os
import json
import argparse
from typing import Any
from llm_sdk import Small_LLM_Model
from .ingestion import load_function_definitions, load_test_prompts
from .generator import JSONGenerator


def handle_unsupported_prompt(
        prompt: str, reason: str
        ) -> dict[str, Any]:
    """Handle invalid prompts or queries with no matching function.

    Args:
        prompt (str): The original user prompt.
        reason (str): Explanation of why the prompt was refused or unmatched.

    Returns:
        dict[str, Any]: A structured fallback response for the output JSON.
    """
    print(f"⚠️ Notice: {reason}")
    return {
        "prompt": prompt,
        "name": "no_matching_function",
        "parameters": {},
        "error": reason
    }


def main() -> None:
    """Parse CLI arguments, run constrained generation, and save outputs.

    Loads function definitions and test prompts, initializes the language
    model and constrained JSON generator, processes each prompt, and writes
    the resulting function calls to the specified output JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - LLM Function Calling"
        )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions JSON"
        )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to prompts JSON"
        )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to output JSON"
        )
    args = parser.parse_args()

    print("--- Loading Data ---")
    functions = load_function_definitions(args.functions_definition)
    prompts = load_test_prompts(args.input)

    if not functions or not prompts:
        print("❌ Error: Missing or invalid input data. Exiting gracefully.")
        return

    print("--- Loading Model (This might take a few seconds) ---")
    try:
        model = Small_LLM_Model()
        generator = JSONGenerator(model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    results_list: list[dict[str, Any]] = []

    for idx, test_prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Processing", end="")
        print(f"Prompt: '{test_prompt.prompt}'")
        try:
            if not test_prompt.prompt or not test_prompt.prompt.strip():
                raise ValueError(
                    "Invalid prompt: Prompt is empty or contains only spaces."
                )

            raw_result = generator.generate(
                test_prompt.prompt, functions, max_tokens=60
                )

            if raw_result == "NO_MATCH":
                results_list.append(
                    handle_unsupported_prompt(
                        test_prompt.prompt,
                        "No available function matches this prompt."
                    )
                )
                continue

            json_start_index = raw_result.find('{"prompt":')
            if json_start_index != -1:
                json_str = raw_result[json_start_index:]
                parsed_json = json.loads(json_str)
                results_list.append(parsed_json)
                print("✅ Successfully generated and parsed.")
            else:
                results_list.append(
                    handle_unsupported_prompt(
                        test_prompt.prompt,
                        "No suitable function could process this prompt."
                    )
                )

        except ValueError as ve:
            results_list.append(
                handle_unsupported_prompt(test_prompt.prompt, str(ve))
            )
        except json.JSONDecodeError:
            results_list.append(
                handle_unsupported_prompt(
                    test_prompt.prompt,
                    "Prompt could not be mapped to valid function parameters."
                )
            )
        except Exception as e:
            print(f"❌ Unexpected error during generation: {e}")
            results_list.append({"prompt": test_prompt.prompt,
                                 "error": f"Exception: {e}"})

    print(f"\n--- Saving Results to {args.output} ---")
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as out_file:
            json.dump(results_list, out_file, indent=4)
        print("✅ All results saved successfully.")
    except Exception as e:
        print(f"❌ Error saving output file: {e}")


if __name__ == "__main__":
    main()
