"""Constrained JSON decoding and function-calling generator module."""

import re
import json
import numpy as np
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


class JSONGenerator:
    """Generate schema-compliant JSON function calls using logit masking.

    Attributes:
        model (Small_LLM_Model): The underlying language model instance.
        vocab (list[str]): Pre-decoded vocabulary cache for fast masking.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """Initialize the generator and build the vocabulary token cache.

        Args:
            model (Small_LLM_Model): The initialized LLM instance.
        """
        self.model = model
        self.vocab: list[str] = []

        print("----- Building Vocabulary Cache (Optimizing Memory) -----")

        vocab_path = self.model.get_path_to_vocab_file()

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        vocab_size = len(vocab_dict)

        for t_id in range(vocab_size):
            token_str = str(self.model.decode([t_id]))
            self.vocab.append(token_str)

        print(f"----- Cache Built: {vocab_size} tokens loaded into ", end="")
        print("memory -----\n")

    def _is_prompt_supported(
            self, user_prompt: str, func_def: FunctionDefinition
            ) -> bool:
        """Check if the user prompt is relevant to the selected function.

        Verifies that numeric functions have sufficient numbers in the
        prompt and that the prompt shares domain keywords with the schema.

        Args:
            user_prompt (str): The natural language query from the user.
            func_def (FunctionDefinition): The candidate function schema.

        Returns:
            bool: True if the prompt can be handled by the function.
        """
        stopwords = {
            "the", "and", "for", "with", "from", "into", "all",
            "two", "given", "that", "this", "what", "are", "you",
            "your", "can", "how", "who", "where", "when", "why"
        }
        synonyms = {
            "add": {"sum", "plus", "total"},
            "multiply": {"product", "times"},
            "execute": {"run", "sql", "query"},
            "replace": {"substitute"},
            "substitute": {"replace"},
        }

        num_params = [
            p for p in func_def.parameters.values()
            if p.type in ["number", "integer"]
        ]
        if num_params:
            found_numbers = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
            if len(found_numbers) < len(num_params):
                return False

        raw_words = re.findall(
            r"[a-z]+",
            f"{func_def.name} {func_def.description}".lower()
        )
        keywords = {
            w for w in raw_words
            if len(w) >= 3 and w != "fn" and w not in stopwords
        }
        for word in list(keywords):
            if word in synonyms:
                keywords.update(synonyms[word])

        prompt_words = set(re.findall(r"[a-z]+", user_prompt.lower()))
        if not (keywords & prompt_words):
            return False

        return True

    def _select_function(
            self, user_prompt: str,
            available_functions: list[FunctionDefinition]
            ) -> tuple[FunctionDefinition, list[int], str]:
        """Select the most appropriate function for the user prompt.

        Constrains the model's token generation to only valid function
        names defined in the provided schema list.

        Args:
            user_prompt (str): The natural language query from the user.
            available_functions (list[FunctionDefinition]): List of
                available function definitions.

        Returns:
            tuple[FunctionDefinition, list[int], str]: The selected
            function definition, updated token IDs, and current text.
        """
        print("\n----- Selecting Function Dynamically -----")
        valid_names = [f.name for f in available_functions]

        # Build the context prompting the LLM to choose a function
        context = "Available functions:\n"
        for f in available_functions:
            params = ", ".join(f.parameters.keys())
            context += f"- {f.name}({params}): {f.description}\n"
        safe_prompt = json.dumps(user_prompt)[1:-1]
        context += (
            f"\nQuestion: {user_prompt}\n"
            f"Action: {{\"prompt\": \"{safe_prompt}\", \"name\": \""
        )
        print(context)
        current_ids = self.model.encode(context).tolist()[0]
        generated_name = ""

        for step in range(15):
            logits = np.array(
                self.model.get_logits_from_input_ids(current_ids)
                )
            masked_logits = np.full_like(logits, -np.inf)

            possible_found = False
            for t_id, token_str in enumerate(self.vocab):
                potential_name = generated_name + token_str

                # Allow tokens that form valid function names
                for v_name in valid_names:
                    if (
                        v_name.startswith(potential_name) or
                        potential_name.startswith(v_name)
                    ):
                        masked_logits[t_id] = logits[t_id]
                        possible_found = True
                        break

            if not possible_found:
                break

            next_token_id = int(np.argmax(masked_logits))
            next_token_str = self.vocab[next_token_id]

            generated_name += next_token_str
            current_ids.append(next_token_id)
            print(f"Name Step {step + 1} | Appended: '{next_token_str}'")

            # Check if a full valid function name has been generated
            for f in available_functions:
                if generated_name == f.name:
                    print(f"✅ LLM Selected: {f.name}")
                    return f, current_ids, context + generated_name

        return available_functions[0], current_ids, context + generated_name

    def _apply_mask(
            self, logits: np.ndarray,
            current_text: str, func_def: FunctionDefinition
            ) -> np.ndarray:
        """Apply grammar and schema constraints to raw token logits.

        Masks invalid tokens with negative infinity based on the current
        JSON parsing state, expected parameter keys, and data types.

        Args:
            logits (np.ndarray): Raw logit scores from the language model.
            current_text (str): Decoded text generated so far.
            func_def (FunctionDefinition): Target function schema.

        Returns:
            np.ndarray: Modified logit array with invalid tokens masked.
        """
        masked_logits = np.copy(logits)

        params_start_idx = current_text.find('"parameters": {')
        if params_start_idx == -1:
            return masked_logits

        text_inside_params = current_text[params_start_idx + 15:]
        required_params = list(func_def.parameters.keys())

        # Count commas and braces ONLY outside of string quotes
        in_quotes = False
        comma_count = 0
        last_comma_idx = -1
        has_closing_brace = False

        for i, ch in enumerate(text_inside_params):
            if ch == '"' and (i == 0 or text_inside_params[i - 1] != '\\'):
                in_quotes = not in_quotes
            elif not in_quotes:
                if ch == ',':
                    comma_count += 1
                    last_comma_idx = i
                elif ch == '}':
                    has_closing_brace = True

        # to determine the number of parameters
        current_param_index = comma_count

        # the model complete generate
        if has_closing_brace:
            for t_id, token_str in enumerate(self.vocab):
                if (
                    ',' in token_str or
                    '"' in token_str or
                    any(c.isalpha() for c in token_str)
                ):
                    masked_logits[t_id] = -np.inf
            return masked_logits

        # to forbid the model create new parameters from here head
        if current_param_index >= len(required_params):
            for t_id, token_str in enumerate(self.vocab):
                if '}' not in token_str and token_str.strip() != "":
                    masked_logits[t_id] = -np.inf
            return masked_logits

        # generate the name of next parameters
        next_param = required_params[current_param_index]
        is_last_param = (current_param_index == len(required_params) - 1)

        # if there is a comma outside quotes, take just the text after it
        if last_comma_idx != -1:
            current_segment = text_inside_params[last_comma_idx + 1:]
        else:
            current_segment = text_inside_params
        # thats men the model still in level generate the parameter name
        if ':' not in current_segment:
            masked_logits.fill(-np.inf)
            expected_key = f'"{next_param}":'
            current_clean = current_segment.strip()

            for t_id, token_str in enumerate(self.vocab):
                potential = (current_segment + token_str).strip()

                if (
                    expected_key.startswith(potential) or
                    potential.startswith(expected_key)
                ):
                    # this line prevent the model to generate
                    # infinite loop of spaces likes
                    if potential == current_clean and token_str.strip() == "":
                        continue
                    # we take the right generted char from the model
                    masked_logits[t_id] = logits[t_id]
        # if we find the (:) and now we will generate the value of parameters
        else:
            # Get the actual value string being generated
            # right now (after the colon)
            raw_after_colon = current_segment[current_segment.find(':') + 1:]
            value_segment = raw_after_colon.strip()

            # Prevent infinite space loops right after the colon
            if len(value_segment) == 0 and len(raw_after_colon) > 0:
                for t_id, token_str in enumerate(self.vocab):
                    if token_str.strip() == "":
                        masked_logits[t_id] = -np.inf

            # we need to determine the type of parameter
            param_type = func_def.parameters[next_param].type

            # if parameter is number we will prevent the model to generate text
            if param_type in ["number", "integer"]:
                has_decimal = (("." in value_segment) and
                               not value_segment.endswith("."))
                for t_id, token_str in enumerate(self.vocab):
                    if (
                        any(c.isalpha() for c in token_str) or
                        '"' in token_str or '{' in token_str or
                        '[' in token_str
                    ):
                        masked_logits[t_id] = -np.inf

                    # Prevent spaces or newlines once digits have
                    # started (unless closing with } or ,)
                    if len(value_segment) > 0:
                        if ((' ' in token_str or '\n' in token_str)
                                and (',' not in token_str and
                                     '}' not in token_str)):
                            masked_logits[t_id] = -np.inf

                    # if we arrived to the last parameters we do not
                    # need to another (,) after it
                    if is_last_param and ',' in token_str:
                        masked_logits[t_id] = -np.inf
                    # If it's a "number" (Float) and doesn't
                    # have a decimal yet, DO NOT allow it to close
                    if param_type == "number" and not has_decimal:
                        if ',' in token_str or '}' in token_str:
                            masked_logits[t_id] = -np.inf

            elif param_type == "string":
                # Check if the string is closed
                # (ignoring escaped quotes like \")
                is_string_closed = (
                    value_segment.startswith('"') and
                    value_segment.endswith('"') and
                    not value_segment.endswith('\\"') and
                    len(value_segment) > 1
                )

                # Extract original prompt to check
                # how many quotes are actually needed
                prompt_start = current_text.find('Question: ') + 10
                prompt_end = current_text.find('\nAction: {"prompt":')
                orig_prompt = current_text[prompt_start:prompt_end]

                for t_id, token_str in enumerate(self.vocab):
                    # Allow (\) ONLY if it's a
                    # valid JSON escape like (\\) or (\")
                    if '\\' in token_str:
                        if '\\\\' not in token_str and '\\"' not in token_str:
                            masked_logits[t_id] = -np.inf
                        if '\\"' in token_str:
                            if (value_segment.count('\\"')
                                    >= orig_prompt.count('"')):
                                masked_logits[t_id] = -np.inf

                    # Prevent single quotes right
                    # after the opening double quote
                    if len(value_segment) == 1 and value_segment == '"':
                        if "'" in token_str:
                            masked_logits[t_id] = -np.inf

                    # close the string and add a comma
                    # at the same time (like '!",')
                    if not is_string_closed and is_last_param:
                        if '"' in token_str and ',' in token_str:
                            if token_str.rfind(',') > token_str.rfind('"'):
                                masked_logits[t_id] = -np.inf

                    if is_string_closed:
                        # the model complete generation
                        # we avoid any another (")
                        if '"' in token_str:
                            masked_logits[t_id] = -np.inf

                        if (
                            any(c.isalpha() for c in token_str) or
                            token_str.strip() == ""
                        ):
                            masked_logits[t_id] = -np.inf
                        # the model complete generation
                        # parameters and everything
                        if is_last_param:
                            if ',' in token_str:
                                masked_logits[t_id] = -np.inf
                        # there are any another parameters
                        else:
                            if '}' in token_str or ']' in token_str:
                                masked_logits[t_id] = -np.inf

        return masked_logits

    def generate(
            self, prompt: str,
            functions: list[FunctionDefinition],
            max_tokens: int = 50
            ) -> str:
        """Generate a complete, valid JSON function call for a prompt.

        Args:
            prompt (str): The user query to process.
            functions (list[FunctionDefinition]): Available functions.
            max_tokens (int): Maximum number of parameter tokens to
                generate. Defaults to 50.

        Returns:
            str: The full generated text containing the JSON payload,
            or 'NO_MATCH' if no function supports the prompt.
        """
        target_func, current_ids, _ = self._select_function(
            prompt, functions
            )

        if not self._is_prompt_supported(prompt, target_func):
            print("⚠️ Prompt does not match any available function.")
            return "NO_MATCH"

        print(f"\n----- Generating Parameters for {target_func.name} -----")
        transition_text = '", "parameters": {'
        transition_ids = self.model.encode(transition_text).tolist()[0]
        current_ids.extend(transition_ids)
        braces_balanced = False

        for step in range(max_tokens):
            current_text = str(self.model.decode(current_ids))
            logits = np.array(
                self.model.get_logits_from_input_ids(current_ids)
                )

            masked_logits = self._apply_mask(logits, current_text, target_func)
            next_token_id = int(np.argmax(masked_logits))
            # if the all logits have -inf then no one is valid
            # then stop generating
            if np.isneginf(masked_logits[next_token_id]):
                print("⚠️ No valid tokens available. Aborting generation.")
                return "NO_MATCH"

            next_token_str = self.vocab[next_token_id]

            print(f"Param Step {step + 1} | Appended Token:", end="")
            print(f"'{next_token_str}' (ID: {next_token_id})")
            current_ids.append(next_token_id)

            # Abort immediately if the model repeats the same token 4 times
            if len(current_ids) >= 4 and len(set(current_ids[-4:])) == 1:
                print("⚠️ Repetition loop detected. Aborting generation.")
                return "NO_MATCH"

            # Brace balance checking to ensure valid closure
            if "}" in next_token_str:
                full_text = current_text + next_token_str
                json_part = full_text[full_text.find('{"prompt":'):]

                open_braces = 0
                close_braces = 0
                in_quotes = False

                for i, ch in enumerate(json_part):
                    if ch == '"' and (i == 0 or json_part[i - 1] != '\\'):
                        in_quotes = not in_quotes
                    elif not in_quotes:
                        if ch == '{':
                            open_braces += 1
                        elif ch == '}':
                            close_braces += 1

                if open_braces > 0 and open_braces == close_braces:
                    print("✅ State: Balanced Braces. Stopping.")
                    braces_balanced = True
                    break

        if not braces_balanced:
            return "NO_MATCH"

        final_text: str = str(self.model.decode(current_ids))
        print("-------- Generation Complete -----------")
        return final_text
