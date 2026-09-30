*This activity has been created as part of the 42 curriculum by anaobeid.*

# Call Me Maybe — Constrained LLM Function Calling

---

## Description

**Call Me Maybe** is a deterministic, schema-driven function-calling engine built around a local Small Language Model (`Small_LLM_Model`). 

The primary goal of this activity is to bridge the gap between natural language user queries and structured programmatic execution. When generating text freely, small language models frequently fail at tool calling: they hallucinate non-existent function names, invent extra parameters, violate expected data types, or produce broken JSON syntax.

Instead of relying on post-generation regex extraction or retry loops, this project implements **Constrained Decoding via Logit Masking**. By intercepting the model's raw probability distribution (`logits`) at every token generation step and masking invalid tokens with negative infinity (`-np.inf`) before greedy selection (`np.argmax`), the engine guarantees 100% syntactically valid JSON output that strictly adheres to predefined function schemas.

---

## Instructions

### Prerequisites
* **Python**: `3.10` or higher
* **Package Manager**: `uv`

### 1. Installation
Install all required dependencies (`pydantic`, `numpy`, `torch`, `transformers`, `huggingface_hub`, `flake8`, `mypy`) into an isolated `.venv` environment:
```bash
make install
```

### 2. Execution
Run the main script using the default input and output paths:
```bash
make run
```

### 3. Debugging
Run the application in interactive debug mode using Python's built-in debugger (`pdb`):
```bash
make debug
```

### 4. Linting & Type Checking
Verify code formatting (`flake8`) and static type annotations (`mypy`) using the mandatory flags:
```bash
make lint
```

### 5. Cleaning
Remove temporary caches and build artifacts (`__pycache__`, `.mypy_cache`, `*.egg-info`, `build/`, `dist/`):
```bash
make clean
```

---

## Example Usage

### Running with Default Paths
```bash
make run
```

### Running with Custom Arguments
You can specify custom paths for the function definitions, test prompts, and output file via CLI flags:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Sample Input & Output
**Input Prompt (`function_calling_tests.json`):**
```json
[
    {
        "prompt": "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years"
    }
]
```

**Generated Output (`function_calls.json`):**
```json
[
    {
        "prompt": "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years",
        "name": "fn_calculate_compound_interest",
        "parameters": {
            "principal": 1234567.89,
            "rate": 0.0375,
            "years": 23
        }
    }
]
```

---

## Algorithm Explanation

Our constrained decoding approach operates as a token-by-token state machine divided into four stages:

### 1. Dynamic Function Selection (`_select_function`)
1. **Prompt Construction:** The engine formats a context block listing all available functions, their parameter names, and their descriptions, followed by the sanitized user prompt (`json.dumps(user_prompt)[1:-1]`) and the JSON prefix:
   `Action: {"prompt": "<safe_prompt>", "name": "`
2. **Prefix Logit Masking:** For up to 15 generation steps, the model's raw logits are initialized to `-np.inf`. The engine iterates through the vocabulary and unmasks a token ID **only** if `generated_name + token_str` forms a valid prefix of (or matches) a defined function name in `available_functions`.
3. **Termination:** As soon as `generated_name` exactly matches a valid function name, selection halts.

### 2. Deterministic Structural Transition (`generate`)
Rather than relying on the LLM to generate structural JSON boilerplate between the function name and its arguments, the engine deterministically encodes and appends the transition tokens:
```python
transition_text = '", "parameters": {'
```

### 3. Schema-Guided Parameter Masking (`_apply_mask`)
At every step inside the `"parameters": {` block, the masker inspects the decoded text generated so far:
* **Quote-Aware Structural Tracking:** The algorithm scans character-by-character, toggling an `in_quotes` flag whenever an unescaped double quote (`"`) is encountered. Commas (`,`) and closing braces (`}`) are counted **strictly outside of quotes** to determine `current_param_index`.
* **Key Generation Mode (Before `:`):** If the current parameter segment does not yet contain a colon `:`, all tokens are masked with `-np.inf` except those that form a valid prefix of the expected schema key `"{next_param}":`.
* **Value Generation Mode (After `:`):** Once `:` is generated, constraints are applied based on the parameter's schema type:
  * **`number` & `integer`:** Alphabetic characters, quotes, and brackets are masked out. For `number` (float), closing delimiters (`,` and `}`) are blocked until a decimal point `.` and at least one fractional digit are generated. Standalone spaces after digits are blocked to prevent repetition loops.
  * **`string`:** Tracks opening and closing quotes while allowing valid JSON escapes (`\\` and `\"`). Once the string is closed, letters and extra quotes are blocked, forcing the model to output `,` (if more parameters remain) or `}` (if it is the final parameter).

### 4. Quote-Aware Brace Balancing (`generate`)
Whenever a token containing `}` is appended, the generator counts `{` and `}` characters outside of string literals across the generated JSON block. When `open_braces == close_braces`, the JSON object is complete and generation terminates immediately.

---

## Design Decisions

1. **In-Memory Vocabulary Pre-Caching (`self.vocab`):**
   The model's vocabulary contains **151,643 tokens**. Decoding token IDs on the fly inside the masking loop would require millions of tokenizer calls per prompt. By pre-decoding the entire vocabulary once inside `JSONGenerator.__init__`, token lookup during masking becomes an $O(1)$ list access.
2. **Pydantic Data Validation (`src/models.py` & `src/ingestion.py`):**
   Used `pydantic.BaseModel` (`FunctionDefinition`, `FunctionParameter`, `TestPrompt`, `FunctionCallResult`) configured with `extra="forbid"` to enforce strict schema validation. This guarantees that malformed input files, unrecognized keys (e.g., typos like `nme` instead of `name`), or unexpected schema structures are caught immediately. The ingestion module provides precise error path tracing (e.g., `parameters -> a -> typ`) and gracefully handles optional schema fields like `returns` before any inference occurs.
3. **Character-Level Quote-Aware Parsing over Regex:**
   Regex struggles with partial, in-progress JSON strings containing escaped quotes or SQL fragments. A lightweight character-level loop tracking `in_quotes` and `\\` escapes provides exact state detection at any arbitrary token boundary.
4. **Modular Package Architecture (`src/`):**
   Separated concerns into distinct modules: CLI & I/O orchestration (`__main__.py`), fault-tolerant file loading (`ingestion.py`), data schemas (`models.py`), and inference logic (`generator.py`).

---

## Performance Analysis
* **Speed:** Pre-caching the 151,643 vocabulary tokens at startup (~1 second) reduced per-step masking latency to milliseconds. Complete generation across all test prompts finishes in a few seconds on standard hardware.
* **Reliability:** Greedy decoding (`np.argmax`) combined with deterministic logit masking ensures 100% reproducible outputs across runs, with comprehensive `try/except` blocks handling file I/O and JSON parsing gracefully.

---

## Challenges Faced

1. **Compound Tokenizer Delimiters (`!",`):**
   * *Problem:* On prompts ending with an exclamation mark (e.g., `Hello {user}'s profile!`), the model selected a single compound token `'!",'` (ID: 18789) on the final parameter, closing the string and adding a trailing comma simultaneously, which forced it to hallucinate a dummy parameter `"..."`.
   * *Solution:* Added a lookahead check on the last parameter that masks any token containing both `"` and `,` if the comma appears after the quote (`token_str.rfind(',') > token_str.rfind('"')`).
2. **Escaped Quotes & Backslashes in Strings:**
   * *Problem:* Windows paths (`C:\Users\john\config.ini`) and nested quotes (`Say "hello" to {name}`) either caused `JSONDecodeError` due to raw backslashes or tricked the model into adding an extra escaped quote `\"` at the very end of the string (`{name}\""`).
   * *Solution:* Restricted `\` tokens strictly to valid JSON escapes (`\\` and `\"`), and capped the allowed number of `\"` sequences in `value_segment` to match the exact count of double quotes in the original prompt (`orig_prompt.count('"')`).
---

## Testing Strategy

1. **Public Test Suite Validation:** Tested basic arithmetic, string reversal, greetings, square roots, and regex substitution prompts against `data/input/function_calling_tests.json`.
2. **JSON Structural Verification:** Every generated output is parsed through `json.loads()` in `__main__.py` to verify structural integrity before saving.
3. **Static Analysis & Linting:** Enforced strict code quality using `flake8` (PEP 8 compliance, <79 chars per line) and `mypy` with `--disallow-untyped-defs`, `--warn-return-any`, and `--strict`.
4. **Fault-Tolerant Ingestion:** Evaluated extreme edge cases in JSON input files (e.g., missing mandatory keys, extraneous fields, invalid root structures). The engine is designed to never crash abruptly, but rather exit gracefully with descriptive, user-friendly error locations to assist debugging.

---

## Resources

### References
* **[Pydantic Documentation](https://docs.pydantic.dev/)** — Schema definition and data validation models.
* **[NumPy Documentation](https://numpy.org/doc/)** — Array masking (`np.full_like`, `-np.inf`) and greedy decoding (`np.argmax`).
* **[Hugging Face Transformers — LogitsProcessor](https://huggingface.co/docs/transformers/internal/generation_utils)** — Theoretical foundation for autoregressive logit manipulation.
* **[RFC 8259 (JSON Specification)](https://datatracker.ietf.org/doc/html/rfc8259)** — Rules for JSON grammar, string escaping, and number representation.
* **[PEP 8](https://peps.python.org/pep-0008/) & [PEP 257](https://peps.python.org/pep-0257/)** — Python style guide and docstring conventions.

### AI Usage Disclosure
All core project architecture, state-machine design, and logit-masking algorithms were implemented and tested manually. AI tools were used selectively as a reference assistant for the following tasks:
* **Conceptual Research:** Reviewing theoretical concepts regarding LLM tokenization, vocabulary ID mapping, and autoregressive greedy decoding.
* **Tooling & Static Analysis:** Looking up configuration flags and exclusion syntax for `uv`, `mypy` strict type checking, `flake8`, and `Makefile` automation rules.
