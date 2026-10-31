*This project has been created as part of the 42 curriculum by asuleime*

# Call me maybe

## Description

Translate natural-language requests into a function name and typed arguments
using Qwen3-0.6B and the supplied `llm_sdk` (can be installed from the zip archive together with example prompts). The application does not execute
functions. Model scores select functions and argument values; byte-level
constraints enforce valid JSON and supported scalar types.

## Instructions

Unpack `llm_sdk` from `zip_archives`:
```bash
unzip zip_archives/llm_sdk.zip -d .
```

Python 3.10+ and `uv` are required. Run `make install` or `uv sync [--group dev]`.

Run with example prompts and functions definitions from `data/input` and default parameters:
```bash
make run
```

Example run with use of `cuda` and specification of functions definition, input (json prompts), and output directories:
```bash
HF_HUB_OFFLINE=1 uv run python -m src --device cuda \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Omit `--device` to use `CMM_DEVICE` or the CPU default. CUDA requires a supported
GPU/driver and the SDK's locked `accelerate` dependency. `--max-new-tokens`
(default 128 per value) and `--max-context-tokens` (default 1536) bound generation.
Missing/malformed input fails before model initialization. Empty requests write
`[]`. Any failed request prevents replacement of the previous output file.

Optional flags:
all flags are used as below:
```bash
make run ARGS=[--flag flag_arg]
```
where `--flag` can be:
- `--functions_definition` to specify the path for functions definitions in the json format
- `--input` and `--output` to specify the paths for input prompts and function calls, both in the json format
- `--device` to choose between `cpu`, `cuda`, or `mps`
- `--max-new-tokens` for maximum tokens per generation, default is 128
- `--max-context-tokens` maximum tokens for the context (including prompts, functions definitions, and examples, when applicable) passed for an LLM call
- `--profile` to choose if bonus features (described at the end of this README) will be added
- `--tokenizer` to choose local tokenizer implementation if set to `local` versus `sdk`
- `--recovery` set to `retry` to enable repeated attempt if the LLM fails to score constrained set of tokens to allow for generation; retries are done with an error reason added to the context of LLM calls
- `--model` to choose from Qwen/Qwen3-0.6B (default), HuggingFaceTB/SmolLM2-135M-Instruct, HuggingFaceTB/SmolLM2-360M-Instruct, and HuggingFaceTB/SmolLM-135M-Instruct
- `--model-snapshot` to select some locally saved snapshot of the model (via its hash key); it contains a set of required tokens, paths to merges and tokenizer config files
- `--diagnostics-output` path to select the output for the diagnostic summary of a run (`None` by default, which does not create a file with this summary)
- `--trace-output`, goes to `data/traces` by default, a .jsonl log that contains the state of each prompt per run, also being a source for visuals (visuals can be run on these .jsonl after the run as well)
- `--visual`, with no argument. If passed, shows live visual of the generation progress.

Extra commands:
- `make clean` cleans up cache files from pytest, mypy, uv
- `make debug` runs the program with the `dbg` debugger
- `make lint` and `make lint-strict` run linting with mypy and flake8, with strict mypy checks for the latter
- `make test` runs the pytest suite
- `make check` runs `make lint-strict` and `make test`
- `make bench` runs the program with 10 extra prompts that represent other prompts that test different cases and some tricky edge cases like empty string parameters
- `make visualize TRACE=path_to_trace_file` runs a visual with a saved trace


## The algorithm

1. Validate function definitions and original prompts with Pydantic.
2. Build a Qwen context containing available definitions. A byte trie limits
   generated names to the JSON spellings of those functions, including the
   closing quote. Only the LLM scores determine which path wins. Once exactly
   one terminal name remains reachable, its fixed suffix needs no further scoring.
3. Rebuild context with the selected definition, then generate arguments in
   schema order. A decimal DFA handles numbers/integers; a JSON-string DFA
   jointly tracks escapes, Unicode surrogate pairing, and partial UTF-8.
   Boundary adapters allow Qwen tokens that combine the opening quote with
   string content, or a completed value with exact known following syntax.
   A small literal trie handles booleans; null is fixed syntax.
4. One shared loop tests whole token byte sequences, chooses the highest
   finite allowed score, and appends exactly the selected ID. Negative infinity
   is unavailable; NaN and positive infinity are errors if among allowed scores.
5. Parse completed fragments into Python values. Re-encode whole context only
   at documented completed-value boundaries. Numbers consume their expected
   comma/brace and may include the final call's closing syntax. Strings consume
   both quotes and may include a prefix of the exact following syntax. No
   token suffix is dropped.
6. Validate the complete batch and atomically serialize `prompt`, `name`, and
   `parameters` with `allow_nan=False`.

## Design decisions

The production `examples` context follows Qwen's recorded tool layout with
thinking disabled, all definitions for selection, and only the selected
definition for argument extraction. Two paired demonstrations clarify regex
extraction when the selected function declares `source_string`, `regex`, and
`replacement`.

External requests, descriptions, names, keys, and completed string values are
JSON-serialized with template-marker escaping. Original input is stored
separately from tokenizer-normalized context, which protects message boundaries.

Supported argument types are `string`, `number`, `integer`, `boolean`, and
`null`; all declared parameters are required. Extra JSON Schema metadata is preserved but constraints such
as `minimum`, `enum`, and `pattern` are not enforced. Decimal notation is generated
without exponent syntax. Floats must be finite and integers retain Python's exact
integer representation. Token, byte, or context exhaustion is an explicit
failure, never generating fabricated value or using any fallback function.

The bounded per-decoder candidate cache includes the complete DFA state and
constraint configuration. No model answers or prompt-to-function mappings are
cached. The application imports only the SDK's public interface; framework
imports remain inside the supplied SDK.

## Testing strategy

```bash
make test
make lint-strict
```

Unit tests use tiny vocabularies and controlled score functions, covering
invalid high-scoring tokens, exact token histories, split UTF-8, escapes,
surrogates, numeric termination, length/overflow failures, scalar dispatch,
marker safety, batch errors, and atomic output.

## Performance analysis

The actual CLI processed the 11 provided prompts in **220.16 seconds** on a WSL2 with 4 GB RAM with CUDA enabled, including startup: 11/11 valid outputs, 11/11 correct functions, and 11/11 exact calls.

Processing speed might differ on other devices.

## Challenges

Context prompt creation to nudge model logit scores and ensuring model logits contain at least on of the valid next tokens at each step of the generation algorithm is the main challenge that was solved with iterative development for each example function. Each function's schema is constructed from its definition from the .json file and parameter types thence
determine what DFA the program uses for constrained decoding.

Nested function parameters were hard to implement as well, and I needed more time to understand them than to write their code. In general, creating wrapping state captures the right way was also difficult, as I was not used to do this much before.

## Example usage

Input: `What is the sum of 2 and 3?`

```json
{"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen function-calling guide](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [JSON specification](https://www.rfc-editor.org/rfc/rfc8259)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [GPT-2 byte alphabet reference](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Trie](https://en.wikipedia.org/wiki/Trie)

### AI Usage
GPT-6 Astra and GPT-6.1 Sol were used to:
- Write unit tests and generate example prompts similar to those from the `zip_archives/data.zip`. 
- Run test suites and generate diagnostic reports.
- Generate docstring drafts for function and method signatures 
- Create an implementation plan with pseudocode for `TraceWriter` and nested function arguments DFA
- Get advice like choosing between having a class (when knowledge of state and object use are required) with methods and a function (when only a change of state is needed, or simple operations are done), and placing lazy imports to import some packages only when they are required
- Explain how tokenizers build the token vocab; differences in raw byte values of symbols and their representation in different places and how unicode characters are handled
- Add partial UTF-8 char handling for string DFAs (get special symbol hex codes and explain what their additions mean)
- Choose optimal limits for maximum context tokens and maximum new tokens per LLM call

## Bonus parts
- Support for several SmolLM models: HuggingFaceTB/SmolLM2-135M-Instruct, HuggingFaceTB/SmolLM2-360M-Instruct, HuggingFaceTB/SmolLM-135M-Instruct
- Tokenizer recode
- Error recovery with failure to get any of the allowed inputs with appended error reason on retry context
- Performance optimization with caching and validation of output batches with mid-batch error exceptions
- Unit tests covering all of the classes and functionality of the project
- Rich visualization during generation with current state of function calls in the run and for previous runs' .jsonl traces
- Support for nested function arguments with depth of up to 8, at most 32 array items and total of 4096 bytes
- Implementation of `decode()` and `encode()` similar to the `llm_sdk`
- Demonstration script for integration of constrained decoding with encoding and decoding used in the function