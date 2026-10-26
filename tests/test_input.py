from src.io.reader import (
    load_function_definitions,
    load_prompt_items,
    validate_function_calls,
)

functions = load_function_definitions("data/input/functions_definition.json")
prompts = load_prompt_items("data/input/function_calling_tests.json")

# A handwritten output for the first input lets you test the validator now.
generated_data = [
    {
        "prompt": prompts[0].prompt,
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    }
]
calls = validate_function_calls(generated_data, functions, prompts[:1])
print(calls[0].model_dump())
