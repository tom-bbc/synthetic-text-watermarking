# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import os
from typing import Optional

import numpy as np
import tiktoken

# --------------------------------------------------------------------------- #
#                  ENCODER FOR PRE-GENERATED MODEL OUTPUTS                    #
# --------------------------------------------------------------------------- #


def format_model_output(
    model_name: str,
    restore_from_file: Optional[str] = None,
) -> Optional[str]:
    # Load pre-generated model output tokens
    if isinstance(restore_from_file, str) and os.path.isfile(restore_from_file):
        with open(restore_from_file, "r", encoding="utf-8") as f:
            model_output = json.load(f)

        print(f" << * >> Restoring LLM logprobs from file: '{restore_from_file}'")

    else:
        print(" -- * -- Error: no valid 'restore_from_file' file provided.")
        return

    # Instantiate token encoder for given model
    encoder = tiktoken.encoding_for_model(model_name)

    # Encode all tokens from the pre-generated model output
    generated_tokens = [token["token"] for token in model_output]
    generated_token_ids = [encoder.encode(token) for token in generated_tokens]
    generated_token_ids = np.array(generated_token_ids).flatten()

    print(f" << * >> Encoded token IDs: {generated_token_ids}")


if __name__ == "__main__":
    model_name = "gpt-4o"
    pregenerated_tokens_file = "outputs/logprobs/test_20251002.json"
    format_model_output(model_name, pregenerated_tokens_file)
