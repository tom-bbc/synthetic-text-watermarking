# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import os
from typing import List, Optional

import numpy as np
import tiktoken
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
)

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


# --------------------------------------------------------------------------- #
#                      SYNTHID FOR TEXT IMPLEMENTATION                        #
# --------------------------------------------------------------------------- #


def watermark_synthid(model_name: str, prompt: str, watermark_keys: List[int]):
    # Instantiate generator model and tokenizer
    print(f" << * >> Instantiating model: '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create SynthID watermarking config
    # Specify word-seq length: The lower the values are, the more likely the watermarks are to survive heavy editing, but the harder they are to detect  # noqa
    word_seq_len = 5
    print(f" << * >> Instantiating watermarker with keys: {watermark_keys}")

    watermarking_config = SynthIDTextWatermarkingConfig(
        keys=watermark_keys, ngram_len=word_seq_len
    )
    print(f" << * >> Watermarking config set up: {watermarking_config}")

    # Format model inputs
    print(f" << * >> Input prompt: '{prompt}'")

    prompt_toks = tokenizer([prompt], return_tensors="pt")
    print(f" << * >> Encoded prompt: {prompt_toks}")

    # Generate output that includes watermark
    print(" << * >> Generating response")
    response = model.generate(
        **prompt_toks, watermarking_config=watermarking_config, do_sample=True
    )

    watermarked_text = tokenizer.batch_decode(response, skip_special_tokens=True)
    print(f" << * >> Watermarked text output: '{watermarked_text}'")

    return watermarked_text


if __name__ == "__main__":
    # Legacy
    # model_name = "gpt-4o"
    # pregenerated_tokens_file = "outputs/logprobs/test_20251002.json"
    # format_model_output(model_name, pregenerated_tokens_file)

    # Load credentials: Hugging Face and SynthID watermarking key
    credentials_filepath = "credentials.json"
    with open(credentials_filepath, "r", encoding="utf-8") as fp:
        credentials = json.load(fp)

    # Authenticate with Hugging Face for model access
    hf_access_token = credentials["hf_access_token"]
    login(hf_access_token)
    print(" << * >> Successfully authenticated with Hugging Face")

    # Specify generator model
    model_name = "google/gemma-2b"

    # Specify waterkarking keys: a list of 20-30 random integers that serve as your private digital signature  # noqa
    watermark_keys = credentials["synthid_watermarking_keys"]

    # Run text watermarking process
    prompt = "What is the capital of France?"

    print(" << * >> Running watermarking process")
    watermark_synthid(model_name, prompt, watermark_keys)
