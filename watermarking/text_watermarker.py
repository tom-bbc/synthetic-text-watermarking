# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import tiktoken
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BayesianDetectorModel,
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
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
#                  SYNTHID TEXT WATERMARKING & DETECTION                      #
# --------------------------------------------------------------------------- #


def load_model(
    model_name: str, device: torch.device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Instantiate langauage model and tokenizer from Hugging Face."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    print(f"Model running on device: {model.device}")

    return model, tokenizer


def watermark_synthid(model_name: str, prompt: str, watermark_keys: List[int]):
    """
    Generate a text response to the input 'prompt' using the HF model 'model_name'
    that is watermarked using Google's SynthID.
    """

    # Instantiate generator model and tokenizer
    print(f" << * >> Instantiating model: '{model_name}'")
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model, tokenizer = load_model(model_name, device)

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
    prompt_toks = prompt_toks.to(device)
    print(f" << * >> Encoded prompt: {prompt_toks}")

    max_new_tokens = 1024
    temperature = 1.0

    # Generate output that includes watermark
    print(" << * >> Generating response")
    response = model.generate(
        **prompt_toks,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        watermarking_config=watermarking_config,
    )

    watermarked_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    print(f" << * >> Watermarked text output: '{watermarked_text}'")

    return watermarked_text


def detect_synthid(input_text: str) -> float:
    """
    Check whether a given text 'input_text' is likely to have been watermarked
    using Google's SynthID.

    Note: This uses a DUMMY detector model for demo purposes only.
    A custom detector should be trained for any use in production.
    https://github.com/huggingface/transformers/blob/v4.46.0/examples/research_projects/synthid_text/detector_training.py
    """

    # Load DUMMY detector model
    detector_model_name = "joaogante/dummy_synthid_detector"
    bayesian_detector_model = BayesianDetectorModel.from_pretrained(detector_model_name)
    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        **bayesian_detector_model.config.watermarking_config, device="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(bayesian_detector_model.config.model_name)

    detector = SynthIDTextWatermarkDetector(
        bayesian_detector_model, logits_processor, tokenizer
    )
    print(" << * >> Instantiated detector model")

    # Pass input text to the detector model
    text_toks = tokenizer([input_text], return_tensors="pt")
    print(f" << * >> Input text: '{input_text}'")
    print(f" << * >> Encoded text: {text_toks}")

    watermark_likelihood = detector(text_toks.input_ids)
    watermark_likelihood = watermark_likelihood[0][0]
    print(f" << * >> Likelihood of watermark: {watermark_likelihood:.4f}")

    return watermark_likelihood


if __name__ == "__main__":
    # Legacy
    # model_name = "gpt-4o"
    # pregenerated_tokens_file = "outputs/logprobs/test_20251002.json"
    # format_model_output(model_name, pregenerated_tokens_file)

    watermark = True
    detect = True
    synthetic_text = "This is a test input"

    if watermark:
        # Load credentials: Hugging Face and SynthID watermarking key
        credentials_filepath = "credentials.json"
        with open(credentials_filepath, "r", encoding="utf-8") as fp:
            credentials = json.load(fp)

        # Authenticate with Hugging Face for model access
        hf_access_token = credentials["hf_access_token"]
        login(hf_access_token)
        print(" << * >> Successfully authenticated with Hugging Face")

        # Specify generator model and input prompt
        model_name = "google/gemma-2-2b-it"
        prompt = "Please re-write the following article in the style of a BBC News Article. \n\nInput Article:\nA report revealed that 253 potential victims of slavery were reported in Hampshire and the Isle of Wight, of which one in four were children. Modern slavery, which includes human trafficking, is the illegal exploitation of people for personal or commercial gain. It can take different forms of slavery, such as domestic or labour exploitation, organ harvesting, EU Status exploitation, and financial, sexual and criminal exploitation. Each year, Hampshire and Isle of Wight Fire and Rescue Authority (HIWFRA), combined by all four authorities, the three unitary councils, and the county council, spends around £99m on making \"life safer\" in the county and preventing slavery and human trafficking. However, a recent report of the HIWFRA has revealed that by June 2023, there were 253 potential victims identified of modern slavery in Hampshire and the Isle of Wight. Of them, one in four were children. According to the Government's UK Annual Report on Modern Slavery, 10,613 potential victims were referred to the National Referral Mechanism in the year ended September 2021. In case any member of the Authority or any of its staff suspects slavery or human trafficking activity either within the community or the organisation, then the concerns will be reported through the Service's Safeguarding Reporting Procedure.\n\nBBC Article:\n"  # noqa

        # Specify waterkarking keys: a list of 20-30 random integers that serve as your private digital signature  # noqa
        watermark_keys = credentials["synthid_watermarking_keys"]

        # Run text generation and watermarking process
        print(" << * >> Running watermarking process")
        synthetic_text = watermark_synthid(model_name, prompt, watermark_keys)

    # Run watermark detection process
    if detect:
        watermarked = detect_synthid(synthetic_text)
