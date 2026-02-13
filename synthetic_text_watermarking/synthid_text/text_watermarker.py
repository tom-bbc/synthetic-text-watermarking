# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BayesianDetectorModel,
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

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
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    print(f"Model running on device: {model.device}")

    return model, tokenizer


def watermark_synthid(
    model_name: str, prompt: str, watermark_keys: List[int], device: torch.device
):
    """
    Generate a text response to the input 'prompt' using the HF model 'model_name'
    that is watermarked using Google's SynthID.
    """

    # Instantiate generator model and tokenizer
    print(f" << * >> Instantiating model: '{model_name}'")

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

    prompt_toks = tokenizer(prompt, return_tensors="pt")
    prompt_toks = prompt_toks.to(device)
    print(f" << * >> Encoded prompt: {prompt_toks}")

    max_new_tokens = 1024
    temperature = 1.0

    # Generate output that includes watermark
    print(" << * >> Generating response")
    response = model.generate(
        input_ids=prompt_toks["input_ids"],
        attention_mask=prompt_toks["attention_mask"],
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        watermarking_config=watermarking_config,
    )

    watermarked_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    print(f" << * >> Watermarked text output: '{watermarked_text}'")

    return watermarked_text


def bayesian_detector(input_text: str, device: torch.device) -> float:
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
    text_toks = tokenizer(input_text, return_tensors="pt")
    print(f" << * >> Input text: '{input_text}'")
    print(f" << * >> Encoded text: {text_toks}")

    watermark_likelihood = detector(text_toks.input_ids)
    watermark_likelihood = watermark_likelihood[0][0]
    print(f" << * >> Likelihood of watermark: {watermark_likelihood * 100:.2f}%")

    return watermark_likelihood


def mean_detector(
    input_text: str,
    watermarking_config: Dict,
    generator_model: str,
    device: torch.device,
) -> float:
    """
    Check whether a given text 'input_text' is likely to have been watermarked
    using Google's SynthID.
    """

    _, tokenizer = load_model(generator_model, device)

    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        **watermarking_config,
        device=device,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    if input_ids.shape[1] == 0:
        return 0.0

    # Employs a pre-computed sampling table to sample watermarking values (g-values)
    # based on the generated keys.
    g_values = logits_processor.compute_g_values(input_ids)
    mean_g = g_values.float().mean().item()

    print(f" << * >> Mean g-score: {mean_g}")

    return mean_g


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Load arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Run model generation with watermarking of generated text.",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default=None,
        help="Input text to pass to generator or detector model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON file defining watermarking config.",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run watermark detection on an input text.",
    )
    parser.add_argument(
        "--detector-type",
        type=str,
        default="mean",
        help="Type of watermark extraction algorithm to use.",
    )

    args = parser.parse_args()
    watermark = args.watermark
    input_text = args.input_text
    config = args.config
    detect = args.detect
    detector_type = args.detector_type

    print(f" << * >> Run watermarking: {watermark}")
    print(f" << * >> Input text: {input_text}")
    print(f" << * >> Watermarking config: {config}")
    print(f" << * >> Run detection: {detect}")
    print(f" << * >> Detector type: {detector_type}")

    if config is None:
        config = Path(__file__).parent / "watermark_config.json"

    if config is not None and os.path.isfile(config):
        # Load waterkarking keys: a list of 20-30 random integers that serve as your private digital signature  # noqa
        with open(config, "r", encoding="utf-8") as fp:
            watermark_config = json.load(fp)

        watermark_keys = watermark_config["keys"]
        print(f" << * >> Watermark key: {watermark_keys}")
    else:
        watermark_config = None
        watermark_keys = None

    # -----------------------------------------------------------------------------
    # Configure model and hyperparams
    model_name = "google/gemma-2-2b-it"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # -----------------------------------------------------------------------------
    # Run model generation with watermarking of generated text
    if watermark and watermark_keys is not None:
        # Validate input text
        if input_text is None:
            prompt = "Please re-write the following article in the style of a BBC News Article. \n\nInput Article:\nA report revealed that 253 potential victims of slavery were reported in Hampshire and the Isle of Wight, of which one in four were children. Modern slavery, which includes human trafficking, is the illegal exploitation of people for personal or commercial gain. It can take different forms of slavery, such as domestic or labour exploitation, organ harvesting, EU Status exploitation, and financial, sexual and criminal exploitation. Each year, Hampshire and Isle of Wight Fire and Rescue Authority (HIWFRA), combined by all four authorities, the three unitary councils, and the county council, spends around £99m on making \"life safer\" in the county and preventing slavery and human trafficking. However, a recent report of the HIWFRA has revealed that by June 2023, there were 253 potential victims identified of modern slavery in Hampshire and the Isle of Wight. Of them, one in four were children. According to the Government's UK Annual Report on Modern Slavery, 10,613 potential victims were referred to the National Referral Mechanism in the year ended September 2021. In case any member of the Authority or any of its staff suspects slavery or human trafficking activity either within the community or the organisation, then the concerns will be reported through the Service's Safeguarding Reporting Procedure.\n\nBBC Article:\n"  # noqa
        else:
            prompt = input_text

        # Run text generation and watermarking process
        print(" << * >> Running watermarking process")
        synthetic_text = watermark_synthid(model_name, prompt, watermark_keys, device)

    elif input_text is None:
        synthetic_text = "A report revealed that 253 potential victims of slavery were reported in Hampshire and the Isle of Wight, of which one in four were children. Modern slavery, which includes human trafficking, is the illegal exploitation of people for personal or commercial gain. It can take different forms of slavery, such as domestic or labour exploitation, organ harvesting, EU Status exploitation, and financial, sexual and criminal exploitation. Each year, Hampshire and Isle of Wight Fire and Rescue Authority (HIWFRA), combined by all four authorities, the three unitary councils, and the county council, spends around £99m on making \"life safer\" in the county and preventing slavery and human trafficking. However, a recent report of the HIWFRA has revealed that by June 2023, there were 253 potential victims identified of modern slavery in Hampshire and the Isle of Wight. Of them, one in four were children. According to the Government's UK Annual Report on Modern Slavery, 10,613 potential victims were referred to the National Referral Mechanism in the year ended September 2021. In case any member of the Authority or any of its staff suspects slavery or human trafficking activity either within the community or the organisation, then the concerns will be reported through the Service's Safeguarding Reporting Procedure.\n\nBBC Article:\n"  # noqa
    else:
        synthetic_text = input_text

    # -----------------------------------------------------------------------------
    # Run watermark detection on an input text
    if detect and watermark_keys is not None:
        if detector_type == "mean" and watermark_config is not None:
            watermarked = mean_detector(
                synthetic_text, watermark_config, model_name, device
            )

        elif detector_type == "bayesian":
            watermarked = bayesian_detector(synthetic_text, device)
