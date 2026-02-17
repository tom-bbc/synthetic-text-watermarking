# --------------------------------------------------------------------------- #
#                                 DISCLAIMER                                  #
# --------------------------------------------------------------------------- #

# coding=utf-8
# Copyright 2024 Google DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import argparse

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
#                                 CLI ENTRYPOINT                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Load arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b-it",
        help=("LM model to train the detector for."),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=("Temperature to sample from the model."),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=("Top K for sampling."),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help=("Top P for sampling."),
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=512,
        help=("Generation length for sampling."),
    )
    parser.add_argument(
        "--hf_hub_model_name",
        type=str,
        default=None,
        help=("HF hub model name for loading of saving the model."),
    )

    args = parser.parse_args()
    model_name = args.model_name
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    generation_length = args.generation_length
    repo_name = args.hf_hub_model_name

    # -----------------------------------------------------------------------------
    # Set hyperparameters
    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if DEVICE.type not in ("cuda", "tpu"):
        raise ValueError(
            "We have found the training stable on GPU and TPU, we are working on"
            " a fix for CPUs"
        )

    if repo_name is None:
        raise ValueError(
            "When loading from pretrained detector model name cannot be None."
        )

    # -----------------------------------------------------------------------------
    # Load detector model
    best_detector = BayesianDetectorModel.from_pretrained(repo_name).to(DEVICE)

    # Load generator model and tokenizer
    model_name = best_detector.config.model_name
    watermark_config_dict = best_detector.config.watermarking_config
    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        **watermark_config_dict, device=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    synthid_text_detector = SynthIDTextWatermarkDetector(
        best_detector, logits_processor, tokenizer
    )

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    watermarking_config = SynthIDTextWatermarkingConfig(**watermark_config_dict)

    # -----------------------------------------------------------------------------
    # Run generation and detection over set of eval prompts
    prompts = ["Write a essay on cats."]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    _, inputs_len = inputs["input_ids"].shape

    outputs = model.generate(
        **inputs,
        watermarking_config=watermarking_config,
        do_sample=True,
        max_length=inputs_len + generation_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    outputs = outputs[:, inputs_len:]
    result = synthid_text_detector(outputs)

    # You should set this based on expected FPR (false positive rate)
    # and TPR (true positive rate). Check our demo at HF Spaces for more info.
    upper_threshold = 0.95
    lower_threshold = 0.12

    if result[0][0] > upper_threshold:
        print("The text is watermarked.")
    elif lower_threshold < result[0][0] < upper_threshold:
        print("It is hard to determine if the text is watermarked or not.")
    else:
        print("The text is not watermarked.")
