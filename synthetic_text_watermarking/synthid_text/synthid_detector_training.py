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
import dataclasses
import enum
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from synthid_training_utils import (
    process_raw_model_outputs,
    update_fn_if_fpr_tpr,
    upload_model_to_hf,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BayesianDetectorConfig,
    BayesianDetectorModel,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

# --------------------------------------------------------------------------- #
#                               TRAINING ARGUMENTS                            #
# --------------------------------------------------------------------------- #


@enum.unique
class ValidationMetric(enum.Enum):
    """Direction along the z-axis."""

    TPR_AT_FPR = "tpr_at_fpr"
    CROSS_ENTROPY = "cross_entropy"


@dataclasses.dataclass
class TrainingArguments:
    """Training arguments pertaining to the training loop itself."""

    eval_metric: Optional[ValidationMetric] = dataclasses.field(
        default=ValidationMetric.TPR_AT_FPR,
        metadata={"help": "The evaluation metric used."},
    )


# --------------------------------------------------------------------------- #
#                           GENERATE TRAINING DATASET                         #
# --------------------------------------------------------------------------- #


def generate_raw_samples(num_negatives, neg_batch_size, tokenizer, device):
    # gsutil cp -r gs://tfds-data/datasets/wikipedia/20230601.en/ ../dataset/wikipedia/
    download_dir = "../dataset/"
    dataset_name = "wikipedia/20230601.en"
    dataset, info = tfds.load(
        dataset_name, split="train", with_info=True, data_dir=download_dir
    )
    dataset = dataset.take(num_negatives)

    # Convert the dataset to a DataFrame
    df = tfds.as_dataframe(dataset, info)
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    tf.random.set_seed(0)
    ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size=neg_batch_size)

    tokenized_uwm_outputs = []
    padded_length = 1000  # Pad to this length (on the right) for batching

    for i, batch in tqdm(enumerate(ds), desc="Extracting raw samples"):
        responses = [val.decode() for val in batch["text"].numpy()]
        inputs = tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
        ).to(device)
        inputs = inputs["input_ids"].cpu().numpy()

        if inputs.shape[1] >= padded_length:
            inputs = inputs[:, :padded_length]
        else:
            inputs = np.concatenate(
                [
                    inputs,
                    np.ones((neg_batch_size, padded_length - inputs.shape[1]))
                    * tokenizer.eos_token_id,
                ],
                axis=1,
            )

        tokenized_uwm_outputs.append(inputs)

        if len(tokenized_uwm_outputs) * neg_batch_size > num_negatives:
            break

    return tokenized_uwm_outputs


def generate_watermarked_samples(
    model,
    tokenizer,
    watermark_config,
    num_pos_batches,
    pos_batch_size,
    temperature,
    max_output_len,
    top_k,
    top_p,
    device,
):
    download_dir = "../dataset/"
    dataset_name = "Pavithree/eli5"
    eli5_prompts = datasets.load_dataset(dataset_name, data_dir=download_dir)

    wm_outputs = []

    for batch_id in tqdm(range(num_pos_batches), desc="Generating watermarked samples"):
        prompts = eli5_prompts["train"]["title"][
            batch_id * pos_batch_size : (batch_id + 1) * pos_batch_size
        ]
        prompts = [prompt.strip('"') for prompt in prompts]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        _, inputs_len = inputs["input_ids"].shape

        outputs = model.generate(
            **inputs,
            watermarking_config=watermark_config,
            do_sample=True,
            max_length=inputs_len + max_output_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        wm_outputs.append(outputs[:, inputs_len:].cpu().detach())

        del outputs, inputs, prompts
        gc.collect()

    gc.collect()
    torch.cuda.empty_cache()
    return wm_outputs


# --------------------------------------------------------------------------- #
#                TRAIN SYNTHID TEXT WATERMARK DETECTOR MODEL                  #
# --------------------------------------------------------------------------- #


def train_detector(
    detector: torch.nn.Module,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    seed: int = 0,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    g_values_val: torch.Tensor = None,
    mask_val: torch.Tensor = None,
    watermarked_val: torch.Tensor = None,
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
) -> Tuple[Dict[str, Any], float]:
    """Trains a Bayesian detector model.

    Args:
      g_values: g-values of shape [num_train, seq_len, watermarking_depth].
      mask: A binary array shape [num_train, seq_len] indicating which g-values
        should be used. g-values with mask value 0 are discarded.
      watermarked: A binary array of shape [num_train] indicating whether the
        example is watermarked (0: unwatermarked, 1: watermarked).
      epochs: Number of epochs to train for.
      learning_rate: Learning rate for optimizer.
      minibatch_size: Minibatch size for training. Note that a minibatch
        requires ~ 32 * minibatch_size * seq_len * watermarked_depth *
        watermarked_depth bits of memory.
      seed: Seed for parameter initialization.
      l2_weight: Weight to apply to L2 regularization for delta parameters.
      shuffle: Whether to shuffle before training.
      g_values_val: Validation g-values of shape [num_val, seq_len,
        watermarking_depth].
      mask_val: Validation mask of shape [num_val, seq_len].
      watermarked_val: Validation watermark labels of shape [num_val].
      verbose: Boolean indicating verbosity of training. If true, the loss will
        be printed. Defaulted to False.
      use_tpr_fpr_for_val: Whether to use TPR@FPR=1% as metric for validation.
        If false, use cross entropy loss.

    Returns:
      Tuple of
        training_history: Training history keyed by epoch number where the
        values are
          dictionaries containing the loss, validation loss, and model
          parameters,
          keyed by
          'loss', 'val_loss', and 'params', respectively.
        min_val_loss: Minimum validation loss achieved during training.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Shuffle the data if required
    if shuffle:
        indices = torch.randperm(len(g_values))
        g_values = g_values[indices]
        mask = mask[indices]
        watermarked = watermarked[indices]

    # Initialize optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    history = {}
    loss = None
    best_val_epoch = None
    min_val_loss = float("inf")

    for epoch in range(epochs):
        losses = []
        detector.train()
        num_batches = len(g_values) // minibatch_size

        for i in range(0, len(g_values), minibatch_size):
            end = i + minibatch_size

            if end > len(g_values):
                break

            loss_batch_weight = l2_weight / num_batches

            optimizer.zero_grad()
            loss = detector(
                g_values=g_values[i:end],
                mask=mask[i:end],
                labels=watermarked[i:end],
                loss_batch_weight=loss_batch_weight,
            )[1]

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)

        val_loss = None
        val_losses = []

        if g_values_val is not None:
            detector.eval()

            if validation_metric == ValidationMetric.TPR_AT_FPR:
                val_loss = update_fn_if_fpr_tpr(
                    detector,
                    g_values_val,
                    mask_val,
                    watermarked_val,
                    minibatch_size=minibatch_size,
                )

            else:
                for i in range(0, len(g_values_val), minibatch_size):
                    end = i + minibatch_size

                    if end > len(g_values_val):
                        break

                    with torch.no_grad():
                        v_loss = detector(
                            g_values=g_values_val[i:end],
                            mask=mask_val[i:end],
                            labels=watermarked_val[i:end],
                            loss_batch_weight=0,
                        )[1]

                    val_losses.append(v_loss.item())

                val_loss = sum(val_losses) / len(val_losses)

        # Store training history
        history[epoch + 1] = {"loss": train_loss, "val_loss": val_loss}

        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch}: loss {loss} (train), {val_loss} (val)")
            else:
                print(f"Epoch {epoch}: loss {loss} (train)")

        if val_loss is not None and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_val_epoch = epoch

    if verbose:
        print(f"Best val Epoch: {best_val_epoch}, min_val_loss: {min_val_loss}")

    return history, min_val_loss


# --------------------------------------------------------------------------- #
#                FULL SYNTHID TEXT DETECTOR TRAINING PIPELINE                 #
# --------------------------------------------------------------------------- #


def train_synthid_detector(
    tokenized_wm_outputs: Union[List[np.ndarray], np.ndarray],
    tokenized_uwm_outputs: Union[List[np.ndarray], np.ndarray],
    logits_processor: SynthIDTextWatermarkLogitsProcessor,
    tokenizer: Any,
    torch_device: torch.device,
    test_size: float = 0.3,
    pos_truncation_length: Optional[int] = 200,
    neg_truncation_length: Optional[int] = 100,
    max_padded_length: int = 2300,
    n_epochs: int = 50,
    learning_rate: float = 2.1e-2,
    l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
):
    """Train and return the best detector given range of hyperparameters.

    In practice, we have found that tuning pos_truncation_length,
    neg_truncation_length, n_epochs, learning_rate and l2_weights can help
    improve the performance of the detector. We reccommend tuning these
    parameters for your data.
    """
    l2_weights = list(l2_weights)

    (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    ) = process_raw_model_outputs(
        logits_processor,
        tokenizer,
        pos_truncation_length,
        neg_truncation_length,
        max_padded_length,
        tokenized_wm_outputs,
        test_size,
        tokenized_uwm_outputs,
        torch_device,
    )

    best_detector = None
    lowest_loss = float("inf")
    val_losses = []

    for l2_weight in l2_weights:
        config = BayesianDetectorConfig(watermarking_depth=len(logits_processor.keys))
        detector = BayesianDetectorModel(config).to(torch_device)

        _, min_val_loss = train_detector(
            detector=detector,
            g_values=train_g_values,
            mask=train_masks,
            watermarked=train_labels,
            g_values_val=cv_g_values,
            mask_val=cv_masks,
            watermarked_val=cv_labels,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            epochs=n_epochs,
            verbose=verbose,
            validation_metric=validation_metric,
        )
        val_losses.append(min_val_loss)

        if min_val_loss < lowest_loss:
            lowest_loss = min_val_loss
            best_detector = detector

    return best_detector, lowest_loss


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
        "--num_negatives",
        type=int,
        default=10000,
        help=("Number of negatives for detector training."),
    )
    parser.add_argument(
        "--pos_batch_size",
        type=int,
        default=32,
        help=("Batch size of watermarked positives while sampling."),
    )
    parser.add_argument(
        "--num_pos_batch",
        type=int,
        default=313,
        help=("Number of positive batches for training."),
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=512,
        help=("Generation length for sampling."),
    )
    parser.add_argument(
        "--save_model_to_hf_hub",
        action="store_true",
        help=(
            "Whether to save the trained model HF hub. "
            "By default it will be a private repo."
        ),
    )
    parser.add_argument(
        "--hf_hub_model_name",
        type=str,
        default=None,
        help=("HF hub model name for loading of saving the model."),
    )
    parser.add_argument(
        "--watermarking_config",
        type=str,
        default=None,
        help=("Path to JSON file defining watermarking config."),
    )

    args = parser.parse_args()
    model_name = args.model_name
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    num_negatives = args.num_negatives
    pos_batch_size = args.pos_batch_size
    num_pos_batch = args.num_pos_batch
    watermarking_config = args.watermarking_config

    if num_pos_batch < 10:
        raise ValueError("--num_pos_batch should be greater than 10.")

    generation_length = args.generation_length
    save_model_to_hf_hub = args.save_model_to_hf_hub
    repo_name = args.hf_hub_model_name

    # -----------------------------------------------------------------------------
    # Set hyperparameters
    NEG_BATCH_SIZE = 32

    # Truncate outputs to this length for training.
    POS_TRUNCATION_LENGTH = 200
    NEG_TRUNCATION_LENGTH = 100

    # Pad trucated outputs to this length for equal shape across all batches.
    MAX_PADDED_LENGTH = 1000

    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f" << * >> Running training on device: {DEVICE}")

    if DEVICE.type not in ("cuda", "tpu"):
        raise ValueError(
            "We have found the training stable on GPU and TPU, we are working on"
            " a fix for CPUs"
        )

    # -----------------------------------------------------------------------------
    # Load watermarking config
    # Check documentation in the paper to understand the impact of these parameters.
    if watermarking_config is not None and os.path.isfile(watermarking_config):
        with open(watermarking_config, "r", encoding="utf-8") as fp:
            WATERMARKING_CONFIG = json.load(fp)
    else:
        WATERMARKING_CONFIG = {
            "ngram_len": 5,
            "keys": [
                654,
                400,
                836,
                123,
                340,
                443,
                597,
                160,
                57,
                29,
                590,
                639,
                13,
                715,
                468,
                990,
                966,
                226,
                324,
                585,
                118,
                504,
                421,
                521,
                129,
                669,
                732,
                225,
                90,
                960,
            ],
            "sampling_table_size": 2**16,
            "sampling_table_seed": 0,
            "context_history_size": 1024,
        }

    watermark_config = SynthIDTextWatermarkingConfig(**WATERMARKING_CONFIG)
    print(f" << * >> Loaded watermarking config: {WATERMARKING_CONFIG}")

    # -----------------------------------------------------------------------------
    # Load generator model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logits_processor = SynthIDTextWatermarkLogitsProcessor(
        **WATERMARKING_CONFIG, device=DEVICE
    )

    # -----------------------------------------------------------------------------
    # Load training dataset
    print(" << * >> Creating training dataset: unwatermarked samples (1/2)")
    tokenized_uwm_outputs = generate_raw_samples(
        num_negatives, NEG_BATCH_SIZE, tokenizer, DEVICE
    )

    print(" << * >> Creating training dataset: watermarked samples (2/2)")
    tokenized_wm_outputs = generate_watermarked_samples(
        model,
        tokenizer,
        watermark_config,
        num_pos_batch,
        pos_batch_size,
        temperature,
        generation_length,
        top_k,
        top_p,
        DEVICE,
    )

    # -----------------------------------------------------------------------------
    # Run training process
    print(" << * >> Starting training process")
    best_detector, lowest_loss = train_synthid_detector(
        tokenized_wm_outputs=tokenized_wm_outputs,
        tokenized_uwm_outputs=tokenized_uwm_outputs,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        torch_device=DEVICE,
        test_size=0.3,
        pos_truncation_length=POS_TRUNCATION_LENGTH,
        neg_truncation_length=NEG_TRUNCATION_LENGTH,
        max_padded_length=MAX_PADDED_LENGTH,
        n_epochs=100,
        learning_rate=3e-3,
        l2_weights=[
            0,
        ],
        verbose=True,
        validation_metric=ValidationMetric.TPR_AT_FPR,
    )

    if best_detector is None:
        print("Detector training process failed.")
        exit(1)

    print(" << * >> Training complete")

    # -----------------------------------------------------------------------------
    # Save trained model to Hugging Face
    best_detector.config.set_detector_information(
        model_name=model_name, watermarking_config=WATERMARKING_CONFIG
    )

    if save_model_to_hf_hub:
        print(" << * >> Saving model to Hugging Face")
        upload_model_to_hf(best_detector, repo_name)
