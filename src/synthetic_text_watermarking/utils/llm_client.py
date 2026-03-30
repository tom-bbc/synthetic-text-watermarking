# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
import os

from openai import OpenAI
from transformers import AutoTokenizer

# from vllm import SamplingParams
# from lm_wm_tools.watermarks import get_watermark

# --------------------------------------------------------------------------- #
#          CLIENT FOR LLM INTERACTIONS VIA DOCKER MODEL RUNNER (vLLM)         #
# --------------------------------------------------------------------------- #


class LLMClient:
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        llm_name: str | None = None,
        llm_endpoint: str | None = None,
        temperature: float = 1.0,
        max_completion_tokens: int = 100,
    ) -> None:
        if llm_name is None:
            llm_name = os.getenv("LLM_MODEL")

        if llm_endpoint is None:
            llm_endpoint = os.getenv("LLM_URL")

        if llm_name is None or llm_endpoint is None:
            raise ValueError(
                "LLM name or endpoint not provided and "
                "cannot be found in environment variables."
            )

        self.llm_name = llm_name
        self.llm_endpoint = llm_endpoint

        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        self.tokenizer = self.get_tokenizer(llm_name)
        self.client = OpenAI(
            base_url=llm_endpoint,
            api_key="",
        )

        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        self.watermark_config = {
            "watermark_class": "KGW",
            "epsilon": 1.0,
            "vocab_size": self.tokenizer.vocab_size,
            "temperature": self.temperature,
            "rng_device": "cpu",
            "seeding_scheme": "sumhash",
            "context_size": 4,
            "seed": 0,
            "top_k": 50,
            "distribution_name": "binomial",
            "distribution_parameters": json.dumps({"total_count": 1, "probs": 0.5}),
        }

    # ---------------------------------------------------------------------------
    def get_tokenizer(self, model_name: str) -> AutoTokenizer:

        if model_name in ("ai/gemma3:270M", "ai/gemma3-vllm:270M"):
            model_name = "google/gemma-3-4b-it"

        else:
            raise ValueError(
                f"Module currently selected is not supported for tokenization: "
                f"'{model_name}'"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return tokenizer

    # ---------------------------------------------------------------------------
    def models(self) -> dict:
        response = self.client.models.list()
        response_content = response.to_dict()

        return response_content

    # ---------------------------------------------------------------------------
    def generate(self, prompt: str) -> str | None:
        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        stop_conditions = ["<end_of_turn>", "<eos>"]

        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=self.messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop_conditions,
            extra_body={
                "include_stop_str_in_output": False,
                "skip_special_tokens": True,
                "stop": stop_conditions,
            },
        )

        response_content = response.choices[0].message.content

        self.messages.append({"role": "assistant", "content": response_content})

        return response_content

    # ---------------------------------------------------------------------------
    # def generate_with_watermark(self, prompt: str) -> str | None:
    #     self.messages.append(
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     )

    #     response = self.client.chat.completions.create(
    #         model=self.llm_name,
    #         messages=self.messages,
    #         max_completion_tokens=self.max_completion_tokens,
    #         extra_body={
    #             "top_k": self.watermark_config.get("top_k", 50),
    #             "vllm_xargs": self.watermark_config,
    #         },
    #     )

    #     response_content = response.choices[0].message.content
    #     self.messages.append({"role": "assistant", "content": response_content})

    #     return response_content

    # ---------------------------------------------------------------------------
    # def detect_watermark(self, input_text: str) -> dict[str, float]:
    #     # Use your sampling parameters
    #     params = SamplingParams(
    #         temperature=self.watermark_config.get("temperature", 1.0),
    #         top_k=self.watermark_config.get("top_k", -1),
    #         extra_args=self.watermark_config,
    #     )

    #     # Setup watermark detector
    #     watermark = get_watermark(self.watermark_config, params)

    #     # Send the input to the watermark detector
    #     tokenized_input = self.tokenizer.encode(input_text)
    #     scores = watermark.detect(tokenized_input)
    #     scores["pvalue"] = float(scores["pvalue"])

    #     return scores
