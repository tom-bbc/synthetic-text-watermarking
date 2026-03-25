# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import os
import json

from openai import OpenAI

# --------------------------------------------------------------------------- #
#          CLIENT FOR LLM INTERACTIONS VIA DOCKER MODEL RUNNER (vLLM)         #
# --------------------------------------------------------------------------- #


class LLMClient:
    def __init__(
        self,
        llm_name: str | None = None,
        llm_endpoint: str | None = None,
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
            "vocab_size": 128256,
            "rng_device": "cpu",
            "seeding_scheme": "sumhash",
            "context_size": 4,
            "seed": 0,
            "top_k": 50,
            "distribution_name": "binomial",
            "distribution_parameters": json.dumps({"total_count": 1, "probs": 0.5}),
        }

    def models(self) -> dict:
        response = self.client.models.list()
        response_content = response.to_dict()

        return response_content

    def generate(self, prompt: str) -> str | None:
        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = self.client.chat.completions.create(
            model=self.llm_name, messages=self.messages
        )

        response_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_content})

        return response_content

    def generate_with_watermark(self, prompt: str) -> str | None:
        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=self.messages,
            extra_body={
                "top_k": 50,
                "vllm_xargs": self.watermark_config,
            },
        )

        response_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_content})

        return response_content
