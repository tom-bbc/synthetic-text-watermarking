# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json

from openai import OpenAI
from transformers import AutoTokenizer, SentencePieceBackend, TokenizersBackend

# --------------------------------------------------------------------------- #
#    LLM CLIENT FOR CHATTING HOSTED LLM (via vLLM or Docker Model Runner)     #
# --------------------------------------------------------------------------- #


class LLMClient:
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        llm_name: str,
        llm_endpoint: str,
        temperature: float = 1.0,
        max_completion_tokens: int = 100,
    ) -> None:
        self.llm_name = llm_name
        self.llm_endpoint = llm_endpoint

        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.stop_conditions = ["<end_of_turn>", "<eos>"]

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
            "vocab_size": self.tokenizer.vocab_size,  # type: ignore
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
    def get_tokenizer(
        self, model_name: str
    ) -> TokenizersBackend | SentencePieceBackend:

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

        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=self.messages,  # type: ignore
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=self.stop_conditions,
            extra_body={
                "include_stop_str_in_output": False,
                "skip_special_tokens": True,
                "stop": self.stop_conditions,
            },
        )

        response_content = response.choices[0].message.content

        self.messages.append(
            {"role": "assistant", "content": response_content}  # type: ignore
        )

        return response_content
