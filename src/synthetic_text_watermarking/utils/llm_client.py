# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import os

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

    def models(self) -> dict:
        response = self.client.models.list()
        response_content = response.to_dict()

        return response_content

    def prompt(self, prompt: str) -> str | None:
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        response_content = response.choices[0].message.content

        return response_content
