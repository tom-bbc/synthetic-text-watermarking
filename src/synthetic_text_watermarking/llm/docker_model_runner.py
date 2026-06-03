# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import os

from synthetic_text_watermarking.llm.llm_client import LLMClient

# --------------------------------------------------------------------------- #
#                        SETUP LLM CLIENT VIA DMR                             #
# --------------------------------------------------------------------------- #


def get_dmr_llm_client() -> LLMClient:
    llm_name = os.getenv("DOCKER_MODEL_RUNNER_LLM_NAME")
    llm_endpoint = os.getenv("DOCKER_MODEL_RUNNER_ENDPOINT")

    if llm_name is None or llm_endpoint is None:
        raise ValueError(
            "LLM name or endpoint not provided and "
            "cannot be found in environment variables."
        )

    llm_client = LLMClient(llm_name, llm_endpoint)

    return llm_client
