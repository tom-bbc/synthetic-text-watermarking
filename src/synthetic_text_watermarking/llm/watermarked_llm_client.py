# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

# import json

# from lm_wm_tools.watermarks import get_watermark
# from openai import OpenAI
# from vllm import SamplingParams
# from synthetic_text_watermarking.llm.llm_client import LLMClient

# --------------------------------------------------------------------------- #
#            LLM CLIENT WITH SYTNHID WATERMARKING AND DETECTION               #
# --------------------------------------------------------------------------- #


# class WatermarkedLLMClient(LLMClient):
#     # ---------------------------------------------------------------------------
#     def __init__(
#         self,
#         llm_name: str,
#         llm_endpoint: str,
#         temperature: float = 1.0,
#         max_completion_tokens: int = 100,
#     ) -> None:
#         super().__init__(llm_name, llm_endpoint, temperature, max_completion_tokens)

#         self.watermark_config = {
#             "watermark_class": "KGW",
#             "epsilon": 1.0,
#             "vocab_size": self.tokenizer.vocab_size,  # type: ignore
#             "temperature": self.temperature,
#             "rng_device": "cpu",
#             "seeding_scheme": "sumhash",
#             "context_size": 4,
#             "seed": 0,
#             "top_k": 50,
#             "distribution_name": "binomial",
#             "distribution_parameters": json.dumps({"total_count": 1, "probs": 0.5}),
#         }

#     # ---------------------------------------------------------------------------
#     def generate_with_watermark(self, prompt: str) -> str | None:
#         self.messages.append(
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         )

#         response = self.client.chat.completions.create(
#             model=self.llm_name,
#             messages=self.messages,  # type: ignore
#             max_completion_tokens=self.max_completion_tokens,
#             extra_body={
#                 "top_k": self.watermark_config.get("top_k", 50),
#                 "vllm_xargs": self.watermark_config,
#             },
#         )

#         response_content = response.choices[0].message.content
#         self.messages.append(
#             {"role": "assistant", "content": response_content}  # type: ignore
#         )

#         return response_content

#     # ---------------------------------------------------------------------------
#     def detect_watermark(self, input_text: str) -> dict[str, float]:
#         # Use your sampling parameters
#         params = SamplingParams(
#             temperature=self.watermark_config.get("temperature", 1.0),
#             top_k=self.watermark_config.get("top_k", -1),
#             extra_args=self.watermark_config,
#         )

#         # Setup watermark detector
#         watermark = get_watermark(self.watermark_config, params)

#         # Send the input to the watermark detector
#         tokenized_input = self.tokenizer.encode(input_text)
#         scores = watermark.detect(tokenized_input)
#         scores["pvalue"] = float(scores["pvalue"])

#         return scores
