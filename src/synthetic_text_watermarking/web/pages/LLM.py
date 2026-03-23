# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import streamlit as st
from synthetic_text_watermarking.utils.llm_client import LLMClient

# --------------------------------------------------------------------------- #
#                                  WEBAPP                                     #
# --------------------------------------------------------------------------- #


def main():
    # Page config
    # ---------------------------------------------------------------------------
    page_title = "Synthetic Text Watermarking | BBC R&D"
    page_icon = "src/synthetic_text_watermarking/web/static/images/rd-logo-favicon.jpeg"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        initial_sidebar_state="collapsed",
    )

    st.header("Synthetic Text Watermarking | BBC R&D", divider="rainbow")

    # Page overview
    # ---------------------------------------------------------------------------
    st.title("SynthID Text")

    st.markdown(
        "Here we describe SynthID-Text, a production-ready text watermarking scheme "
        "that preserves text quality and enables high detection accuracy, with minimal "
        "latency overhead. SynthID-Text does not affect LLM training and modifies only "
        "the sampling procedure; watermark detection is computationally efficient, "
        "without using the underlying LLM. To enable watermarking at scale, we develop "
        "an algorithm integrating watermarking with speculative sampling, an "
        "efficiency technique frequently used in production systems. Evaluations "
        "across multiple LLMs empirically show that SynthID-Text provides improved "
        "detectability over comparable methods, and standard benchmarks and human "
        "side-by-side ratings indicate no change in LLM capabilities. To demonstrate "
        "the feasibility of watermarking in large-scale-production systems, we "
        "conducted a live experiment that assessed feedback from nearly 20 million "
        "Gemini responses, again confirming the preservation of text quality. We hope "
        "that the availability of SynthID-Text will facilitate further development of "
        "watermarking and responsible use of LLM systems."
    )

    # Chat interface for getting watermarked outputs from LLM
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("LLM Info")

    llm_client = LLMClient()
    available_models = llm_client.models()

    st.markdown(" * **Models:**")
    st.json(available_models)

    st.markdown(f" * **Selected model:** {llm_client.llm_name}")
    st.markdown(f" * **Model endpoint:** {llm_client.llm_endpoint}")

    st.space()
    st.subheader("LLM Chat")

    question = "Please write 500 words about the fall of Rome."
    answer = llm_client.prompt(question)

    st.markdown(f" * **Question:** {question}")
    st.markdown(f" * **Answer:** {answer}")


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
