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

    # Information about the LLM
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("LLM Info")

    llm_client = LLMClient()
    available_models = llm_client.models()

    st.markdown(" * **Models:**")
    st.json(available_models)

    st.markdown(f" * **Selected model:** {llm_client.llm_name}")
    st.markdown(f" * **Model endpoint:** {llm_client.llm_endpoint}")

    # Example LLM responses
    # ---------------------------------------------------------------------------

    # st.space()
    # st.subheader("LLM Chat")

    # question = "Please write a short extract about the fall of Rome."
    # st.markdown(f" * **Question:** {question}")

    # answer_raw = llm_client.generate(question)
    # st.markdown(f" * **Answer (raw):** {answer_raw}")

    # if answer_raw is not None:
    #     raw_watermark_likelihood = llm_client.detect_watermark(answer_raw)
    #     st.markdown(" * **Watermark likelihood:**")
    #     st.json(raw_watermark_likelihood)

    # answer_watermarked = llm_client.generate_with_watermark(question)
    # st.markdown(f" * **Answer (with watermark):** {answer_watermarked}")

    # if answer_watermarked is not None:
    #     wmk_watermark_likelihood = llm_client.detect_watermark(answer_watermarked)
    #     st.markdown(" * **Watermark likelihood:**")
    #     st.json(wmk_watermark_likelihood)

    # Chat interface for getting watermarked outputs from LLM
    # ---------------------------------------------------------------------------

    st.space()
    st.subheader("LLM Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        response = llm_client.generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
