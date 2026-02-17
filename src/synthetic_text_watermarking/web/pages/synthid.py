# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import streamlit as st

# --------------------------------------------------------------------------- #
#                                  WEBAPP                                     #
# --------------------------------------------------------------------------- #


def main():
    # Page config
    # ---------------------------------------------------------------------------
    st.set_page_config(
        page_title="Synthetic Text Watermarking | BBC R&D",
        page_icon="synthetic_text_watermarking/app/images/rd-logo-favicon.jpeg",
        initial_sidebar_state="collapsed",
    )

    st.header("Synthetic Text Watermarking | BBC R&D", divider="rainbow")

    # Page overview
    # ---------------------------------------------------------------------------
    st.title("SynthID Text")

    st.markdown(
        """
        Here we describe SynthID-Text, a production-ready text watermarking scheme that preserves text quality and enables high detection accuracy, with minimal latency overhead.
        SynthID-Text does not affect LLM training and modifies only the sampling procedure; watermark detection is computationally efficient, without using the underlying LLM.
        To enable watermarking at scale, we develop an algorithm integrating watermarking with speculative sampling, an efficiency technique frequently used in production systems.
        Evaluations across multiple LLMs empirically show that SynthID-Text provides improved detectability over comparable methods, and standard benchmarks and human side-by-side ratings indicate no change in LLM capabilities.
        To demonstrate the feasibility of watermarking in large-scale-production systems, we conducted a live experiment that assessed feedback from nearly 20 million Gemini responses, again confirming the preservation of text quality.
        We hope that the availability of SynthID-Text will facilitate further development of watermarking and responsible use of LLM systems.
        """
    )

    # Chat interface for getting watermarked outputs from LLM
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("Chat with watermarked Gemma model")

    with st.container(
        border=True,
        height="stretch",
    ):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # React to user input
        prompt = st.chat_input("What is up?")

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response to prompt from LLM
            response = f"Echo: {prompt}"

            # Add LLM response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

    # Verify a text as watermarked
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("Verify a text")

    with st.form(
        "c2pa_verify",
        clear_on_submit=True,
        enter_to_submit=True,
    ):
        input_text = st.text_area("Please enter some text to verify")
        submitted = st.form_submit_button("Submit")

        if submitted:
            result = True

            st.write(f"**Input text:** \n\n{input_text}")
            st.write(f"**Verification result:** \n\n{result}")


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
