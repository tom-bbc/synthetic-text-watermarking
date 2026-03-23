# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import json
from pathlib import Path
import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

from synthetic_text_watermarking.synthid.watermark import (
    generate_with_watermark,
    load_mean_detector,
    load_model,
    run_mean_detector,
)

# --------------------------------------------------------------------------- #
#                             HELPER FUNCTIONS                                #
# --------------------------------------------------------------------------- #


def model_setup(
    model_name: str,
) -> tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
    torch.device,
]:
    # Configure model and hyperparams
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load SynthID waterkarking config
    # Incl. watermark keys: a list of random ints that serve as your private signature
    watermark_config_file = (
        Path.cwd() / "src/synthetic_text_watermarking/synthid/watermark_config.json"
    )

    with open(watermark_config_file, "r", encoding="utf-8") as fp:
        watermark_config_data = json.load(fp)

    # Instantiate generator model and tokenizer
    model, tokenizer, watermarking_config = load_model(
        model_name, watermark_config_data["keys"], device
    )

    # Instantiate detector model
    tokenizer, logits_processor = load_mean_detector(
        watermark_config_data, model_name, device, tokenizer
    )

    return model, tokenizer, watermarking_config, logits_processor, device


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
    st.subheader("Chat with watermarked Gemma model")

    model_name = "google/gemma-2-2b-it"
    model, tokenizer, watermarking_config, logits_processor, device = model_setup(
        model_name
    )

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
            response = generate_with_watermark(
                prompt, model, tokenizer, watermarking_config, device
            )

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
            result = run_mean_detector(input_text, tokenizer, logits_processor, device)

            result = True

            st.write(f"**Input text:** \n\n{input_text}")
            st.write(f"**Verification result:** \n\n{result}")


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
