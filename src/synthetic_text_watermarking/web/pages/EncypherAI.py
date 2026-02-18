# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from encypher.core.payloads import BasicPayload, C2PAPayload, ManifestPayload

from synthetic_text_watermarking.c2pa.encypherai import C2PAText
from synthetic_text_watermarking.c2pa.generate_key_pair import generate_c2pa_cert

# --------------------------------------------------------------------------- #
#                             HELPER FUNCTIONS                                #
# --------------------------------------------------------------------------- #


def get_c2pa_certs() -> Tuple[Path, Path]:
    c2pa_cert_dir = os.getenv("C2PA_CERT_DIR")

    if c2pa_cert_dir is not None:
        cert_path = Path(c2pa_cert_dir)

        public_key_file = cert_path / "C2PATextPublicKey.pem"
        private_key_file = cert_path / "C2PATextPrivateKey.pem"

    else:
        cert_path = Path.home() / ".ssh/"
        cert_path.mkdir(parents=True, exist_ok=True)

        public_key_file = cert_path / "C2PATextPublicKey.pem"
        private_key_file = cert_path / "C2PATextPrivateKey.pem"

        if not os.path.isfile(public_key_file) or not os.path.isfile(private_key_file):
            public_key_file, private_key_file = generate_c2pa_cert(cert_path)

    return public_key_file, private_key_file


def c2pa_sign(input_text: str, public_key: Path, private_key: Path) -> str:
    """Embed C2PA metadata into a given text."""

    c2pa_processor = C2PAText(
        public_key_file=public_key,
        private_key_file=private_key,
    )

    c2pa_mainfest = {
        "metadata_format": "cbor_manifest",  # Use CBOR or jumbf manifest format
        "timestamp": int(time.time()),
        "claim_generator": "EncypherAI README Example v2.3",
        "actions": [
            {
                "action": "c2pa.created",
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "description": "Text was created by an AI model.",
            }
        ],
        "ai_info": {
            "model_id": "gpt-4o-2024-05-13",
            "prompt": "Write a short, important statement.",
        },
    }

    signed_text = c2pa_processor.sign(input_text, c2pa_mainfest)

    return signed_text


def c2pa_verify(
    candidate_text: str, public_key: Path, private_key: Path
) -> Tuple[bool, Optional[BasicPayload | ManifestPayload | C2PAPayload]]:
    """Verify whether a given text contains C2PA metadata."""

    c2pa_processor = C2PAText(
        public_key_file=public_key,
        private_key_file=private_key,
    )

    is_valid, signer, payload = c2pa_processor.verify(candidate_text)

    print(f"Input text: {candidate_text}")
    print(f"Result: {is_valid}")

    return is_valid, payload


# --------------------------------------------------------------------------- #
#                                  WEBAPP                                     #
# --------------------------------------------------------------------------- #


def main():
    # Page config
    # ---------------------------------------------------------------------------
    # Set global variables and global page settings
    public_key_file, private_key_file = get_c2pa_certs()
    page_title = "Synthetic Text Watermarking | BBC R&D"
    page_icon = "src/synthetic_text_watermarking/web/static/images/rd-logo-favicon.jpeg"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        initial_sidebar_state="collapsed",
    )

    # Set header
    st.header("Synthetic Text Watermarking | BBC R&D", divider="rainbow")

    # Page overview
    # ---------------------------------------------------------------------------
    st.title("C2PA Text")

    st.markdown(
        """
        C2PA manifests are typically embedded in binary files (JPEG, PNG, MP4).
        For unstructured text where traditional file-based embedding is not practical, such as content intended for copy-paste operations across different systems, C2PA Manifests may be embedded directly into a Unicode-encoded text stream.
        This method uses a sequence of Unicode Variation Selectors to encode a C2PA Manifest Store in a way that is not visually rendered, ensuring that Content Credentials persist with the content itself across platforms.

        The EncypherAI SDK provides a robust, C2PA-compliant solution for embedding provenance and authenticity metadata directly into plain text.
        This self-contained example demonstrates the end-to-end workflow: creating a manifest, embedding it, and verifying it.
        """
    )

    # Signing process
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("Sign a text")

    with st.form(
        "c2pa_sign",
        clear_on_submit=True,
        enter_to_submit=True,
    ):
        input_text = st.text_area("Please enter some text to be signed")
        submitted = st.form_submit_button("Submit")

        if submitted:
            signed_text = c2pa_sign(input_text, public_key_file, private_key_file)

            st.write(f"**Signed text:** \n\n{signed_text}")

    # Verification process
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
            status, payload = c2pa_verify(input_text, public_key_file, private_key_file)

            if status is True:
                st.write(f"**Verification result:** :green[{status}]")
            else:
                st.write(f"**Verification result:** :red[{status}]")

            if payload is not None:
                st.write("**Payload:**")
                st.json(payload)


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
