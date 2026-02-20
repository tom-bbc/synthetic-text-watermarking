# --------------------------------------------------------------------------- #
#                                  IMPORTS                                    #
# --------------------------------------------------------------------------- #

import os
from pathlib import Path
from typing import Tuple

import streamlit as st

from synthetic_text_watermarking.c2pa.c2pa_text import C2PAText
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


# --------------------------------------------------------------------------- #
#                                  WEBAPP                                     #
# --------------------------------------------------------------------------- #


def main():
    # Page config
    # ---------------------------------------------------------------------------
    # Set global variables and global page settings
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

        The C2PA Text library provides a standard wrapper structure that encodes the binary C2PA Manifest Store (JUMBF) into invisible characters that persist through copy-paste operations.
        To generate the valid C2PA JUMBF manifest bytes, the core C2PA library is used. The C2PA Text library then handles the embedding layer (text steganography) and validation.
        """
    )

    # Signing process
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("Sign a text")

    # Link to C2PA test certificate (change in production use cases)
    public_key = (
        Path.cwd()
        / "src/synthetic_text_watermarking/c2pa/certificates/test-cert-es256.pub"
    )
    private_key = (
        Path.cwd()
        / "src/synthetic_text_watermarking/c2pa/certificates/test-key-es256.pem"
    )

    c2pa_processor = C2PAText(
        public_key_file=public_key,
        private_key_file=private_key,
    )

    with st.form(
        "c2pa_sign",
        clear_on_submit=True,
        enter_to_submit=True,
    ):
        input_text = st.text_area("Please enter some text to be signed")
        submitted = st.form_submit_button("Submit")

        if submitted:
            input_text = input_text.strip()
            signed_text = c2pa_processor.sign(input_text)

            print(f"\n >> **Signed text:** \n\n{signed_text}\n")
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
            is_valid, validation_code, manifest = c2pa_processor.verify(input_text)

            if is_valid is True:
                st.write("**Verification result:** \n:green[Valid C2PA manifest found]")

                if manifest:
                    st.space()
                    st.json(manifest)

            else:
                st.write(
                    f"**Verification result:** \n:red[No valid C2PA manifest found] \n\nValidation code: \n:red[{validation_code}]"
                )


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
