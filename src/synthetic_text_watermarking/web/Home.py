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
    st.title("Home")

    st.markdown(
        """
        Large language models (LLMs) have enabled the generation of high-quality synthetic text, often indistinguishable from human-written content, at a scale that can markedly affect the nature of the information ecosystem.
        Watermarking can help identify synthetic text and limit accidental or deliberate misuse, but has not been adopted in production systems owing to stringent quality, detectability and computational efficiency requirements.
        """
    )

    # Navigate through different watermarking demos
    # ---------------------------------------------------------------------------
    st.space()
    st.subheader("Approaches")
    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/C2PA.py", label="C2PA", icon="📰")
    st.page_link("pages/SynthID.py", label="SynthID", icon="🍭")


# --------------------------------------------------------------------------- #
#                                ENTRYPOINT                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
