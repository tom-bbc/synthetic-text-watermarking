# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import json
import logging
import os
from pathlib import Path

import flask

from synthetic_text_watermarking.c2pa_text.encipher_ai import C2PAText

# -----------------------------------------------------------------
# Webapp Setup
# -----------------------------------------------------------------

script_dir = Path(__file__).parent.resolve()
static_dir = os.path.join(script_dir, "static")

app = flask.Flask(__name__, static_folder=static_dir)

app.config["static_dir"] = static_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------
# Webapp Routes
# -----------------------------------------------------------------


@app.route("/")
def index() -> str:
    text = "Hello and welcome to BBC Radio 4 evening news."

    c2pa_processor = C2PAText()
    signed_text = c2pa_processor.sign(text)

    is_valid, signer, payload = c2pa_processor.verify(signed_text)

    logger.info(f"Result: {is_valid}")

    contents = flask.render_template(
        "index.html.j2", text=text, result=is_valid, payload=payload
    )
    return contents


# -----------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------


def main() -> None:
    app.run(host="127.0.0.1", port=4040)


if __name__ == "__main__":
    main()
