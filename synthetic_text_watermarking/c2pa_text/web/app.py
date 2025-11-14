# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import json
import logging
import os
from pathlib import Path

import flask

from synthetic_text_watermarking.c2pa_text.c2pa_text import C2PAText

# -----------------------------------------------------------------
# Webapp Setup
# -----------------------------------------------------------------

script_dir = Path(__file__).parent.resolve()
static_dir = os.path.join(script_dir, "static")

app = flask.Flask(__name__, static_folder=static_dir)

app.config["static_dir"] = static_dir
app.config["public_key_file"] = "/Users/tompo/setup-data/C2PATextPublicKey.pem"
app.config["private_key_file"] = "/Users/tompo/setup-data/C2PATextPrivateKey.pem"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------
# Webapp Routes
# -----------------------------------------------------------------


@app.route("/")
def index() -> str:
    contents = flask.render_template(
        "index.html.j2", text=None, result=None, payload=None
    )
    return contents


@app.route("/verify", methods=["POST"])
def verify() -> str:
    public_key_file = app.config["public_key_file"]
    private_key_file = app.config["private_key_file"]

    c2pa_processor = C2PAText(
        public_key_file=public_key_file,
        private_key_file=private_key_file,
    )

    text = flask.request.form.get("text", "")
    is_valid, signer, payload = c2pa_processor.verify(text)

    logger.info(f"Input text: {text}")
    logger.info(f"Result: {is_valid}")

    if payload is not None:
        payload = json.dumps(payload, indent=4)

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
