# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import json
import logging
import os
import time
from pathlib import Path

import flask

from synthetic_text_watermarking.c2pa.c2pa_text import C2PAText

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


@app.route("/", methods=["GET"])
def index() -> str:
    contents = flask.render_template(
        "index.html.j2", text=None, result=None, payload=None
    )
    return contents


@app.route("/c2pa_verify", methods=["GET", "POST"])
def c2pa_verify() -> str:
    candidate_text = None
    is_valid = None
    payload = None

    if flask.request.method == "POST":
        candidate_text = flask.request.form.get("text", None)

        if candidate_text is not None:
            public_key_file = app.config["public_key_file"]
            private_key_file = app.config["private_key_file"]

            c2pa_processor = C2PAText(
                public_key_file=public_key_file,
                private_key_file=private_key_file,
            )

            is_valid, signer, payload = c2pa_processor.verify(candidate_text)

            logger.info(f"Input text: {candidate_text}")
            logger.info(f"Result: {is_valid}")

            if payload is not None:
                payload = json.dumps(payload, indent=4)

    contents = flask.render_template(
        "c2pa_verify.html.j2", text=candidate_text, result=is_valid, payload=payload
    )

    return contents


@app.route("/c2pa_sign", methods=["GET", "POST"])
def c2pa_sign() -> str:
    signed_text = None

    if flask.request.method == "POST":
        input_text = flask.request.form.get("text", None)

        if input_text is not None:
            public_key_file = app.config["public_key_file"]
            private_key_file = app.config["private_key_file"]

            c2pa_processor = C2PAText(
                public_key_file=public_key_file,
                private_key_file=private_key_file,
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

    contents = flask.render_template("c2pa_sign.html.j2", text=signed_text)

    return contents


# -----------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------


def main() -> None:
    app.run(host="127.0.0.1", port=4040, debug=True)


if __name__ == "__main__":
    main()
