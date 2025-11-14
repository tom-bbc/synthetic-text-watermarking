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
    public_key_file = "/Users/tompo/setup-data/C2PATextPublicKey.pem"
    private_key_file = "/Users/tompo/setup-data/C2PATextPrivateKey.pem"

    c2pa_processor = C2PAText(
        public_key_file=public_key_file,
        private_key_file=private_key_file,
    )

    with open("output/c2pa_text_test.json", "r", encoding="utf-8") as fp:
        text = json.load(fp).get("encoded_text", "")

    is_valid, signer, payload = c2pa_processor.verify(text)

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
