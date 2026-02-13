# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import json
import time

from synthetic_text_watermarking.c2pa.c2pa_text import C2PAText

# -----------------------------------------------------------------
# Demo Entrypoint
# -----------------------------------------------------------------


def main():
    # -----------------------------------------------------------------
    # Key Management
    # -----------------------------------------------------------------

    print(
        "\n========================= CREATING KEY PAIR =========================",
        end="\n\n",
    )

    public_key_file = "/Users/tompo/setup-data/C2PATextPublicKey.pem"
    private_key_file = "/Users/tompo/setup-data/C2PATextPrivateKey.pem"

    c2pa_processor = C2PAText(
        public_key_file=public_key_file,
        private_key_file=private_key_file,
    )

    # -----------------------------------------------------------------
    # Create Text and Manifest
    # -----------------------------------------------------------------

    print(
        "\n======================== TEXT AND MANIFEST =========================",
        end="\n\n",
    )

    # Original text to be signed
    original_text = "Ahead of every Budget, the chancellor submits their plans to the Office for Budget Responsibility (OBR), which then make forecasts on whether the government will spend more money than it raises and whether the economy will grow or shrink."

    # Embed a C2PA-inspired manifest
    # The library automatically calculates the content hash of the original text.
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

    print(f"<< * >> Original text: '{original_text}'")
    print(f"\n<< * >> C2PA manifest: '{c2pa_mainfest}'")

    print(
        "\n======================== SIGNING PROCESS =========================",
        end="\n\n",
    )

    encoded_text_manifest = c2pa_processor.sign(original_text, c2pa_mainfest)

    with open("output/c2pa_text_test.json", "w", encoding="utf-8") as fp:
        json.dump({"encoded_text": encoded_text_manifest}, fp)

    print(f"\n<< * >> Text with embedded manifest: '{encoded_text_manifest}'")

    # -----------------------------------------------------------------
    # Verifying the Manifest
    # -----------------------------------------------------------------

    print(
        "\n======================== VERIFYING SIGNED TEXT =========================",
        end="\n\n",
    )

    # Verify the original, unmodified text
    is_valid, signer, payload = c2pa_processor.verify(encoded_text_manifest)

    print(f"\n<< * >> Verification of original text successful: {is_valid}")
    if is_valid and payload:
        print(f"  - Signer ID: {signer}")
        print(f"  - ManifestPayload: {payload}")

    # -----------------------------------------------------------------
    # Detecting Tampering
    # -----------------------------------------------------------------

    print(
        "\n======================= VERIFYING MANIPULATED TEXT ========================",
        end="\n\n",
    )

    # Attempt to verify tampered text
    # tampered_text = (
    #     "It's " + encoded_text_manifest[encoded_text_manifest.index("an") + 3 :]
    # )

    with open("output/c2pa_text_test.json", "r", encoding="utf-8") as fp:
        tampered_text = json.load(fp).get("encoded_text", "")

    is_tampered_valid, tampered_signer, tampered_payload = c2pa_processor.verify(
        tampered_text
    )

    print(f"<< * >> Manipulated text: '{tampered_text}'")
    print(f"<< * >> Verification of tampered text successful: {is_tampered_valid}")

    if is_tampered_valid and payload:
        print(f"  - Signer ID: {tampered_signer}")
        print(f"  - ManifestPayload: {tampered_payload}", end="\n\n")

    else:
        print(
            "<< * >> Verification failed, indicating the text was tampered with.",
            end="\n\n",
        )
