# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import io
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from c2pa import Builder, C2paSignerInfo, C2paSigningAlg, Reader, Signer
from c2pa_text import (
    ValidationCode,
    embed_manifest,
    extract_manifest,
    validate_manifest,
)

# -----------------------------------------------------------------
# EmcipherAI C2PA Text Embedding and Verification Process
# -----------------------------------------------------------------


class C2PAText:
    """
    This library allows you to embed and extract C2PA manifests in unstructured text
    (UTF-8) using invisible Unicode Variation Selectors.
    """

    # ---------------------------------------------------------------------------
    def __init__(
        self,
        public_key_file: Path | str,
        private_key_file: Path | str,
    ) -> None:
        # Check certificate exists
        if not os.path.isfile(public_key_file):
            raise FileNotFoundError(
                f"Cannot find C2PA certificate file at: {public_key_file}"
            )

        if not os.path.isfile(private_key_file):
            raise FileNotFoundError(
                f"Cannot find C2PA private key file at: {private_key_file}"
            )

        # Load your Ed25519 key pair from their pem files
        print(f" >> Using C2PA certificate public key: {public_key_file}")
        print(f" >> Using C2PA certificate private key: {private_key_file}\n")

        self.public_key = public_key_file
        self.private_key = private_key_file

    # ---------------------------------------------------------------------------
    def encode_manifest(self, input_text: str, manifest: dict):
        # Create a signer from certificate and key files
        with (
            open(self.public_key, "rb") as cert_file,
            open(self.private_key, "rb") as key_file,
        ):
            cert_data = cert_file.read()
            key_data = key_file.read()

            # Create signer info using cert and key info
            signer_info = C2paSignerInfo(
                alg=C2paSigningAlg.ES256,
                sign_cert=cert_data,
                private_key=key_data,
                ta_url="http://timestamp.digicert.com",
            )

            # Create signer using the defined SignerInfo
            signer = Signer.from_info(signer_info)

            # Create builder with manifest to sign the input file
            with Builder(manifest) as builder:
                # print(
                #     f" >> Builder supported MIME types: {builder.get_supported_mime_types()}"
                # )
                input_text = input_text.strip()
                # input_text_bytes = input_text.encode(encoding="ASCII")
                # source_data = io.BytesIO(input_text_bytes)
                text_stream = io.StringIO(input_text)

                builder.set_no_embed()
                manifest_bytes = builder.sign(
                    signer,
                    "application/x-c2pa-manifest-store",
                    text_stream,
                )

        return manifest_bytes

    # ---------------------------------------------------------------------------
    def decode_manifest(
        self, signed_text: str, manifest_bytes: bytes
    ) -> Optional[dict]:
        """Decode C2PA manifest data from extracted manifest bytes."""

        # 1. Convert manifest bytes into a readable byte stream
        manifest_byte_steam = io.BytesIO(manifest_bytes)
        text_stream = io.StringIO(signed_text)
        print(f"\n >> Supported MIME types: {Reader.get_supported_mime_types()}")

        # 2. Create a reader from a format and stream
        with Reader(
            format_or_path="application/x-c2pa-manifest-store",
            stream=text_stream,
            manifest_data=manifest_bytes,
        ) as reader:
            # Print manifest store as JSON, as extracted by the Reader
            print(f" >> Extracted manifest store: \n{reader.json()}")

            # 3. Get the active manifest
            manifest = json.loads(reader.json())

            # active_manifest = manifest["manifests"][manifest["active_manifest"]]

            # if active_manifest:
            #     # get the uri to the manifest's thumbnail and write it to a file
            #     uri = active_manifest["thumbnail"]["identifier"]
            #     with open("thumbnail_v2.jpg", "wb") as f:
            #         reader.resource_to_stream(uri, f)

        return manifest

    # ---------------------------------------------------------------------------
    def sign(self, text: str, manifest: Optional[Dict] = None) -> Optional[str]:
        """Embed a C2PA-inspired manifest into the text."""

        # 1. You have a binary C2PA manifest (JUMBF)
        manifest = {
            "claim_generator": "python_test/0.1",
            "assertions": [
                {
                    "label": "cawg.training-mining",
                    "data": {
                        "entries": {
                            "cawg.ai_inference": {"use": "notAllowed"},
                            "cawg.ai_generative_training": {"use": "notAllowed"},
                        }
                    },
                }
            ],
        }

        manifest_bytes = self.encode_manifest(text, manifest)

        if manifest_bytes is None:
            return None

        # 2. Validate manifest before embedding
        manifest_validation = validate_manifest(manifest_bytes)
        print(f" >> Valid manifest bytes created: {manifest_validation.valid}")

        if not manifest_validation.valid:
            return None

        # 3. Embed it into text
        watermarked_text = embed_manifest(text, manifest_bytes)

        return watermarked_text

    # ---------------------------------------------------------------------------
    def verify(self, text: str) -> Tuple[bool, ValidationCode | str, Optional[dict]]:
        """
        Verification confirms both the signature's authenticity and the text's
        integrity. Any change to the original text will cause verification to fail.
        """

        # 1. Attempt to manifest bytes from candidate text
        extracted_bytes, clean_text = extract_manifest(text)

        if extracted_bytes is None:
            return False, "No manifest found", None

        # 2. Validate extracted manifest
        validation = validate_manifest(extracted_bytes)

        manifest_valid = validation.valid
        manifest_validation_code = validation.primary_code

        # 3. Decode manifest JSON
        mainfest = self.decode_manifest(text, extracted_bytes)

        return manifest_valid, manifest_validation_code, mainfest
