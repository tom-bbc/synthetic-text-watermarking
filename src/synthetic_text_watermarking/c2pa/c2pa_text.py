# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from c2pa import Builder, C2paSignerInfo, C2paSigningAlg, Reader, Signer
from c2pa_text import embed_manifest, extract_manifest, validate_manifest

from synthetic_text_watermarking.c2pa.generate_key_pair import generate_c2pa_cert

# -----------------------------------------------------------------
# EmcipherAI C2PA Text Embedding and Verification Process
# -----------------------------------------------------------------


class C2PAText:
    """
    This library allows you to embed and extract C2PA manifests in unstructured text
    (UTF-8) using invisible Unicode Variation Selectors.
    """

    def __init__(
        self,
        public_key_file: Optional[Path],
        private_key_file: Optional[Path],
    ) -> None:
        # Check certificate exists, and if not create one
        if (
            public_key_file is None
            or private_key_file is None
            or not os.path.isfile(public_key_file)
            or not os.path.isfile(private_key_file)
        ):
            cert_path = Path.home() / ".ssh/"
            cert_path.mkdir(parents=True, exist_ok=True)

            public_key_file = cert_path / "C2PATextPublicKey.pem"
            private_key_file = cert_path / "C2PATextPrivateKey.pem"

            public_key_file, private_key_file = generate_c2pa_cert(cert_path)

        # Load your Ed25519 key pair from their pem files
        print(f"Using C2PA certificate public key: {public_key_file}")
        print(f"Using C2PA certificate private key: {private_key_file}")

        self.public_key = public_key_file
        self.private_key = private_key_file

    def generate_manifest(self, input_text: str, manifest: dict):
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
                # source_data = input_text.encode()
                source_data = io.BytesIO(b"JournalDev Python: \x00\x01")
                manifest_bytes = builder.sign(
                    signer,
                    "application/x-c2pa-manifest-store",
                    source_data,
                )

        return manifest_bytes

    def sign(self, text: str, manifest: Optional[Dict] = None) -> Optional[str]:
        """Embed a C2PA-inspired manifest into the text."""

        # 1. You have a binary C2PA manifest (JUMBF)
        manifest = {
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

        manifest_bytes = self.generate_manifest(text, manifest)

        if manifest_bytes is None:
            return None

        # 2. Validate manifest before embedding
        result = validate_manifest(manifest_bytes)
        print(result)

        # 3. Embed it into text
        if result.valid:
            watermarked_text = embed_manifest(text, manifest_bytes)
        else:
            watermarked_text = None

        print(result)
        print(watermarked_text)

        return watermarked_text

    def verify(self, text: str) -> bool:
        """
        Verification confirms both the signature's authenticity and the text's
        integrity. Any change to the original text will cause verification to fail.
        """

        # 1. Attempt to manifest bytes from candidate text
        extracted_bytes, clean_text = extract_manifest(text)

        # 2. Validate extracted data
        if extracted_bytes is not None:
            result = validate_manifest(extracted_bytes)
            manifest_validated = result.valid

        else:
            manifest_validated = False

        return manifest_validated
