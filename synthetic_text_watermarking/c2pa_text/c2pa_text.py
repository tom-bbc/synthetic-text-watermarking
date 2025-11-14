# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import time
from typing import Dict, Optional, Tuple, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from encypher.core.payloads import BasicPayload, ManifestPayload
from encypher.core.unicode_metadata import UnicodeMetadata

# -----------------------------------------------------------------
# EmcipherAI C2PA Text Embedding and Verification Process
# -----------------------------------------------------------------


class C2PAText:
    def __init__(
        self,
        public_key_file: str,
        private_key_file: str,
    ) -> None:
        """
        The EncypherAI SDK provides a robust, C2PA-compliant solution for embedding
        provenance and authenticity metadata directly into plain text. This
        self-contained example demonstrates the end-to-end workflow: creating a
        manifest, embedding it, and verifying it.
        """

        # Load your Ed25519 key pair from their pem files
        public_key, private_key = self.load_keypair(public_key_file, private_key_file)
        self.public_key = public_key
        self.private_key = private_key

        # Store public keys and create a provider function
        self.signer_id_manifest = "manifest-signer-001"
        self.public_keys_store = {self.signer_id_manifest: self.public_key}

    @staticmethod
    def load_keypair(
        public_key_file: str, private_key_file: str
    ) -> Tuple[Ed25519PublicKey, Ed25519PrivateKey]:
        """Load an Ed25519 key pair from the local public/private PEM files"""

        # Get bytes data from PEM files
        with open(public_key_file, "rb") as pem_in:
            public_key_data = pem_in.read()

        with open(private_key_file, "rb") as pem_in:
            private_key_data = pem_in.read()

        # Convert bytes into correct key format
        public_key = load_pem_public_key(public_key_data, default_backend())
        private_key = load_pem_private_key(private_key_data, None, default_backend())

        return public_key, private_key

    def public_key_resolver(self, signer_id: str) -> Optional[Ed25519PublicKey]:
        """Function to retrieve public key during signing process."""

        return self.public_keys_store.get(signer_id)

    def sign(self, text: str, manifest: Optional[Dict] = None) -> str:
        """Embed a C2PA-inspired manifest into the text."""

        # Embed a C2PA-inspired manifest
        if manifest is None:
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

        # Embed manifest into text
        encoded_text_manifest = UnicodeMetadata.embed_metadata(
            text=text,
            private_key=self.private_key,
            signer_id=self.signer_id_manifest,
            **manifest,
        )

        return encoded_text_manifest

    def verify(
        self, text: str
    ) -> Tuple[bool, Optional[str], Union[BasicPayload, ManifestPayload, None]]:
        """
        Verification confirms both the signature's authenticity and the text's
        integrity. Any change to the original text will cause verification to fail.
        """

        # Verify the original, unmodified text
        is_valid, signer, payload = UnicodeMetadata.verify_metadata(
            text=text, public_key_resolver=self.public_key_resolver
        )

        return (is_valid, signer, payload)
