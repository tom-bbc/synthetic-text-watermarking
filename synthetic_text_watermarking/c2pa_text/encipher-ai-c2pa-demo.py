import hashlib
from encypher.core.keys import generate_ed25519_key_pair
from encypher.core.unicode_metadata import UnicodeMetadata

def run_c2pa_text_demo():
    """Demonstrates embedding and verifying a C2PA manifest in text."""

    # 1. Generate a key pair for signing
    private_key, public_key = generate_ed25519_key_pair()
    signer_id = "com.example.news-outlet"

    # 2. Define the clean text content
    original_text = "The new AI regulations will take effect next quarter, according to sources."

    # 3. Create a C2PA manifest dictionary
    # This includes a hard-binding content hash to protect against tampering.
    c2pa_manifest = {
        "claim_generator": "EncypherAI-SDK/1.1.0",
        "assertions": [
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "author": {
                        "@type": "Organization",
                        "name": "Example News Co."
                    },
                },
            },
            {
                "label": "c2pa.hash.data.v1",
                "data": {
                    "hash": hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
                    "alg": "sha256",
                },
                "kind": "ContentHash",
            },
        ],
    }

    # 4. Embed the manifest into the text
    # The `c2pa` format handles all COSE signing and CBOR encoding internally.
    embedded_text = UnicodeMetadata.embed_metadata(
        text=original_text,
        private_key=private_key,
        signer_id=signer_id,
        c2pa_manifest=c2pa_manifest,
    )

    print(f"Original text length: {len(original_text)}")
    print(f"Embedded text length: {len(embedded_text)}")
    print("--- Text with embedded C2PA manifest ---")
    print(embedded_text)
    print("-----------------------------------------")

    # 5. Verify the embedded manifest
    # The public key resolver function allows the verifier to look up the correct
    # public key based on the signer_id found in the manifest.
    def public_key_resolver(kid):
        if kid == signer_id:
            return public_key
        return None

    is_verified, extracted_id, payload = UnicodeMetadata.verify_metadata(
        text=embedded_text,
        public_key_resolver=public_key_resolver,
    )

    print(f"\nVerification result: {'SUCCESS' if is_verified else 'FAILURE'}")
    print(f"Extracted Signer ID: {extracted_id}")

    if payload:
        print("Extracted and Verified Payload:")
        # The payload contains the original manifest assertions
        for assertion in payload.assertions:
            print(f"- Assertion Label: {assertion.label}")

    # --- Tampering Demo ---
    print("\n--- Demonstrating Tamper Detection ---")
    tampered_text = embedded_text.replace("next quarter", "immediately")

    is_verified_tampered, _, _ = UnicodeMetadata.verify_metadata(
        text=tampered_text, public_key_resolver=public_key_resolver
    )

    print(f"Verification of tampered text: {'SUCCESS' if is_verified_tampered else 'FAILURE'}")
    assert not is_verified_tampered, "Tamper detection failed!"
    print("Tampering was successfully detected, as expected.")

if __name__ == "__main__":
    run_c2pa_text_demo()
