# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

import os

from cryptography.hazmat.primitives import serialization
from encypher.core.keys import generate_ed25519_key_pair

# -----------------------------------------------------------------
# Key Pair Generation Entrypoint
# -----------------------------------------------------------------


def main(keypair_output_path: str) -> None:
    private_key, public_key = generate_ed25519_key_pair()

    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path = os.path.join(keypair_output_path, "C2PATextPrivateKey.pem")
    public_key_path = os.path.join(keypair_output_path, "C2PATextPublicKey.pem")

    with open(private_key_path, "wb") as pem_out:
        pem_out.write(private_key_pem)

    with open(public_key_path, "wb") as pem_out:
        pem_out.write(public_key_pem)


if __name__ == "__main__":
    key_dir = "/Users/tompo/setup-data"
    main(key_dir)
