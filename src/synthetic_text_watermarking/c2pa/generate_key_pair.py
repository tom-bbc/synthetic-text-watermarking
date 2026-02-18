# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------

from pathlib import Path
from typing import Tuple

from cryptography.hazmat.primitives import serialization
from encypher.core.keys import generate_ed25519_key_pair

# -----------------------------------------------------------------
# Key Pair Generation Entrypoint
# -----------------------------------------------------------------


def generate_c2pa_cert(keypair_output_path: Path | str) -> Tuple[Path, Path]:
    private_key, public_key = generate_ed25519_key_pair()

    # Generate public key
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    if isinstance(keypair_output_path, str):
        keypair_output_path = Path(keypair_output_path)

    public_key_path = keypair_output_path / "C2PATextPublicKey.pem"

    with open(public_key_path, "wb") as pem_out:
        pem_out.write(public_key_pem)

    # Generate private key
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    private_key_path = keypair_output_path / "C2PATextPrivateKey.pem"

    with open(private_key_path, "wb") as pem_out:
        pem_out.write(private_key_pem)

    return public_key_path, private_key_path


if __name__ == "__main__":
    cert_path = Path.home() / ".ssh/"
    cert_path.mkdir(parents=True, exist_ok=True)

    generate_c2pa_cert(cert_path)
