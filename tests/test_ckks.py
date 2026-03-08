import pytest
from rory.core.security.cryptosystem.pqc.ckks import Ckks


@pytest.mark.skip(reason="This test is for development purposes only and should not be run in CI.")
def test_generate_keys():
    sl = 256
    x = Ckks.create_client(
        enable_relinearize = True,
        output_path        = f"/rory/worker/keys/{sl}",
        save               = True,
        scheme             = "CKKS",
        security_level     = sl
    )
    print(x)