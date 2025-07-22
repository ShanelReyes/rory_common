import pytest
from rory.core.security.dataowner_paillier import DataOwner
from rorycommon import Common as RoryCommon
import numpy as np
from concurrent.futures import ProcessPoolExecutor 
import time as T

# @pytest.mark.skip("test phe parallel")
def test_paralell_phe():
    t1  = T.time()
    dataset_size = (10000,10)
    xs = np.random.random(size=dataset_size )
    max_workers = 2
    ppe = ProcessPoolExecutor(max_workers=max_workers)
    sls = [128,192,256]
    print(f"Benchmark Config - SECURITY LEVELS: {sls} DATASET SIZE: {dataset_size} MAX WORKERS: {max_workers}")
    # security_level = 128
    for security_level in sls:
    # do = DataOwner(securitylevel=security_level)
    # do.generate_keys(output_path="/rory/schemes_experiments/keys",filename=f"rory-phe-{security_level}",save=True)
        do = DataOwner.from_keys(path="/rory/schemes_experiments/keys",filename=f"rory-phe-{security_level}")
        res = RoryCommon.segment_and_encrypt_paillier_with_executor(dataowner=do,executor=ppe,key="x",plaintext_matrix=xs,n=len(xs))
        print(f"Security Level: {security_level} - Dataset Size: {dataset_size} - ST = {T.time()-t1}" )

@pytest.mark.skip("test phe parallel")

def test_generate_keys():
    do = DataOwner(securitylevel=256)
    do.generate_keys(output_path="/rory/schemes_experiments/keys",filename="rory-phe-256",save=True)