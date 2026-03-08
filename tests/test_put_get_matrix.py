import pytest
from rorycommon import Common as RoryCommon
import os 
import numpy as np
from option import Some
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC
from uuid import uuid4
from dotenv import load_dotenv

RORY_COMMON_ENV_FILE_PATH = os.environ.get("RORY_COMMON_ENV_FILE_PATH","./.env.test")
if os.path.exists(RORY_COMMON_ENV_FILE_PATH):
    load_dotenv(dotenv_path=RORY_COMMON_ENV_FILE_PATH)


MICTLANX_CLIENT_ID             = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT               = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION           = int(os.environ.get("MICTLANX_API_VERSION","4"))
MICTLANX_MAX_WORKERS           = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_BUCKET_ID             = os.environ.get("MICTLANX_BUCKET_ID","rory")
MICTLANX_OUTPUT_PATH           = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")
MICTLANX_PROTOCOL              = os.environ.get("MICTLANX_PROTOCOL","http")
MICTLANX_URI                   = os.environ.get("MICTLANX_URI",f"mictlanx://mictlanx-router-0@localhost:63666?api_version={MICTLANX_API_VERSION}&protocol={MICTLANX_PROTOCOL}")
MICTLANX_DEBUG                 = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
RORY_COMMON_KEY_PATH           = os.environ.get("RORY_COMMON_KEY_PATH","/rory/keys")
RORY_COMMON_CTX_FILENAME       = os.environ.get("RORY_COMMON_CTX_FILENAME","ctx")
RORY_COMMON_PUBKEY_FILENAME    = os.environ.get("RORY_COMMON_PUBKEY_FILENAME","pubkey")
RORY_COMMON_SECRETKEY_FILENAME = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME","secretkey")
RORY_COMMON_SECURITY_LEVEL     = int(os.environ.get("RORY_COMMON_SECURITY_LEVEL",128))
RORY_COMMON_RECORDS            = int(os.environ.get("RORY_COMMON_RECORDS","10"))
RORY_COMMON_ATTRIBUTES         = int(os.environ.get("RORY_COMMON_ATTRIBUTES","10"))




# dataowner_pqc = DataOwnerPQC(
#     scheme=ckks,
#     securitylevel=128

# )


@pytest.fixture(scope="module")
def ckks():
    _ = Ckks.create_client(
        scheme             = "CKKS",
        decimals           = 2,
        enable_relinearize = True,
        security_level     = 128,
        save               = True,
        output_path        = RORY_COMMON_KEY_PATH
    )
    ckks = Ckks.from_pyfhel(
        path=RORY_COMMON_KEY_PATH,
    )
    return ckks

@pytest.fixture(scope="module") 
def dataowner_pqc(ckks):
    dataowner_pqc = DataOwnerPQC(
        scheme=ckks,
        securitylevel=RORY_COMMON_SECURITY_LEVEL
    )
    return dataowner_pqc

@pytest.fixture(scope="module")
def dataowner():
    dataowner = DataOwner(
        liu_scheme    = Liu(),
        seed          = None,
        securitylevel = RORY_COMMON_SECURITY_LEVEL,
    )
    return dataowner

@pytest.fixture(scope="module")
def client():
    client = AsyncClient(
        uri              = MICTLANX_URI,
        client_id        = MICTLANX_CLIENT_ID,
        capacity_storage = "200mb",
        debug            = MICTLANX_DEBUG,
        eviction_policy  = "LRU",
        max_workers      = MICTLANX_MAX_WORKERS,
        verify           = False
    )
    return client


@pytest.fixture(scope="module")
def matrix():
    return np.random.rand(RORY_COMMON_RECORDS, RORY_COMMON_ATTRIBUTES)

@pytest.mark.asyncio
async def test_get_encrypt_ckks_matrix_chunk(dataowner_pqc, client,matrix):
    # bucket_id = "rory"
    ball_id   = uuid4().hex.replace("-","")
    res = await RoryCommon.encrypt_ckks_and_put_chunk(
        client       = client,
        ball_id      = ball_id,
        bucket_id    = MICTLANX_BUCKET_ID,
        dataowner    = dataowner_pqc,
        index        = 0,
        full_shape   = (RORY_COMMON_RECORDS, RORY_COMMON_ATTRIBUTES),
        max_attempts = 5,
        ndarray      = matrix,
        num_chunks   = 1,
        max_backoff  = 5,
        timeout      = 120

    )
    assert res.is_ok, "Failed to encrypt and put chunk: {}".format(res.unwrap_err())

    res = await RoryCommon.get_pyctxt_chunk(
        client    = client,
        ckks      = dataowner_pqc.scheme,
        ball_id   = ball_id,
        bucket_id = MICTLANX_BUCKET_ID,
        index     = 0
    )
    assert res.is_ok, "Failed to get chunk: {}".format(res.unwrap_err())




@pytest.mark.asyncio
async def test_put_chunks(dataowner, client,matrix):
    key        = uuid4().hex.replace("-","")
    n          = matrix.shape[0]*matrix.shape[1]*dataowner.m
    num_chunks = 2
    emt = RoryCommon.segment_and_encrypt_liu_with_executor(
        executor         = ProcessPoolExecutor(max_workers=num_chunks),
        dataowner        = dataowner,
        key              = key,
        n                = n,
        np_random        = True,
        num_chunks       = num_chunks,
        plaintext_matrix = matrix
    )
    # emt.sort()
    for c in emt:
        print(c.chunk_id,c.checksum,c.to_ndarray())
    # checksum,_ = XoloUtils.sha256_stream(emt.to_generator())
    # checksum1= XoloUtils.sha256(emt.to_bytes())


    put_res = await RoryCommon.put_chunks(
        client    = client,
        key       = key,
        bucket_id = MICTLANX_BUCKET_ID,
        chunks    = emt,
        tags      = {
            "full_shape": str((matrix.shape[0],matrix.shape[1],dataowner.m)),
            "full_dtype": str(matrix.dtype)
        }
    ) 
    assert put_res.is_ok, "Failed to put chunks: {}".format(put_res.unwrap_err())

    get_res = await RoryCommon.get_and_merge_safe(
        client = client,
        key = key,
        bucket_id = MICTLANX_BUCKET_ID
    )
    assert get_res.is_ok, "Failed to get and merge chunks: {}".format(get_res.unwrap_err())
    merged_array = get_res.unwrap()
    assert merged_array.shape == (matrix.shape[0],matrix.shape[1],dataowner.m), "Merged array shape mismatch: expected {}, got {}".format((matrix.shape[0],matrix.shape[1],dataowner.m), merged_array.shape)




# @pytest.mark.skip("")
@pytest.mark.asyncio
async def test_put_get_pqx(ckks, client,matrix):
    key        = uuid4().hex.replace("-","")
    n          = matrix.shape[1]*matrix.shape[1]
    num_chunks = 2

    emt = RoryCommon.segment_and_encrypt_ckks_with_executor(
        executor           = ProcessPoolExecutor(max_workers=num_chunks),
        key                = key,
        plaintext_matrix   = matrix,
        n                  = n,
        num_chunks         = num_chunks,
        _round             = False,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        path               = RORY_COMMON_KEY_PATH,
        decimals           = 2
    ) 
    assert len(emt) == num_chunks

    res = await RoryCommon.delete_and_put_chunks(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        chunks    = emt,
        tags      = {
            "x": "Something_test"
        }
    )
    assert res.is_ok, "Failed to delete and put chunks: {}".format(res.unwrap_err())
    
    x = await RoryCommon.get_pyctxt(
        client    = client,
        key       = key,
        bucket_id = MICTLANX_BUCKET_ID,
        ckks      = ckks
    )
    assert x is not None, "Failed to get pyctxt: {}".format(x)

# @pytest.mark.skip("")
@pytest.mark.asyncio
async def test_full_skmeans_pqc(dataowner_pqc, client,matrix,ckks):
    executor            = ProcessPoolExecutor(max_workers=2)
    init_sm_id          = f"initsmid{uuid4().hex.replace('-','')}"
    num_chunks          = 2
    _round              = False
    decimals            = 2
    dataowner           = dataowner_pqc
    plaintext_matrix    = matrix
    encrypted_matrix_id = uuid4().hex.replace("-","")
    r                   = plaintext_matrix.shape[0]
    a                   = plaintext_matrix.shape[1]
    n                   = a*r
    k                   = 2
    udm_id              = f"udmid{uuid4().hex.replace('-','')}"
    encrypted_shift_matrix_id = f"encryptedshiftmatrixid{uuid4().hex.replace('-','')}"

    encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
        executor           = executor,
        key                = encrypted_matrix_id,
        plaintext_matrix   = plaintext_matrix,
        n                  = n,
        _round             = _round,
        decimals           = decimals,
        path               = RORY_COMMON_KEY_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        num_chunks         = num_chunks,
    )
    put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = encrypted_matrix_id,
        chunks         = encrypted_matrix_chunks,
        tags = {
            "full_shape": str((r,a)),
            "full_dtype":"float64"
        }
    )
    assert put_encrypted_matrix_result.is_ok, "Failed to put encrypted matrix chunks: {}".format(put_encrypted_matrix_result.unwrap_err())

    # ===================================================
    zero_shiftmatrix = np.zeros((k, a))
    n2 = a*k
    encrypted_zero_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
        executor           = executor,
        key                = init_sm_id,
        plaintext_matrix   = zero_shiftmatrix,
        n                  = n2,
        num_chunks         = num_chunks,
        _round             = _round,
        decimals           = decimals,
        path               = RORY_COMMON_KEY_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME
    )
    assert len(encrypted_zero_shiftmatrix_chunks) == num_chunks, "Number of encrypted zero shift matrix chunks mismatch: expected {}, got {}".format(num_chunks, len(encrypted_zero_shiftmatrix_chunks))

 
    put_encrypted_zero_shiftmatrix_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = init_sm_id,
        chunks         = encrypted_zero_shiftmatrix_chunks,
        tags = {
            "full_shape": str((k,a)),
            "full_dtype":"float64"
        }
    )
    assert put_encrypted_zero_shiftmatrix_result.is_ok, "Failed to put encrypted zero shift matrix chunks: {}".format(put_encrypted_zero_shiftmatrix_result.unwrap_err())
    # # ===================================================
    udm            = dataowner.get_U(
        plaintext_matrix = plaintext_matrix,
        algorithm        = Constants.ClusteringAlgorithms.SKMEANS_PQC
    )

    maybe_udm_matrix_chunks = Chunks.from_ndarray(
        ndarray      = udm,
        group_id     = udm_id,
        chunk_prefix = Some(udm_id),
        num_chunks   = num_chunks,
    )


    udm_put_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = udm_id,
        chunks         = maybe_udm_matrix_chunks.unwrap(),
        tags = {
            "full_shape": str(udm.shape),
            "full_dtype": str(udm.dtype)
        }
    )
    assert udm_put_result.is_ok, "Failed to put UDM matrix chunks: {}".format(udm_put_result.unwrap_err())
    # # ===================================================
    init_shiftmatrix = await RoryCommon.get_pyctxt(
            client    = client,
            bucket_id = MICTLANX_BUCKET_ID,
            key       = init_sm_id,
            ckks      = ckks,
    )
    

    skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
    _encryptedMatrix = await RoryCommon.get_pyctxt(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = encrypted_matrix_id,
        ckks      = ckks
    )
    _udm = await RoryCommon.get_and_merge(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = udm_id,
    )



    S1,_Cent_i,_Cent_j, label_vector  = skmeans.run1(
        status= Constants.ClusteringStatus.WORK_IN_PROGRESS, 
        k = k, 
        encryptedMatrix= _encryptedMatrix, 
        Cent_j=init_shiftmatrix,
        num_attributes=plaintext_matrix.shape[1],
        UDM=_udm,
    ).unwrap()

    encrypted_shift_matrix_chunks = RoryCommon.from_pyctxts_to_chunks(key=encrypted_shift_matrix_id, xs = S1,num_chunks=num_chunks).unwrap()
    assert len(encrypted_shift_matrix_chunks) == num_chunks, "Number of encrypted shift matrix chunks mismatch: expected {}, got {}".format(num_chunks, len(encrypted_shift_matrix_chunks))
    z = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = encrypted_shift_matrix_id,
        chunks         = encrypted_shift_matrix_chunks
    )
    assert z.is_ok, "Failed to put encrypted shift matrix chunks: {}".format(z.unwrap_err())
    # await asyncio.sleep(5)
    zz = await RoryCommon.get_pyctxt(
        client=client, 
        bucket_id=MICTLANX_BUCKET_ID,
        key=encrypted_shift_matrix_id,
        ckks=ckks,
        delay=2,
        max_retries=20
    )
    assert zz is not None, "Failed to get encrypted shift matrix: {}".format(zz)
    # assert zz.is_ok, "Failed to get encrypted shift matrix: {}".format(zz.unwrap_err())
