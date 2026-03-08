import hashlib as H
import pytest
from rorycommon import Common as RoryCommon
import os 
import pickle as PK
import numpy as np
from option import Result,Some,NONE
from mictlanx import AsyncClient
from mictlanx.utils import Utils
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from xolo.utils.utils import Utils as XoloUtils
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC



MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory") 
MICTLANX_OUTPUT_PATH         = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")
MICTLANX_PROTOCOL            = os.environ.get("MICTLANX_PROTOCOL","http")
MICTLANX_URI = os.environ.get("MICTLANX_URI",f"mictlanx://mictlanx-router-0@localhost:63666?api_version={MICTLANX_API_VERSION}&protocol={MICTLANX_PROTOCOL}")
MICTLANX_DEBUG  = bool(int(os.environ.get("MICTLANX_DEBUG",0)))

client = AsyncClient(
    uri              = MICTLANX_URI,
    client_id        = MICTLANX_CLIENT_ID,
    capacity_storage = "200mb",
    debug            = MICTLANX_DEBUG,
    eviction_policy  = "LRU",
    max_workers      = MICTLANX_MAX_WORKERS,
    # routers          = list(Utils.routers_from_str(routers_str=MICTLANX_ROUTERS,protocol="http")),
    verify           = False
)
dataowner = DataOwner(
    liu_scheme= Liu(
        _round         = True,
        decimals       = 2,
        secure_random  = False,
        seed           = 1,
        use_np_random  = True,
        security_level = 128
    ),
)

# key = "encryptedskmeansy"
key = "encryptedskmeanspqc1"
bucket_id = "rory"

ckks = Ckks.from_pyfhel(
    path="/rory/keys/128",
)


dataowner_pqc = DataOwnerPQC(
    scheme=ckks,
    securitylevel=128

)

key = "encryptedskmeansy"
bucket_id = "rory"


# @pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_encrypt_ckks_matrix_chunk():
    bucket_id = "rory"
    ball_id   = "x"
    res = await RoryCommon.encrypt_ckks_and_put_chunk(
        client       = client,
        ball_id      = ball_id,
        bucket_id    = bucket_id,
        dataowner    = dataowner_pqc,
        index        = 0,
        full_shape   = (100,10),
        max_attempts = 5,
        ndarray      = np.random.rand(100,10),
        num_chunks   = 1,
        max_backoff  = 5,
        timeout      = 120

    )
    assert res.is_ok, "Failed to encrypt and put chunk: {}".format(res.unwrap_err())

    res = await RoryCommon.get_pyctxt_chunk(
        client    = client,
        ckks      = dataowner_pqc.scheme,
        ball_id   = ball_id,
        bucket_id = bucket_id,
        index     = 0
    )
    assert res.is_ok, "Failed to get chunk: {}".format(res.unwrap_err())




@pytest.mark.asyncio
async def test_put_chunks():

    test_numpy_array = np.random.rand(100,10)
    np.save("/rory/source/test_array.npy", test_numpy_array)

    pmt_result = await RoryCommon.read_numpy_from(
        path="/rory/source/test_array.npy",
        extension="npy"
    )
    assert pmt_result.is_ok, "Failed to read numpy array: {}".format(pmt_result.unwrap_err())
    pmt = pmt_result.unwrap()

    n = pmt.shape[0]*pmt.shape[1]*dataowner.m
    num_chunks = 2
    emt = RoryCommon.segment_and_encrypt_liu_with_executor(
        executor         = ProcessPoolExecutor(max_workers=num_chunks),
        dataowner        = dataowner,
        key              = key,
        n                = n,
        np_random        = True,
        num_chunks       = num_chunks,
        plaintext_matrix = pmt
    )
    # emt.sort()
    for c in emt:
        print(c.chunk_id,c.checksum,c.to_ndarray())
    # checksum,_ = XoloUtils.sha256_stream(emt.to_generator())
    # checksum1= XoloUtils.sha256(emt.to_bytes())


    put_res = await RoryCommon.put_chunks(
        client=client,
        key=key,
        bucket_id=bucket_id,
        chunks=emt,
        tags={
            "full_shape":str((pmt.shape[0],pmt.shape[1],dataowner.m)),
            "full_dtype":str(pmt.dtype)
        }
    ) 
    assert put_res.is_ok, "Failed to put chunks: {}".format(put_res.unwrap_err())

    get_res = await RoryCommon.get_and_merge_safe(
        client = client,
        key = key,
        bucket_id = bucket_id
    )
    assert get_res.is_ok, "Failed to get and merge chunks: {}".format(get_res.unwrap_err())
    merged_array = get_res.unwrap()
    assert merged_array.shape == (pmt.shape[0],pmt.shape[1],dataowner.m), "Merged array shape mismatch: expected {}, got {}".format((pmt.shape[0],pmt.shape[1],dataowner.m), merged_array.shape)



@pytest.mark.asyncio
async def test_put_pqx():
    bucket_id = "rory"
    key       = "encryptedput_pqx"
    test_numpy_array = np.random.rand(100,10)
    np.save("/rory/source/test_array.npy", test_numpy_array)

    pmt_result = await RoryCommon.read_numpy_from(
        path="/rory/source/test_array.npy",
        extension="npy"
    )
    assert pmt_result.is_ok, "Failed to read numpy array: {}".format(pmt_result.unwrap_err())
    pmt        = pmt_result.unwrap()
    n          = pmt.shape[1]*pmt.shape[1]
    num_chunks = 2

    emt = RoryCommon.segment_and_encrypt_ckks_with_executor(
        executor           = ProcessPoolExecutor(max_workers=num_chunks),
        key                = key,
        plaintext_matrix   = pmt,
        n                  = n,
        num_chunks         = num_chunks,
        _round             = False,
        ctx_filename       = "ctx",
        pubkey_filename    = "pubkey",
        secretkey_filename = "secretkey",
        path               = "/rory/keys/128",
        decimals           = 2
    ) 

    res = await RoryCommon.delete_and_put_chunks(
        client    = client,
        bucket_id = bucket_id,
        key       = key,
        chunks    = emt,
        tags      = {
            "x": "Something_test"
        }
    )
    
    assert res.is_ok, "Failed to delete and put chunks: {}".format(res.unwrap_err())

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_pqx():
    # key = "encryptedskmeansy"
    # bucket_id = "rory"
    ckks = Ckks.from_pyfhel(
        path="/rory/keys",
    )
    print(ckks.he_object)
    # print(ckks.he_object)
    key ="xxx"
    x = await RoryCommon.get_pyctxt(
        client = client, 
        key = key, 
        bucket_id = bucket_id,
        ckks=ckks
    )
    print(x,len(x))
    print("="*50)
    maybe_chs = RoryCommon.from_pyctxts_to_chunks(key=key,xs=x)
    assert maybe_chs.is_some
    chs = maybe_chs.unwrap()
    xx = RoryCommon.from_chunks_to_pyctxts_list(ckks = ckks, chunks=chs)
    print(xx,len(xx))
    tmp = np.array(ckks.decrypt_list(xs = xx, take=5)).reshape((10,5))
    print(tmp)
    # for e in xx:
    #     # tmp = ckks.he_object.decrypt(e)[:5
    #     tmp = ckks.decrypt_list()
    #     print(tmp)
    # xxx = ckks.decrypt_list(xs = xx)
    # xxx = ckks.decryptVe2ctor(np.array(xx))
    # print(xxx,len(xxx))
    # for c in chs:
    #     x = c.to_list().unwrap()
    #     xx = RoryCommon.bytes_to_pyctxt_list(ckks=ckks, serialized_ctxt_bytes=x)

    
    # xx = dataowner.liu_scheme.decryptMatrix(x)
    # print(xx)
    
    # print(x)
    # <ckks Pyfhel obj at 0x7fa556cd1160, [pk:Y, sk:Y, rtk:-, rlk:-, contx(n=16384, t=0, sec=128, qi=[60, 40, 40, 40, 60], scale=1099511627776.0, )]>


@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_full_skmeans_pqc():
    executor = ProcessPoolExecutor(max_workers=2)
    init_sm_id = "initsmid"
    num_chunks = 2
    _round = False
    decimals = 2
    path = "/rory/keys"
    ctx_filename = "ctx"
    pubkey_filename = "pubkey"
    secretkey_filename = "secretkey"
    ckks = Ckks.from_pyfhel(
        path="/rory/keys",
    )
    dataowner = DataOwnerPQC(scheme = ckks)  ##

    BUCKET_ID = "rory"

    encrypted_matrix_id = "encryptedmatrixid"
    # plaintext_matrix = None 
    extension = "npy"
    plaintext_matrix_path = f"/rory/source/clusteringc0r10a5k20.{extension}"
    plaintext_matrix  = (await RoryCommon.read_numpy_from(
        path      = plaintext_matrix_path,
        extension = extension,
    )).unwrap()
    r = plaintext_matrix.shape[0]
    a = plaintext_matrix.shape[1]
    n = a*r
    k = 2

    print("*"*50)
    encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
        executor           = executor,
        key                = encrypted_matrix_id,
        plaintext_matrix   = plaintext_matrix,
        n                  = n,
        _round             = _round,
        decimals           = decimals,
        path               = path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename,
        num_chunks         = num_chunks,
    )
    put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = BUCKET_ID,
        key            = encrypted_matrix_id,
        chunks         = encrypted_matrix_chunks,
        tags = {
            "full_shape": str((r,a)),
            "full_dtype":"float64"
        }
    )
    print("PUT_EM", put_encrypted_matrix_result)

    # ===================================================
    zero_shiftmatrix = np.zeros((k, a))
    n2 = a*k
    print("___"*20)
    encrypted_zero_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
        executor           = executor,
        key                = init_sm_id,
        plaintext_matrix   = zero_shiftmatrix,
        n                  = n2,
        num_chunks         = num_chunks,
        _round             = _round,
        decimals           = decimals,
        path               = path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename
    )

 
    put_encrypted_zero_shiftmatrix_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = BUCKET_ID,
        key            = init_sm_id,
        chunks         = encrypted_zero_shiftmatrix_chunks,
        tags = {
            "full_shape": str((k,a)),
            "full_dtype":"float64"
        }
    )
    print("PUT_SM", put_encrypted_zero_shiftmatrix_result)
    print("__"*20)
    # # ===================================================
    udm            = dataowner.get_U(
        plaintext_matrix = plaintext_matrix,
        algorithm        = Constants.ClusteringAlgorithms.SKMEANS_PQC
    )
    udm_id="udmid"

    maybe_udm_matrix_chunks = Chunks.from_ndarray(
        ndarray      = udm,
        group_id     = udm_id,
        chunk_prefix = Some(udm_id),
        num_chunks   = num_chunks,
    )


    udm_put_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = BUCKET_ID,
        key            = udm_id,
        chunks         = maybe_udm_matrix_chunks.unwrap(),
        tags = {
            "full_shape": str(udm.shape),
            "full_dtype": str(udm.dtype)
        }
    )
    print("PUT_UDM", udm_put_result)
    print("__"*20)
    # # ===================================================
    init_shiftmatrix = await RoryCommon.get_pyctxt(
            client = client, 
            bucket_id=BUCKET_ID, 
            key=init_sm_id, 
            ckks = ckks,
    )
    print(init_shiftmatrix)
    print("__"*20)
    skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
    _encryptedMatrix = await RoryCommon.get_pyctxt(
        client    = client,
        bucket_id = BUCKET_ID,
        key       = encrypted_matrix_id,
        ckks      = ckks
    )
    print(_encryptedMatrix)
    print("__"*20)
    _udm = await RoryCommon.get_and_merge(
        client    = client,
        bucket_id = BUCKET_ID,
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

    print("S1",S1)
    print("Ci",_Cent_i)
    print("Cj",_Cent_j)
    cent_i_id = "centiid"
    cent_j_id = "centjid"
    encrypted_shift_matrix_id = "encryptedshiftmatrixid"
    # cent_i_chunks                 = RoryCommon.from_pyctxts_to_chunks(key=cent_i_id,xs= _Cent_i,num_chunks=num_chunks).unwrap()
    # cent_j_chunks                 = RoryCommon.from_pyctxts_to_chunks(key=cent_j_id,xs= _Cent_j,num_chunks=num_chunks).unwrap()
    encrypted_shift_matrix_chunks = RoryCommon.from_pyctxts_to_chunks(key=encrypted_shift_matrix_id, xs = S1,num_chunks=num_chunks).unwrap()
    
    z = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = BUCKET_ID,
        key            = encrypted_shift_matrix_id,
        chunks         = encrypted_shift_matrix_chunks
    )
    print(z)
    # await asyncio.sleep(5)
    zz = await RoryCommon.get_pyctxt(
        client=client, 
        bucket_id=BUCKET_ID,
        key=encrypted_shift_matrix_id,
        ckks=ckks,
        delay=2,
        max_retries=20
    )
    print(zz)

    # print(skmeans)
    # status = 0
    # encryptedMatrix = None
    # udm = None
    # __Cent_j = None

    # num_attrs = 0
    # run1_result:Result[Tuple[npt.NDArray, List[PyCtxt], List[PyCtxt], List[int]],Exception] = skmeans.run1( # The first part of the skmeans is done
    #         status          = status,
    #         k               = k,
    #         encryptedMatrix = encryptedMatrix, 
    #         UDM             = udm,
    #         Cent_j          = __Cent_j,
    #         num_attributes  = num_attrs
    # )

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_encrypted_shift_matrix():
    encrypted_shift_matrix_id = "encryptedshiftmatrixid"
    BUCKET_ID = "rory"
    
    ckks = Ckks.from_pyfhel(
        path="/rory/keys",
    )

    zz = await RoryCommon.get_pyctxt(
        client=client, 
        bucket_id=BUCKET_ID,
        key=encrypted_shift_matrix_id,
        ckks=ckks,
        delay=2,
        max_retries=20,
        force=True
    )
    print(zz)


@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_getx():
    # key = "encryptedsknnpqc1aa"
    key = "distancessknn1pqc1aa"
    BUCKET_ID = "rory"
    
    ckks = Ckks.from_pyfhel(
        path="/rory/keys",
    )
    encrypted_model = await RoryCommon.get_pyctxt_matrix(client=client, ckks= ckks, bucket_id=BUCKET_ID,key=key)
    # encrypted_records = await RoryCommon.get_pyctxt_matrix(client=client, ckks = ckks, bucket_id=bucket_id, key="encryptedsknnpqc1aa")
    print(encrypted_model.shape)
    # res = RoryCommon.from_pyctxt_matrix_to_chunks(key=key, xs= encrypted_model).unwrap()
    # res1 = RoryCommon.
    # for x in res:
    #     xx = PK.loads(x.data)
    #     xxx = RoryCommon.from_bytes_to_pyctxt_matrix(ckks= ckks, x = x.data)
    #     print(xxx)
    # print(res)
    # res = SKNNPQC.calculate_distances(
    #     model= encrypted_model,
    #     dataset=encrypted_records,
    #     dataset_shape=encrypted_records.shape, 
    #     model_shape=encrypted_model.shape
    # )
    # print("RESS",res)
    # print(res.shape)
    # res = client.get_chunks(bucket_id=BUCKET_ID, key=key)
    # async for (m,c) in res:
    #     x = RoryCommon.from_bytes_to_pyctxt_matrix(ckks= ckks, x = c)
        
        # print(c)
 

