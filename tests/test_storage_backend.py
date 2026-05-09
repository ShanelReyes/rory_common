import pytest
import numpy as np
import pandas as pd
from uuid import uuid4
from Pyfhel import PyCtxt
from rorycommon import StorageBuilder, StorageBackend, StorageParams, Algorithm
from rory.core.security.dataowner_paillier import DataOwner as DataOwnerPHE
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC

import os
RORY_KEYS_PATH             = os.environ.get("RORY_KEYS_PATH", "/rory/keys/test2")
RORY_COMMON_CTX_FILENAME   = os.environ.get("RORY_COMMON_CTX_FILENAME", "ctx")
RORY_COMMON_PUBKEY_FILENAME = os.environ.get("RORY_COMMON_PUBKEY_FILENAME", "pubkey")
RORY_COMMON_SECRETKEY_FILENAME = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME", "secretkey")
RORY_COMMON_RELINKEY_FILENAME  = os.environ.get("RORY_COMMON_RELINKEY_FILENAME", "relinkey")
RORY_COMMON_ROTATEKEY_FILENAME = os.environ.get("RORY_COMMON_ROTATEKEY_FILENAME", "rotatekey")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    return np.random.random((4, 4)).astype(np.float64)


@pytest.fixture
def storage_ids():
    iid = uuid4().hex[:6]
    return {"bucket_id": f"test{iid}", "ball_id": f"ball{iid}"}


@pytest.fixture
def paillier_do():
    do = DataOwnerPHE(securitylevel=128)
    do.generate_keys()
    return do


@pytest.fixture
def tmp_npy_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.npy"
    np.save(path, small_matrix)
    return str(path), "npy"


@pytest.fixture
def tmp_csv_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(small_matrix).to_csv(path, header=False, index=False)
    return str(path), "csv"


def ckks_builder(client, ckks):
    """StorageBuilder wired for CKKS with full key config."""
    return StorageBuilder(
        storage_client     = client,
        algorithm          = Algorithm.CKKS,
        ckks               = ckks,
        keys_path          = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        relinkey_filename  = RORY_COMMON_RELINKEY_FILENAME,
        rotatekey_filename = RORY_COMMON_ROTATEKEY_FILENAME,
        decimals           = 2,
        _round             = True,
    ).build()


def liu_builder(client, dataowner):
    return StorageBuilder(storage_client=client, algorithm=Algorithm.LIU, dataowner=dataowner).build()


# ---------------------------------------------------------------------------
# StorageParams — unit tests (no network)
# ---------------------------------------------------------------------------

def test_storage_params_defaults():
    p = StorageParams()
    assert p.backoff_factor == 0.5
    assert p.num_chunks == 2
    assert p.chunk_index == 0
    assert p.chunk_size == "256kb"
    assert p.delay == 1
    assert p.force is True
    assert p.headers == {}
    assert p.http2 is False
    assert p.max_attempts == 5
    assert p.max_parallel_gets == 10
    assert p.timeout == 300


def test_storage_params_custom():
    p = StorageParams(num_chunks=4, timeout=60, chunk_size="128kb")
    assert p.num_chunks == 4
    assert p.timeout == 60
    assert p.chunk_size == "128kb"


# ---------------------------------------------------------------------------
# StorageBuilder — unit tests (no network)
# ---------------------------------------------------------------------------

async def test_builder_with_algorithm(client, ckks):
    backend = (
        StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS, ckks=ckks)
        .with_algorithm(Algorithm.LIU)
        .build()
    )
    assert backend.algorithm == Algorithm.LIU


async def test_builder_with_ckks(client, ckks):
    backend = (
        StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS)
        .with_ckks(ckks)
        .build()
    )
    assert backend.ckks is ckks


async def test_storage_params_applied(client, ckks):
    params = StorageParams(backoff_factor=1.5, num_chunks=4, timeout=60)
    backend = (
        StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS, ckks=ckks)
        .with_storage_params(params)
        .build()
    )
    assert backend.params.backoff_factor == 1.5
    assert backend.params.num_chunks == 4
    assert backend.params.timeout == 60


async def test_builder_defaults_to_storage_params(client, ckks):
    backend = StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS, ckks=ckks).build()
    assert isinstance(backend.params, StorageParams)


async def test_builder_ckks_key_config(client, ckks):
    backend = ckks_builder(client, ckks)
    assert backend.keys_path == RORY_KEYS_PATH
    assert backend.ctx_filename == RORY_COMMON_CTX_FILENAME


# ---------------------------------------------------------------------------
# Default put/get (no segment, no encrypt) — single blob
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_default_ckks(client, ckks, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_ckks(client, ckks, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_default_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


# ---------------------------------------------------------------------------
# Segment only (segment=True, encrypt=False) — chunked, unencrypted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_segment_only_ckks(client, ckks, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_ckks(client, ckks, small_matrix, storage_ids):
    backend:StorageBackend = ckks_builder(client, ckks)
    bucket_id = storage_ids["bucket_id"]
    ball_id = storage_ids["ball_id"]
    res = await backend.put(bucket_id=bucket_id, ball_id=ball_id, data=small_matrix, segment=True)
    assert res.is_ok, res.unwrap_err()
    result = await backend.get(bucket_id=bucket_id, ball_id=ball_id, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


@pytest.mark.asyncio
async def test_put_segment_only_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    await backend.put(**storage_ids, data=small_matrix, segment=True)
    result = await backend.get(**storage_ids, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


# ---------------------------------------------------------------------------
# Segment + encrypt — CKKS and LIU
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_ckks(client, ckks, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_ckks(client, ckks, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, list)
    assert all(isinstance(x, PyCtxt) for x in value.raw_value)


@pytest.mark.asyncio
async def test_put_encrypt_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_liu(client, dataowner, small_matrix, storage_ids):
    backend = liu_builder(client, dataowner)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


# ---------------------------------------------------------------------------
# Pre-processed TList — CKKS List[PyCtxt]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_tlist_ckks(client, ckks, small_matrix, storage_ids):
    dataowner_pqc = DataOwnerPQC(scheme=ckks)
    ciphertexts = dataowner_pqc.ckks_encrypt_matrix_chunk(small_matrix)
    backend = ckks_builder(client, ckks)
    result = await backend.put(**storage_ids, data=ciphertexts)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# put_from_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_from_file_default_ckks(client, ckks, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_ckks(client, ckks, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_default_liu(client, dataowner, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, dataowner)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_liu(client, dataowner, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, dataowner)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_csv_default_ckks(client, ckks, storage_ids, tmp_csv_file):
    path, ext = tmp_csv_file
    backend = ckks_builder(client, ckks)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()
