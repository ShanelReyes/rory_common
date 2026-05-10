---
icon: lucide/rocket
---

# rory-common

**rory-common** is the storage abstraction layer for the Rory encrypted-analytics platform.
It provides a single `StorageBackend` interface that handles segmentation, homomorphic
encryption, and chunked upload/download — all through [mictlanx](https://jub-ecosystem.github.io/mictlanx-client).

## Architecture

```
StorageBuilder  ──►  StorageBackend  (put / put_from_file / get)
                            │
                         Common      (static helpers: segment, encrypt, serialize, I/O)
                            │
                    mictlanx.AsyncClient   (network layer)
```

`StorageBackend` is the only surface callers should use.
`Common` is an internal helper class — methods are called by the backend automatically.

## Supported encryption schemes

| Scheme | `Scheme` value | Description | Status |
|---|---|---|---|
| CKKS | `Scheme.CKKS` | Approximate HE via Pyfhel — initialized-executor pipeline, fully abstracted | Stable |
| Liu | `Scheme.LIU` | Symmetric additive HE | Stable |
| FDHOPE | `Scheme.FDHOPE` | FDHoPE chunk encryption for caller-computed UDM matrices; reads return merged `ndarray` chunks | Stable |
| Paillier | `Scheme.PAILLIER` | Probabilistic additive HE | **Not implemented yet** |

!!! note "Deprecated Liu helpers"
    The legacy `Common` helpers `segment_and_encrypt_liu` and
    `segment_and_encrypt_liu_with_executor` are deprecated — they emit a `DeprecationWarning`
    and will be removed in rory-common 1.0.0. Use `StorageBackend.put` with `Scheme.LIU`
    instead.

## Quick start

```python
from mictlanx import AsyncClient
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams, LiuParams, FdhopeParams
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.security.dataowner import DataOwner
import numpy as np

ckks   = Ckks.from_pyfhel(_round=True, decimals=2, path="/rory/keys")
client = AsyncClient(uri="mictlanx://...", client_id="my-app")
matrix = np.random.random((64, 64))
```

=== "Fluent builder"

    ```python
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
        .with_ckks(ckks)
        .with_storage_params(StorageParams(num_chunks=4, timeout=300))
        .build()
    )
    ```

=== "Full constructor"

    ```python
    backend = StorageBuilder(
        storage_client = client,
        scheme      = Scheme.CKKS,
        ckks           = ckks,
        ckks_params    = CkksParams(
            keys_path          = "/rory/keys",
            ctx_filename       = "ctx",
            pubkey_filename    = "pubkey",
            secretkey_filename = "secretkey",
            relinkey_filename  = "relinkey",
            rotatekey_filename = "rotatekey",
            decimals           = 2,
            _round             = True,
        ),
    ).build()
    ```

```python
# Upload plaintext
result = await backend.put(bucket_id="rory", ball_id="model_v1", data=matrix)

# Upload encrypted matrix (2-D)
result = await backend.put(bucket_id="rory", ball_id="model_v1_enc", data=matrix, encrypt=True)

# Upload encrypted vector (1-D) — automatically detected from ndim
vector = np.random.random((64,))
result = await backend.put(bucket_id="rory", ball_id="vector_v1", data=vector, encrypt=True)

# Pass a file path directly — extension is inferred, delegates to put_from_file
result = await backend.put(bucket_id="rory", ball_id="model_v2", data="/rory/data/model.npy")
result = await backend.put(bucket_id="rory", ball_id="model_v2_enc", data="/rory/data/model.npy", encrypt=True)

# Overwrite an existing object — delete before upload
result = await backend.put(bucket_id="rory", ball_id="model_v1_enc", data=matrix, encrypt=True, delete=True)

# Download — mirror the same flags used in put
result = await backend.get(bucket_id="rory", ball_id="model_v1_enc", encrypt=True)
ciphertexts = result.unwrap().raw_value   # List[PyCtxt]
```

### FDHOPE put/get

FDHOPE uses the same backend interface, but the caller is responsible for generating
the UDM first. `StorageBackend.put(..., encrypt=True)` only handles segmentation,
FDHOPE encryption, and upload of that precomputed ndarray.

```python
fdhope_backend = (
    StorageBuilder(storage_client=client, scheme=Scheme.FDHOPE)
    .with_fdhope_params(FdhopeParams(scheme="DBSKMEANS", sens=0.2))
    .build()
)

dataowner = DataOwner(...)
udm = dataowner.get_U(
    algorithm="DBSKMEANS",
    plaintext_matrix=matrix,
)

result = await fdhope_backend.put(bucket_id="rory", ball_id="model_fdhope", data=udm, encrypt=True)

# FDHOPE reads return the merged stored chunks as an ndarray.
result = await fdhope_backend.get(bucket_id="rory", ball_id="model_fdhope", encrypt=True)
encrypted_udm = result.unwrap().raw_value   # np.ndarray
```

## Error handling

All operations return `Result[T, Exception]` from the `option` library.

```python
result = await backend.get(bucket_id="rory", ball_id="model_v1")
if result.is_err:
    raise result.unwrap_err()
value = result.unwrap()           # GetResult[np.ndarray]
matrix = value.raw_value          # np.ndarray
```

## Advanced usage

### Forking a backend for a different scheme

`as_builder()` snapshots every field of a running backend (client, params, key filenames,
CKKS context, etc.) into a fresh `StorageBuilder`. Override only what differs with the
fluent `.with_*()` methods, then call `.build()`.

This is the recommended way to run the same workload under multiple schemes without
re-wiring the shared infrastructure:

```python
from rorycommon import StorageBuilder, StorageParams, Scheme

# base CKKS backend
ckks_backend = (
    StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
    .with_ckks(ckks)
    .with_storage_params(StorageParams(num_chunks=4))
    .build()
)

# fork into a Liu backend — client and params are inherited
liu_backend = (
    ckks_backend.as_builder()
    .with_scheme(Scheme.LIU)
    .with_liu_params(LiuParams(security_level=128, decimals=2, _round=True))
    .build()
)

result = await liu_backend.put(bucket_id="rory", ball_id="model_liu", data=matrix, encrypt=True)
```

The same pattern applies to FDHOPE: switch to `Scheme.FDHOPE`, provide
`FdhopeParams`, and pass a caller-computed UDM ndarray to `put(..., encrypt=True)`.
The caller-side `get_U` API uses `algorithm="DBSKMEANS"` (or another caller-side algorithm value), while the backend FDHOPE config uses
`FdhopeParams.scheme`.

## Generating CKKS keys

Keys must exist on disk before any encrypted `put` or `get` call.
Run `scripts/keygen.py` once per key set you need.

| Argument | Default | Description |
|---|---|---|
| `--output-path` | *(required)* | Directory to write key files into |
| `--mode` | `default` | CKKS parameter preset (`default`, `lite_ml`, …) |
| `--security-level` | `128` | Security level in bits — `128`, `192`, or `256` |
| `--decimals` | `5` | Decimal precision preserved after encryption |
| `--round` | off | Enable rounding |
| `--enable-relinearize` | off | Generate relinearization keys |
| `--enable-rotate` | off | Generate rotation keys |

**Example — `lite_ml` keys:**

```bash
python scripts/keygen.py \
  --output-path /rory/keys/lite_ml \
  --mode lite_ml \
  --security-level 128 \
  --decimals 2 \
  --round \
  --enable-relinearize \
  --enable-rotate
```

## Running tests

Integration tests require a running mictlanx instance and pre-generated CKKS keys.

```bash
# Copy and fill in the test environment file
cp .env.test.example .env.test

# Start a local mictlanx cluster
bash deploy_storage.sh

# Generate CKKS keys — see Generating CKKS keys above
python scripts/keygen.py --output-path /rory/keys --mode default

# Run tests
cd tests && pytest test_new_api.py test_storage_backend.py -v
```
