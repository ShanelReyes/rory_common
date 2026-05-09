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

| Scheme | `Algorithm` value | Description | Status |
|---|---|---|---|
| CKKS | `Algorithm.CKKS` | Approximate HE via Pyfhel — initialized-executor pipeline, fully abstracted | Stable |
| Liu | `Algorithm.LIU` | Symmetric additive HE | **Not fully implemented** — lacks the initialized-executor abstraction; may crash the platform |
| Paillier | `Algorithm.PAILLIER` | Probabilistic additive HE | Reserved |

!!! warning "LIU stability"
    The LIU backend does not yet have a proper initialized-executor abstraction equivalent to
    CKKS. Calling `put` with `encrypt=True` on a LIU backend can cause crashes in the Rory
    platform. Use CKKS for production workloads until LIU is fully generalized.

    Additionally, the Liu helpers (`segment_and_encrypt_liu[_with_executor]`) return only
    `Chunks` — no `segment_time` or `encrypt_time` is measured. The `PutCiphertextResult`
    produced by a LIU `put` hardcodes `segment_time = 0.0` and is missing `encrypt_time`
    entirely, which causes a `TypeError` at runtime. Aligning Liu with the CKKS timing model
    requires changing the return type from `Chunks` to `Tuple[Chunks, float, float]` — a
    breaking change that must be carefully coordinated across the Rory Platform.

## Quick start

```python
from mictlanx import AsyncClient
from rorycommon import StorageBuilder, StorageParams, Algorithm
from rory.core.security.cryptosystem.pqc.ckks import Ckks
import numpy as np

ckks   = Ckks.from_pyfhel(_round=True, decimals=2, path="/rory/keys")
client = AsyncClient(uri="mictlanx://...", client_id="my-app")
matrix = np.random.random((64, 64))
```

=== "Fluent builder"

    ```python
    backend = (
        StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS)
        .with_ckks(ckks)
        .with_storage_params(StorageParams(num_chunks=4, timeout=300))
        .build()
    )
    ```

=== "Full constructor"

    ```python
    backend = StorageBuilder(
        storage_client     = client,
        algorithm          = Algorithm.CKKS,
        ckks               = ckks,
        keys_path          = "/rory/keys",
        ctx_filename       = "ctx",
        pubkey_filename    = "pubkey",
        secretkey_filename = "secretkey",
        relinkey_filename  = "relinkey",
        rotatekey_filename = "rotatekey",
        decimals           = 2,
        _round             = True,
    ).build()
    ```

```python
# Upload plaintext
result = await backend.put(bucket_id="rory", ball_id="model_v1", data=matrix)

# Upload encrypted
result = await backend.put(bucket_id="rory", ball_id="model_v1_enc", data=matrix, encrypt=True)

# Download — mirror the same flags used in put
result = await backend.get(bucket_id="rory", ball_id="model_v1_enc", encrypt=True)
ciphertexts = result.unwrap().raw_value   # List[PyCtxt]
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

### Forking a backend for a different algorithm

`as_builder()` snapshots every field of a running backend (client, params, key filenames,
CKKS context, etc.) into a fresh `StorageBuilder`. Override only what differs with the
fluent `.with_*()` methods, then call `.build()`.

This is the recommended way to run the same workload under multiple algorithms without
re-wiring the shared infrastructure:

```python
from rorycommon import StorageBuilder, StorageParams, Algorithm

# base CKKS backend
ckks_backend = (
    StorageBuilder(storage_client=client, algorithm=Algorithm.CKKS)
    .with_ckks(ckks)
    .with_storage_params(StorageParams(num_chunks=4))
    .build()
)

# fork into a Liu backend — client and params are inherited
liu_backend = (
    ckks_backend.as_builder()
    .with_algorithm(Algorithm.LIU)
    .with_dataowner(dataowner)
    .build()
)

result = await liu_backend.put(bucket_id="rory", ball_id="model_liu", data=matrix, encrypt=True)
```

!!! warning
    LIU `put` with `encrypt=True` is not yet stable. See [Supported encryption schemes](#supported-encryption-schemes).

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
