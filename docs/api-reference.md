---
icon: lucide/book-open
---

# API Reference

## StorageBackend

The primary interface for storing and retrieving encrypted data.
Construct it with `StorageBuilder` — never instantiate directly.

### put / get flags

`put` and `get` accept boolean flags that control segmentation, encryption, and pre-upload deletion.
Always use the **same `encrypt` and `segment` flags** in the `get` call that you used in `put`.

| `data` type | `encrypt` | `segment` | `data.ndim` | scheme | What happens |
|---|---|---|---|---|---|
| `str` (file path) | any | any | — | any | Extension derived from path suffix; delegates to `put_from_file` |
| `ndarray` | `False` | `False` | any | any | Single blob, no encryption |
| `ndarray` | `False` | `True` | any | any | Split into `num_chunks` plaintext chunks |
| `ndarray` | `True` | — | 1 | CKKS | Encrypt the whole vector as one ciphertext chunk |
| `ndarray` | `True` | — | ≥2 | CKKS | CKKS-encrypt each chunk (keys loaded from disk per worker) |
| `ndarray` | `True` | — | any | LIU | Liu-encrypt each chunk via an initialized process pool |
| `ndarray` | `True` | — | any | FDHOPE | Treat input as caller-computed UDM, FDHoPE-encrypt each chunk, then upload |

### FDHOPE contract

FDHOPE follows the same `StorageBuilder`/`StorageBackend` public pattern as CKKS and LIU,
with one important boundary: the backend does **not** compute `get_U` for you.
Callers must build the UDM first, then pass that ndarray to `put(..., encrypt=True)`
on a backend configured with `Scheme.FDHOPE`.
On the caller side, `get_U` uses its own `algorithm=...` parameter; that is separate
from `FdhopeParams.scheme`, which configures the backend's FDHOPE chunk-encryption step.

For reads, `get(..., encrypt=True)` on an FDHOPE backend uses the generic
`get_and_merge(...)` path and returns a merged `ndarray`.
It does **not** run a separate FDHOPE reconstruction or CKKS-style ciphertext loader.

#### `delete` flag

Pass `delete=True` to `put` or `put_from_file` to remove any existing object at `ball_id`
before uploading. Safe when the key does not yet exist — the delete step completes
immediately with zero deletions and the upload proceeds normally.

::: rorycommon.StorageBackend
    options:
      members:
        - put
        - put_from_file
        - get
        - as_builder

---

## StorageBuilder

Fluent builder — chain `.with_*()` calls then `.build()`.

::: rorycommon.StorageBuilder
    options:
      members:
        - __init__
        - with_ckks
        - with_ckks_params
        - with_fdhope_params
        - with_liu_params
        - with_scheme
        - with_storage_params
        - build

---

## StorageParams

Tuning parameters applied to every put/get call on the backend.
Pass a custom instance to `StorageBuilder.__init__` or `.with_storage_params()`.

::: rorycommon.StorageParams

---

## CkksParams

CKKS key-file locations and encoding configuration.
Required for `StorageBackend.put` with `encrypt=True` on a CKKS backend.

::: rorycommon.CkksParams

---

## LiuParams

Liu-scheme construction parameters.
Required for `StorageBackend.put` with `encrypt=True` on a LIU backend.

::: rorycommon.LiuParams

---

## FdhopeParams

FDHOPE scheme construction parameters.
Required for `StorageBackend.put` with `encrypt=True` on an FDHOPE backend.

::: rorycommon.FdhopeParams

---

## Scheme

::: rorycommon.Scheme

---

## Common — internal helpers

!!! note
    `Common` is not part of the public API. `StorageBackend` calls these methods
    internally. They are documented here for contributors and advanced users.

### Plaintext

::: rorycommon.Common
    options:
      members:
        - from_matrix_to_cloud_storage
        - from_matrix_on_disk_to_cloud_storage
        - from_cloud_storage_to_matrix

### CKKS

::: rorycommon.Common
    options:
      members:
        - from_matrix_to_cloud_storage_ckks
        - from_matrix_on_disk_to_cloud_storage_ckks
        - from_vector_to_cloud_storage_ckks
        - from_vector_on_disk_to_cloud_storage_ckks

### Liu

The preferred path is `StorageBackend.put` with `Scheme.LIU`, which uses the
initialized-executor pipeline below. The `segment_and_encrypt_liu` and
`segment_and_encrypt_liu_with_executor` helpers are deprecated and will be removed in
**rory-common 1.0.0** — they emit a `DeprecationWarning` at runtime.

::: rorycommon.Common
    options:
      members:
        - init_liu_worker_context
        - encrypt_chunk_liu_with_initialized_executor
        - segment_and_encrypt_liu_with_initialized_executor_timed
        - segment_and_encrypt_liu_timed
        - segment_and_encrypt_liu_with_executor_timed
        - segment_and_encrypt_liu
        - segment_and_encrypt_liu_with_executor

### FDHOPE

The preferred path is `StorageBackend.put` with `Scheme.FDHOPE`, using
caller-computed UDM input plus the initialized-executor FDHOPE pipeline below.
`StorageBackend.get(..., encrypt=True)` reads FDHOPE data back through the generic
`get_and_merge(...)` path.

::: rorycommon.Common
    options:
      members:
        - init_fdhope_worker_context
        - encrypt_chunk_fdhope_with_initialized_executor
        - segment_and_encrypt_fdhope_with_initialized_executor_timed

### Retrieval

::: rorycommon.Common
    options:
      members:
        - get_matrix_or_error
        - get_and_merge
        - get_pyctxt
        - get_by_chunk_index

### Low-level I/O

::: rorycommon.Common
    options:
      members:
        - put_ndarray
        - put_chunks
        - from_pyctxts_to_chunks

### Serialization

::: rorycommon.Common
    options:
      members:
        - serialize_matrix_with_pickle
        - deserialize_matrix_with_pickle
        - from_pyctxt_list_to_bytes
        - from_bytes_to_pyctxt_list
