---
icon: lucide/book-open
---

# API Reference

## StorageBackend

The primary interface for storing and retrieving encrypted data.
Construct it with `StorageBuilder` — never instantiate directly.

### put / get flags

`put` and `get` accept two boolean flags that control segmentation and encryption.
Always use the **same flags** in the `get` call that you used in `put`.

| `encrypt` | `segment` | algorithm | What happens |
|---|---|---|---|
| `False` | `False` | any | Single blob, no encryption |
| `False` | `True` | any | Split into `num_chunks` plaintext chunks |
| `True` | — | CKKS | CKKS-encrypt each chunk (keys loaded from disk per worker) |
| `True` | — | LIU | Liu-encrypt each chunk via a process pool |

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
        - with_liu_params
        - with_algorithm
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

## Algorithm

::: rorycommon.Algorithm

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

The preferred path is `StorageBackend.put` with `Algorithm.LIU`, which uses the
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
