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
        - with_dataowner
        - with_algorithm
        - with_storage_params
        - build

---

## StorageParams

Tuning parameters applied to every put/get call on the backend.
Pass a custom instance to `StorageBuilder.__init__` or `.with_storage_params()`.

::: rorycommon.StorageParams

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

!!! warning "Missing time measurements — migration required"
    `segment_and_encrypt_liu` and `segment_and_encrypt_liu_with_executor` return only
    `Chunks`. Unlike the CKKS helpers (which return `(Chunks, segment_time, encrypt_time)`),
    no `segment_time` or `encrypt_time` is captured.

    As a consequence, a LIU `put` with `encrypt=True` currently hardcodes
    `segment_time = 0.0` and omits `encrypt_time` from `PutCiphertextResult` entirely
    (a `TypeError` at runtime).

    Fixing this requires changing the return type from `Chunks` to
    `Tuple[Chunks, float, float]`. That is a **breaking change** for any Rory Platform
    component that calls these functions directly — a coordinated migration is needed
    before making that change.

::: rorycommon.Common
    options:
      members:
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
