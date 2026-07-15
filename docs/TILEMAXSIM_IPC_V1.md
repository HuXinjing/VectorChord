# VectorChord TileMaxSim IPC Protocol v1

## Scope

This protocol connects a PostgreSQL VectorChord backend to a local TileMaxSim
GPU sidecar. It carries full query/page tensors for Phase 3A. It is not a
network authentication protocol and must be exposed only through an
operations-controlled Unix socket.

All integers and floating-point bit patterns are little-endian. The sidecar
must read and write exact frame lengths. Partial results are forbidden.

## Common Frame Header

Every frame starts with 24 bytes:

| Offset | Size | Type | Meaning |
| --- | ---: | --- | --- |
| 0 | 4 | bytes | ASCII magic `VCTM` |
| 4 | 2 | `u16` | protocol version, currently `1` |
| 6 | 2 | `u16` | message kind: `1` request, `2` response |
| 8 | 8 | `u64` | request ID, echoed unchanged by the response |
| 16 | 8 | `u64` | body length, excluding the 24-byte header |

Unknown versions, message kinds, request IDs, and inconsistent lengths are
fatal protocol errors.

## Rerank Request Body

The fixed portion is 16 bytes:

| Offset in body | Size | Type | Meaning |
| --- | ---: | --- | --- |
| 0 | 4 | `u32` | tensor dimension |
| 4 | 4 | `u32` | query row/token count |
| 8 | 4 | `u32` | candidate page count |
| 12 | 1 | `u8` | dtype: `1` f32, `2` IEEE f16 |
| 13 | 1 | `u8` | scoring: `1` = sum of query-token max document dot products |
| 14 | 2 | `u16` | reserved, must be zero |

The fixed portion is followed by the query tensor in row-major order. Each
candidate then has:

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `u32` | opaque request-local candidate ID |
| 4 | `u32` | page tensor row/token count |
| `rows * dimension * dtype_size` | bytes | row-major page tensor |

Candidate IDs are assigned densely from zero in v1, but the sidecar must treat
them as opaque and echo them unchanged. Heap CTIDs and public GBrain page IDs
are not exposed to the sidecar.

Every query and candidate tensor in one request has the same dtype and
dimension. v1 accepts f32 and f16 only. RaBitQ arrays must use CPU exact or a
future protocol version that defines their representation.

## Successful Response Body

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `u32` | status `0` |
| 4 | `u32` | result count; must equal request candidate count |

Each result then contains:

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `u32` | echoed candidate ID |
| 4 | `f32` bits | positive MaxSim similarity |

Each requested candidate must appear exactly once. Unknown IDs, duplicate IDs,
missing IDs, non-finite scores, and trailing bytes fail the entire SQL query.
VectorChord converts public positive similarity to its SQL distance convention
by negating the value before ascending sort. Equal exact distances are ordered
deterministically by the candidate's internal heap key after response IDs have
been mapped back inside the backend.

## Error Response Body

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `u32` | nonzero sidecar-defined status |
| 4 | `u32` | UTF-8 error-message length |
| `length` | bytes | UTF-8 diagnostic, maximum 64 KiB |

The status namespace is reserved for the sidecar implementation in v1. Any
nonzero status fails `gpu` mode. In `auto` mode it triggers a complete CPU exact
rerank of the original candidate set; no partial GPU scores are reused.

## Limits and Timeouts

Before connecting, VectorChord enforces:

- `vchordrq.maxsim_candidate_limit`;
- `vchordrq.maxsim_gpu_max_batch_tokens`, including query and page tokens;
- `vchordrq.maxsim_gpu_max_batch_bytes`, including the request frame.

`vchordrq.maxsim_gpu_timeout_ms` is one overall connect-plus-write-plus-read
deadline. Connection and backend I/O poll PostgreSQL interrupts at least every
50 ms. The response is also bounded from the expected candidate count and the
64 KiB error limit.

v1 sends one bounded request. Splitting a candidate set into multiple GPU
batches is future work and must preserve one overall SQL deadline and
all-or-nothing result semantics.

External immutable tensor descriptors use the separate Phase 3B
[`TILEMAXSIM_IPC_V2`](TILEMAXSIM_IPC_V2.md) contract. They are never smuggled
into a v1 inline-tensor frame.

## Reference Sidecar

`devtools/tilemaxsim_reference_sidecar.py` is a dependency-free, single-threaded
CPU protocol oracle for v1 and resolver-injected v2 tests. It is intended for
codec and end-to-end development only; it is not a production or GPU executor.
The production-oriented CUDA implementation is documented in
[`TILEMAXSIM_CUDA_SIDECAR`](TILEMAXSIM_CUDA_SIDECAR.md).

Start one request/response cycle with:

```text
python3 devtools/tilemaxsim_reference_sidecar.py \
  --socket /tmp/vectorchord-tilemaxsim.sock \
  --once
```

Then configure the PostgreSQL session as an administrator and select the GPU
backend in the query session:

```sql
SET vchordrq.maxsim_gpu_endpoint = '/tmp/vectorchord-tilemaxsim.sock';
SET vchordrq.maxsim_candidate_limit = 256;
SET vchordrq.maxsim_backend = 'gpu';
```

The endpoint setting is `SUSET`. The reference sidecar creates its socket with
mode `0600` by default and refuses to replace a non-socket filesystem path.
Its default request limit is intentionally 64 MiB, lower than the extension's
configurable production ceiling.

Run its protocol and live-socket tests with:

```text
python3 -m unittest -v devtools/test_tilemaxsim_reference_sidecar.py
```

## Sidecar Conformance Cases

A conforming implementation must be tested for:

- f32 and f16 score equivalence with CPU exact;
- reordered response IDs;
- duplicate, missing, and unknown IDs;
- invalid magic/version/kind/request ID/body length;
- truncated and trailing data;
- NaN and infinite similarity;
- explicit error response;
- connection close during request and response;
- queue/compute duration exceeding the overall deadline.
