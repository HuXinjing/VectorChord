# VectorChord TileMaxSim IPC Protocol v2 and v3

## Scope

Version 2 carries an inline query tensor plus immutable external tensor
descriptors. It is the Phase 3B transport for a coarse/sketch `vchordrq` index
whose final score comes from a different full tensor. It does not change the
meaning of ordinary SQL `@#` and is not enabled by source registration alone.

Version 3 is the backward-compatible scheduled form. It adds an authenticated
upstream scheduling domain, priority, and end-to-end timeout. These fields
control resource scheduling only; they are not authorization claims and never
replace PostgreSQL row visibility or application ACL checks.

The transport is an operations-controlled Unix socket. It is not a network
authentication protocol. All integers and floating-point bit patterns are
little-endian. The sidecar must read and write exact frame lengths; partial
results are forbidden.

## Common Header

The 24-byte header is the same shape as v1:

| Offset | Size | Type | Meaning |
| --- | ---: | --- | --- |
| 0 | 4 | bytes | ASCII magic `VCTM` |
| 4 | 2 | `u16` | protocol version `2` |
| 6 | 2 | `u16` | message kind: `1` request, `2` response |
| 8 | 8 | `u64` | request ID, echoed unchanged |
| 16 | 8 | `u64` | body length, excluding the header |

The response must echo the request version. Unknown versions, message kinds,
request IDs, or inconsistent lengths fail the complete request.

## Version 3 Scheduling Extension

For a version 3 request, the v2 fixed request body is immediately followed by
12 additional fixed bytes:

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `i32` | priority from -100 through 100; higher values are more urgent |
| 4 | `u32` | end-to-end timeout in milliseconds, from 1 through 600000 |
| 4 | `u32` | scheduling-tenant UTF-8 byte length |

The variable body then contains model-contract bytes, scheduling-tenant bytes,
the query tensor, and the candidate descriptors, in that order. The tenant is
nonempty, limited to 256 bytes, and cannot contain control characters.

The timeout begins when the daemon accepts the connection, not when GPU work
starts. A server-side timeout may shorten it. Priority affects queue order only;
it cannot expand the request deadline, cache quota, candidate scope, or access
rights. Protocol v2 requests enter the default scheduling domain at priority
zero and use the server timeout.

## External Rerank Request

The fixed request body is 20 bytes:

| Offset | Size | Type | Meaning |
| --- | ---: | --- | --- |
| 0 | 4 | `u32` | tensor dimension |
| 4 | 4 | `u32` | query row/token count |
| 8 | 4 | `u32` | candidate count |
| 12 | 1 | `u8` | dtype: `1` f32, `2` IEEE f16 |
| 13 | 1 | `u8` | scoring: `1` = sum of query-token max document dot products |
| 14 | 2 | `u16` | reserved, must be zero |
| 16 | 4 | `u32` | UTF-8 model-contract byte length |

The fixed body is followed, in order, by:

1. the model-contract UTF-8 bytes;
2. the inline query tensor in row-major order;
3. exactly `candidate_count` descriptor entries.

Each descriptor entry is:

| Size | Type | Meaning |
| ---: | --- | --- |
| 4 | `u32` | opaque request-local candidate ID |
| 4 | `u32` | full tensor row/token count |
| 4 | `u32` | tensor-reference UTF-8 byte length |
| 4 | `u32` | checksum UTF-8 byte length |
| variable | bytes | tensor reference |
| variable | bytes | checksum |

All tensors in one request have the fixed body's dtype and dimension. The
backend validates each registered row descriptor against them before IPC. The
model contract and every descriptor field are nonempty, bounded UTF-8 without
control characters. Current bounds are 512 bytes for the contract and
checksum, and 4096 bytes for the reference.

The checksum encoding is `sha256:` followed by 64 lowercase hexadecimal
digits. It covers the canonical row-major scalar payload after any immutable
object envelope is decoded. This makes verification independent of an object
store's metadata and compression. A resolver for formats such as safetensors
must decode the selected tensor, validate its declared shape/dtype, produce the
canonical payload, and then verify this digest before compute.

Candidate IDs are opaque. PostgreSQL heap CTIDs, registered public document
IDs, application routing identifiers, and credentials are never encoded.
VectorChord maps response IDs back to heap keys inside the backend.

## Resolution and Security

The sidecar owns descriptor resolution. A production resolver must:

- allowlist schemes, buckets/namespaces, and immutable reference forms;
- obtain credentials from operations-managed sidecar configuration, never from
  PostgreSQL rows or this protocol;
- reject redirects or network destinations outside its allowlist;
- enforce the model contract, dtype, shape, and checksum before GPU access;
- bound queue, fetch, decode, host-to-device, and compute work by the one SQL
  request deadline.

The complete control frame must be parsed and validated before external I/O.
An absent resolver fails closed. The dependency-free reference sidecar exposes
v2 resolution only through an injected callback; its CLI deliberately has no
filesystem, HTTP, or object-store resolver.

## Limits

Both peers enforce:

- candidate count;
- total query plus declared document tokens;
- total declared canonical tensor bytes;
- control-frame bytes;
- one overall connect/write/read deadline.

`vchordrq.maxsim_gpu_max_batch_tokens` bounds total declared tokens.
`vchordrq.maxsim_gpu_max_batch_bytes` bounds both the control frame and total
declared canonical tensor bytes independently. A small descriptor frame cannot
therefore authorize unbounded object loads.

Version 2 currently sends one bounded batch. Future splitting must retain one
overall deadline and all-or-nothing result semantics.

## Response

The success and error bodies are identical to v1 except that the header version
echoes 2 or 3. A success body begins with status `0` and a result count equal to the
number of accepted descriptors, followed by `(candidate_id u32, similarity
f32)` pairs. Every candidate appears exactly once. Unknown, duplicate, missing,
non-finite, or trailing results fail the whole SQL operation.

Similarity is positive public MaxSim. VectorChord negates it to its ascending
distance convention and uses the internal heap key as a deterministic tie
breaker for the query execution.

## Current Integration State

The Rust descriptor source, strict v2/v3 encoder/decoder, and reference-sidecar
protocol oracle are implemented and unit tested. The internal runtime resolver
also consumes the privilege-aware SQL registry boundary and produces physical
attribute bindings. The restricted `vchordrq_maxsim_search` score API now sends
validated, SQL-visible external descriptors over v2 and returns exact positive
similarity. Ordinary `@#` scans keep their existing stored-array semantics.

The native Rust/CUDA daemon additionally provides bounded global and per-tenant
admission, fair/priority/fair-priority policies, request aging, candidate/token
quanta, deadline and disconnect cancellation between CUDA launches, per-tenant
cache caps, optional reservations, and immutable-shard reload on `SIGHUP`.
An optional operations-only Unix status socket exposes HTTP `/healthz` and
Prometheus `/metrics`; it is separate from this binary scoring protocol.

The dependency-free reference sidecar remains a protocol oracle. The separate
[`TILEMAXSIM_CUDA_SIDECAR`](TILEMAXSIM_CUDA_SIDECAR.md) implementation executes
v1/v2 on CUDA and provides a per-model, allowlisted, content-addressed local
resolver with checksum, resource-limit, deadline, backpressure, and structured
metric enforcement. GBrain remains responsible for populating that immutable
node-local cache from its storage system. Production acceptance still requires
the committed GBrain corpus recall, latency, concurrency, and failure matrix.
