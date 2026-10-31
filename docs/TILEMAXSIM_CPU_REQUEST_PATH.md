# TileMaxSim CPU request path

The daemon supports persistent forms of the existing catalog (v10 -> v15) and
manifest (v9 -> v16) requests. Only the envelope version changes: candidate
visibility, descriptor validation, scoring profile, top-k and payload layout
remain the same. The SQL transport negotiates support per endpoint/version,
returns only successful validated responses to its bounded connection cache,
and preserves the existing deadline and single reconnect limit. A legacy daemon
falls back to the original one-shot protocol. Upgrade the daemon before the SQL
extension. PostgreSQL installations that preload `vchord` must restart their
server to activate the new extension; replacing the scorer image alone does not
activate SQL connection reuse.

Three independent daemon options control host overhead:

- `--prepared-batch-window`: first drain a bounded amount of prepared work into
  the existing fair-priority queue, then deduct elapsed ready time from the
  collection window. Request deadlines and admission limits remain enforced.
- `--resident-readiness-proofs`: cache only positive readiness observations,
  scoped by descriptor-vector identity, candidate range, tenant, model,
  scoring/quantization contract and every device cache generation. Reload clears
  observations. Weak references do not retain candidate vectors. These are
  observations, not leases; scoring-time cache validation and pinning remain.
- `--cache-log-interval=N`: include a full cache snapshot in the first and every
  Nth request log. Other request logs have `cache: null` and
  `cache_snapshot_sampled: false`. Basic request/error logs and live management
  metrics remain available. N must be positive.

Defaults preserve the original scheduler/readiness checks and log every cache
snapshot (`N=1`). Enable the options only for hardware and workloads validated
for the deployment. Remove the first two options and set the interval to one to
restore the original host policies. GPU math and precision are unaffected.

The request log's `protocol_version` identifies negotiated persistent frames.
Do not infer complete application throughput, cold-data performance, or relevance
quality from a hot scorer-only benchmark. Encoding precomputation is not part of
this change.

## Opt-in exact-FP16 prepared resident selections

`TILEMAXSIM_PREPARED_CANDIDATES=1` enables preparation reuse for exact-FP16
single-query and fused scoring in the daemon. It does not change PostgreSQL
visibility checks, descriptor loading, wire precision, scoring kernels, or the
scheduler's candidate/token/FMA quantum limits. The default remains disabled.

A reusable plan is keyed by the immutable candidate `Arc`, the exact selected
interval, tenant, model contract, profile, quantization contract, query and
candidate dtypes, dimension, and every device cache generation. Plans are built
only after full residency and descriptor validation, and a structural cache
change or shard reload prevents reuse. Cached scoring still updates TinyLFU and
hit counters; batch scoring preserves the original per-logical-candidate cache
access policy even if the preparation originated from deduplicated single-query
scoring. A synchronous engine call owns the arena while a checked plan is used;
these plans are not asynchronous GPU leases.

The cache is bounded to 64 plans and, when enabled, an estimated 64 MiB including
retained descriptor storage and prepared metadata. Repeated references to the
same candidate `Arc` count its storage once. The estimate reserves lazy raw-FP8
key storage for every descriptor up front, including currently untouched slices,
and includes shared-allocation headers and retained deque capacity. Allocator
internal bookkeeping is outside this estimate. `status_json` request cache snapshots
include `prepared_candidates` with hits, misses, plan count and estimated bytes.
The existing raw-FP8 preparation/coalescing behavior remains available; FP16 never
uses its enlarged-quantum path. Warmup must complete GPU admission and then
populate the exact intervals before steady-state comparisons.
