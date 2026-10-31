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
