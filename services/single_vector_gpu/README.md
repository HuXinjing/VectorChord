# Experimental single-vector GPU batch search

This is an opt-in backend prototype in the VectorChord service tree. It does not
replace a PostgreSQL access method or change NeoBrain's default SQL route.

The immutable index supports cosine search, bounded query/document tiles and
on-device TopK merging. It never creates the full queries-by-corpus matrix.
FP32 disables TF32 in this process. The separate FP16 profile stores normalized
inputs in FP16 and accumulates matrix products into FP32; it requires independent
numerical and ranking qualification before use.

One operator-issued token grants one frozen tenant/model/generation snapshot.
A request may narrow its allowed sources but may not select another tenant,
model, generation or filesystem path. This is an experimental authentication
boundary, not a replacement for a database transaction snapshot or application
ACL checks. Production integration must feed revision-qualified snapshots,
apply deletion/invalidation, and recheck authority before delivering evidence.
Batch only requests with the same authorized source set. Do not union scopes.

The API listens on loopback. A request is a little-endian uint32 JSON header
length, JSON metadata, then row-major little-endian float32 query vectors:

```json
{"version":1,"rows":256,"k":20,"contract":{"tenant":"example","model":"encoder-v1","generation":"snapshot-sha256"},"allowed_sources":[1,2]}
```

The response has the same framing, metadata with `rows`, `k`, and timings, then
row-major int64 identities followed by float32 scores. A missing slot has ID
`-1` and score negative infinity. Boundary ties may select any tied identity,
matching SQL ordered only by distance. Unknown fields, malformed shapes,
non-finite/zero vectors and foreign snapshot contracts are rejected. Admission
is bounded to one active batch, a configured query limit, request/result byte
limits and a bounded GPU workspace.

Example commands, run with a compatible PyTorch CUDA runtime and NumPy:

```sh
PYTHONPATH=services python -m unittest discover -s services/single_vector_gpu -p 'test_*.py'
PYTHONPATH=services python -m single_vector_gpu.server \
  --data vectors.npz --metadata metadata.json --token-file token.private \
  --device cuda:0 --profile fp32
```

Operator metadata supplies the `contract` and `sha256` of the NPZ. Its generation
must equal that hash. The NPZ contains numeric `vectors`, `ids` and `sources`;
loading never permits pickle. The benchmark also expects `holdout_queries` and
uses a disposable database named `single_vector_bench`, with no business text.

`benchmark.py` records CPU HNSW, CPU exact scans, GPU resident execution and the
binary loopback API separately. Query encoding and real WAN transmission are
outside these backend measurements. CPU approximate and GPU exact TopK must be
compared for quality as well as speed. Do not describe sampled/shared-device
throughput as an end-to-end application or hardware upper limit.

## Adaptive dispatch

`calibration.estimate_crossover` consumes repeated burst-arrival P95 samples,
interpolates an observed stable CPU/GPU crossover, then selects the first
integer strictly above `crossover * 1.2`. The measured GPU latency must include
the full transport and collection allowance, rather than CUDA time alone.
Calibration is specific to the corpus, model, CPU budget, hardware and network
path. An absent, expired or mismatched calibration keeps CPU routing.

`dispatch.Dispatcher` is a host-side integration module. It collects compatible
arrivals for at most two milliseconds by default, grouping only identical
tenant/model/generation/source-set/TopK/dimension contracts. Its default device
budget is one running batch plus one waiting batch, with bounded total and
per-tenant pending requests. A busy/unavailable backend falls back to the
existing read-only CPU implementation; encoding is never replayed. A failure
cooldown prevents a broken backend from receiving every new cohort.

The host must supply the following adapters; this repository does not silently
substitute a permissive default:

- `ClusterAdmission`: the existing distributed tenant lease authority, shared
  by every application replica. CPU, GPU and waiting work consume the same
  original request lease. A test-local counter is not production admission.
- A live `authorize` callback per request, covering subject permissions and
  current source/revision state. It runs before submission, before execution
  and before delivery, including CPU fallback. It must reject lost authority.
- A scoped CPU search callback preserving the existing SQL filters/ranking,
  and a bounded GPU transport callback returning one validated result per row.
- A snapshot producer that verifies database publication state before calling
  `SnapshotCatalog.publish`, invalidates on deletion/revision/model changes,
  and reloads the new revision outside the publication transaction. Expired,
  retired or unpublished snapshots are never used for new GPU queries.

The catalog atomically swaps versions, enforces its tracked resident budget,
and retains leased retired versions until real work ends. Invalidation during
GPU execution discards that output and retries only the read-only CPU lookup,
with the original deadline and authority. Cancellation/deadline expiry blocks
delivery and does not release quota early while a backend is still running.

The binary experimental server is still a batch endpoint. It does not
automatically install this host module into NeoBrain, discover changes in a
production database, or enable any production default route. Those adapters
and their integration validation are required before enabling it in dev.
