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
