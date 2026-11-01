# Registered single-vector broker

`python -m single_vector_gpu.registered --config /private/broker.json` starts
an optional candidate scorer. The separate experimental `server` is not the
production integration endpoint. Python configuration must be mode 0600 and
owned by the runtime UID.

The broker accepts planner manifests registered by NeoBrain schema 161, reads
vectors through a dedicated restricted PostgreSQL role, and borrows already
admitted tenant leases. It does not call an encoder, accept SQL in HTTP, grant
new tenant quota, or replace final PostgreSQL evidence/ranking checks.

Ordinary traffic requires a current calibration matching epoch, transport and
runtime. The production minimum margin is two times the measured crossover;
missing or invalid policies return a CPU decision. Only the separate private
operator token permits calibration to measure GPU below that threshold.

Snapshots, pending batches, loader work and resident bytes are bounded. A
cohort uses a short ownership receipt that never outlives its parents; final
HTTP delivery forces a fresh ownership and epoch check. Cancellation cannot
free borrowed capacity before actual work finishes. Database connections
carry schema writer version 161; the service must be upgraded with the schema,
not allowed to bypass an upgrade fence.

Use the optional Containerfile with build context `services`, a pinned PyTorch
2.8.0 / CUDA 12.8 base and an explicit source revision. The service is FP32.
NeoBrain's deployment guide contains the matching role grants, Compose files,
private configuration fields, calibration worker and rollback procedure.

Verification: pure tests default to CPU and make no GPU throughput claim.
`NEOBRAIN_VECTOR_GPU_TEST_CONFIG` enables disposable PostgreSQL tests, including
column permissions, tenant leases, source changes, parent cancellation,
transaction ordering, and expired metadata cleanup. Never point these tests
at a business database.
