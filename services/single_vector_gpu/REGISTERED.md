# Registered single-vector broker

`python -m single_vector_gpu.registered --config /private/broker.json` starts
an optional candidate scorer. The separate experimental `server` is not the
production integration endpoint. Python configuration must be mode 0600 and
owned by the runtime UID.

The broker accepts planner manifests registered by NeoBrain schema 167, reads
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
carry schema writer version 167; the service must be upgraded with the schema,
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

Pipeline bounds: the registered endpoint accepts at most 512 rows and a 16 MiB
request. Dispatcher budgets remain finite (2,048 globally, 1,024 per tenant),
with one GPU worker and a bounded waiting slot. Transport cohorts are split
into at most 64-row FP32 compute blocks to retain the previous numerical
behavior. Optional `profile_timings` logs request lifecycle durations without
logging query vectors, source contents or credentials. Matching NeoBrain uses
two in-flight endpoint RPCs and independent bounded result projection.

Each newly opened connection exercises the real database write fence using
the already-granted release function with empty identities in a rollback-only
transaction. An incompatible writer is closed before entering the pool. This
probe neither changes leases nor grants access to the configuration table.
The declared supported version is part of the binary, never copied from the
database minimum or overridden by operator configuration. Upgrades beyond the
supported version still require a tested broker release.

Schema 167 compatibility verification includes 39 CPU functional tests on an
isolated current-schema PostgreSQL fixture. They cover permission rejection,
lease borrowing/release, source epoch changes, cancellation, and the connection
probe's unchanged lease rows. A connection claiming the prior writer version
is rejected before caller work and is not returned to the pool. These are
correctness tests, not GPU throughput measurements. A changed broker source
fingerprint invalidates old calibrations; ordinary requests retain CPU fallback
until a current calibration is available.

The schema 167 release also verifies a registered 2048-dimensional shadow recipe
alongside legacy 1536-dimensional vectors. Schema 165 widens shadow vector
storage; 166 adds publication state and 167 replaces transcript storage. The
broker does not write those new tables. Its restricted vector reads and the
existing borrowed-capacity functions remain the integration contract. All 39
checks pass against an isolated migrated schema 167 database, including the
real writer-fence probe and rejection of an older writer.

This service is deployed independently of the NeoBrain Compose project. Before
advancing the database schema, release a verified compatible broker image as
well; HTTP readiness alone does not establish writer compatibility. Preserve
its private configuration, GPU allocation and previous image. Do not change
application_name on a running process or derive its claim from the database.

For a host whose legacy image builder cannot convert an existing cached layer
format, `oci_overlay.py` can construct the code-only layer without downloading
or rewriting CUDA dependencies. Read the immutable base manifest and its config
from containerd's content store; pass their files, the reviewed `registered.py`,
a full source commit and the new image reference to the tool. Import its archive
into the same content store with `ctr images import --no-unpack`. The base blobs
must already exist there. Keep the old image tagged until rollout validation.
The archive changes only the module and provenance labels; it preserves the base
entrypoint, user, environment, health check, layer descriptors and rootfs chain.
Two pure archive tests verify those invariants, digest checks and deterministic
output. This is image construction, never an edit of a running container.
