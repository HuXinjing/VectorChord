# Exact current-quantum batching

`--exact-quantum-batching` is opt-in. The existing overlap scheduler remains the
unchanged default and supplies the same-build control for measurement.

The old scheduler compares complete remaining candidate sets, while the engine
can fuse only identical ordered current slices. A token/FMA limit can give two
otherwise identical requests different quantum ends. Exact mode compares the
current legal slices and aligns them down to the smaller admitted
prefix length. It does not enlarge candidate, token or FMA budgets. Requests
with a differing tensor within that prefix remain separate; overlap alone is
not sufficient. Immutable shared candidate arrays with the same start offset
use pointer equality. Independent arrays compare contract, digest, shape and
dtype directly without allocating formatted keys. Per-request candidate IDs
remain attached by the engine to the original request slot.

The scheduler still considers only the globally next policy-selected request.
Different tenants, explicit priorities, model contracts, scoring profiles,
quantization contracts and tensor dtypes cannot share an exact batch. A blocked,
expired or disconnected winner is not skipped to find another compatible
request. Priority aging, tenant weights, admission permits, deadlines,
cooperative requeue and connection-scoped response IDs retain their existing
paths. After a prefix is shortened, selected FMA cost is reconciled so later
fair scheduling charges actual submitted work. A subsequent cancellation still
uses the existing completion and permit-release path.

Exact mode does not run the full-set, queue-pair shadow telemetry at enqueue.
The existing `tilemaxsim_batch_*` and shadow-dispatch counters retain their
legacy meanings and remain zero when only exact mode is used. New dispatch
observations are separate:

- `tilemaxsim_exact_batch_pairs_total`: admitted pairs against the batch leader.
- `tilemaxsim_exact_batch_candidates_total`: final shared prefix candidates
  summed over admitted pairs; these are not whole-request overlap counts.
- `tilemaxsim_exact_batch_prefix_shrinks_total`: selected quantum shrink events.
- `tilemaxsim_exact_batch_blocked_next_total`: batch formation stopped at the
  next policy winner, including liveness and compatibility boundaries.

The existing minimum-shared-candidate setting remains a gate: an exact prefix
has a shared ratio of 1000, so a configured value above 1000 still prevents a
batch. Backend residency/capability validation and final fusion remain in the
engine; exact scheduler admission is not a claim that the backend executed it.
Use existing fused-request counters to measure actual execution.

CPU verification covers independently allocated and shared candidate arrays,
misaligned and reordered overlap, tensor identity and tenant/profile boundaries,
query-specific FMA budgets, duplicate connection-local request IDs, weighted
service refunds, policy-winner preservation, expired/dead peers, plus the
existing scheduler and lifecycle tests. GPU numerical equivalence, throughput
and tail latency must be measured independently on matched inputs; CPU tests do
not establish those claims.
