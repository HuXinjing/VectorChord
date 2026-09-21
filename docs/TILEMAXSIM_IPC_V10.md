# TileMaxSim external catalog protocol v10

Version 10 extends the typed compact logical request (v8) with a bounded,
versioned descriptor catalog. It does not change response framing or scoring.

After the query tensor, every v10 request carries:

| Field | Encoding | Meaning |
| --- | --- | --- |
| mode | `u8` | `1` selection, `2` registration |
| catalog digest | 32 bytes | SHA-256 of the caller's opaque source revision |

A selection frame then carries `candidate_count` positive `public_id` values in
strictly increasing order. The first value and subsequent deltas use canonical
unsigned LEB128. A registration frame instead carries `candidate_count` entries
of `public_id:i64`, `rows:u32`, and `sha256:32`; entries must also be positive and
strictly increasing. Response candidate IDs remain zero-based ordinals in this
wire order.

The daemon namespaces catalogs by tenant, model contract, scoring profile,
dimension, storage dtype, and catalog digest. Registration merges previously
unseen public IDs into the same revision and rejects conflicting redefinitions.
Selection fails explicitly with `descriptor catalog miss` if the revision or any
ID is absent. Clients retry that request as a registration frame. Bounded LRU
eviction means a miss is normal after restart or pressure; it never permits stale
or cross-tenant reuse.

Clients may fall back to v9 manifest references when a daemon rejects v10 during
a rolling upgrade. ACL and source filtering remain the PostgreSQL caller's
responsibility and must happen before the catalog token and candidate list are
formed.
