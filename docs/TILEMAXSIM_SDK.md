# TileMaxSim Rust SDK

`tilemaxsim-client` is the supported Rust boundary for components that must
communicate with the native TileMaxSim daemon. Applications that only query
VectorChord should continue to use the SQL API; they should not construct daemon
frames themselves.

The SDK currently owns the latency-sensitive catalog protocol:

- v10 additive catalog registration and explicit public-ID selection;
- v11/v14 global selection references;
- v12/v13 scoped selection references;
- stable catalog and selection digest calculation;
- raw FP8, row-scaled FP8, FP16, INT8, PQ, and OPQ/RPQ profile identifiers;
- partial top-k response decoding; and
- typed `CatalogMiss`, `ManifestMiss`, and remote failures.

The intended miss path is deterministic:

1. Send a persistent v14 selection reference (or v13 for dual-scope search).
2. On `SdkError::CatalogMiss`, send a v10 catalog registration.
3. Send a v10 explicit selection to populate the bounded selection cache.
4. Retry the v14/v13 reference on the reusable connection.

Callers supply canonical tensor bytes. The SDK validates shape, dtype, scheduler
bounds, sorted positive public IDs, scoped subset membership, top-k, and
quantization-contract compatibility before producing a frame.

```rust
use tilemaxsim_client::{
    encode_catalog_selection_reference, CatalogRequest, ScoringProfile,
    TensorDtype,
};

let request = CatalogRequest {
    request_id: 7,
    model_contract: "colqwen@1",
    tenant: "tenant-42",
    priority: 0,
    timeout_ms: 2_000,
    query_rows: 32,
    dimension: 128,
    query_dtype: TensorDtype::Float16,
    candidate_dtype: TensorDtype::Fp8E4m3,
    scoring_profile: ScoringProfile::Fp8E4m3Raw,
    quantization_contract: None,
    top_k: 50,
    query: &query_bytes,
    catalog_revision: "sha256:application-owned-stable-revision",
};

let frame = encode_catalog_selection_reference(
    &request,
    &sorted_public_ids,
    None,
    true,
)?;
```

`tilemaxsim-client` is an accelerator-independent workspace crate shared by the
PostgreSQL extension and daemon. The daemon parser is exercised directly by an
interoperability test. Protocol changes must update that test in the same
commit; duplicating frame layouts in downstream applications is unsupported.
