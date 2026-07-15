# MaxSim Tensor-Source Registry

## Scope

The registry binds one physical `vchordrq` MaxSim index to the tensor contract
that Phase 3B will use for exact reranking. It stores metadata only. Tensor
credentials and tensor contents never belong in this catalog.

The registry lifecycle, internal descriptor reader, and restricted Phase 3B
score executor are implemented. The reader validates model contract, shape,
dtype, reference, and checksum before the descriptor v2 IPC encoder accepts a
candidate. The internal Rust resolver calls `vchordrq_maxsim_source_info` and
never reads the private registry table directly. Same-heap bindings resolve
current names back to physical attribute numbers; independent descriptors are
read with one snapshot-consistent SQL join after candidate visibility is
resolved. Registering a source does not change ordinary `@#` execution.

Search an external full-tensor source with:

```sql
SELECT public_id, similarity
FROM vchordrq_maxsim_search(
  'visual_page_embeddings_maxsim_idx'::regclass,
  $1::halfvec[],
  1024, -- bounded coarse page candidates
  20    -- final exact rows
)
ORDER BY similarity DESC, public_id;
```

The current function is deliberately restricted:

- the registered source must use `external_ref` or `external_relation` storage;
- the named index must use `vector_maxsim_ops` or `halfvec_maxsim_ops` and the
  query array type must match it exactly;
- `vchordrq.maxsim_backend` must be `gpu`, with an operations-controlled Unix
  socket endpoint;
- `candidate_limit` is between 1 and 65,536 and `top_k` cannot exceed it;
- the caller must have table-level `SELECT` on the indexed heap and, for
  `external_relation`, the descriptor relation; projection uses the caller's
  MVCC snapshot and PostgreSQL row visibility;
- the caller selects the physical relation/partition. Arbitrary SQL predicates
  would require an optional future CustomScan path and are not accepted as
  strings;
- HOT root TIDs from the index are resolved through the table AM before the
  descriptor query, while backend IDs remain opaque root-candidate IDs.

The result contains the registered stable `bigint` public ID and positive exact
similarity. Equal scores are ordered by public ID. GPU/sidecar failures fail the
whole statement; this surface does not silently fall back to coarse scores.

The bundled CUDA sidecar uses `sha256://<digest>` tensor references backed by
an operations-configured, per-model local content-addressed cache. GBrain may
populate that cache from its own storage layer; storage credentials and remote
destinations are not registry fields. See
[`TILEMAXSIM_CUDA_SIDECAR`](TILEMAXSIM_CUDA_SIDECAR.md) for the file layout,
runtime limits, and deployment contract.

## Registration

Only the index owner, a member of the owning role, or a superuser can register
or replace a binding. An independently owned descriptor relation must also be
owned by the caller (or a role it belongs to). Descriptor columns are resolved
to attribute numbers at registration time.

`external_relation` is the recommended production layout. It avoids widening
and rewriting an already indexed multi-vector heap: the heap keeps its stable
public ID, model contract, and coarse array, while a compact relation keeps one
external descriptor per public ID.

```sql
CREATE TABLE visual_page_tensor_descriptors (
  document_page_id bigint PRIMARY KEY,
  tensor_ref text NOT NULL,
  tensor_rows integer NOT NULL,
  tensor_dim integer NOT NULL,
  tensor_dtype text NOT NULL,
  tensor_checksum text NOT NULL
);

SELECT vchordrq_register_maxsim_source(
  index_relation => 'visual_page_embeddings_maxsim_idx'::regclass,
  model_contract_id => 'colqwen3.5@revision+preprocessing-hash',
  storage => 'external_relation',
  model_contract_column => 'model_contract_id'::name,
  public_id_column => 'document_page_id'::name,
  tensor_ref_column => 'tensor_ref'::name,
  tensor_rows_column => 'tensor_rows'::name,
  tensor_dim_column => 'tensor_dim'::name,
  tensor_dtype_column => 'tensor_dtype'::name,
  tensor_checksum_column => 'tensor_checksum'::name,
  descriptor_relation => 'visual_page_tensor_descriptors'::regclass,
  descriptor_public_id_column => 'document_page_id'::name
);
```

The index must be a valid, ready, single-key `vchordrq` index using one of the
four MaxSim opclasses. The model-contract and descriptor string columns must be
`NOT NULL text`; the public ID must be `NOT NULL bigint`; rows and dimension
must be `NOT NULL integer`. For `external_relation`, its public ID must have a
valid, non-partial, single-key unique index. Its public ID and five descriptor
columns must be distinct. Missing descriptor rows fail the complete search.

`external_ref` remains available when all five descriptor columns already live
on the indexed heap. In that layout the two heap metadata columns and five
descriptor columns must all be distinct; omit the two descriptor-relation
arguments.

For `heap_array`, omit all five external descriptor columns. The model-contract
and public-ID columns are still required:

```sql
SELECT vchordrq_register_maxsim_source(
  index_relation => 'visual_page_embeddings_maxsim_idx'::regclass,
  model_contract_id => 'colqwen3.5@revision+preprocessing-hash',
  storage => 'heap_array',
  model_contract_column => 'model_contract_id'::name,
  public_id_column => 'document_page_id'::name
);
```

Unregister explicitly with:

```sql
SELECT vchordrq_unregister_maxsim_source(
  'visual_page_embeddings_maxsim_idx'::regclass
);
```

Resolve and revalidate a binding without reading tensor values:

```sql
SELECT *
FROM vchordrq_maxsim_source_info(
  'visual_page_embeddings_maxsim_idx'::regclass
);
```

This function resolves current column names from stored attribute numbers,
rechecks the live index/opclass and every column type, and requires either index
ownership or table `SELECT` privilege. It is the metadata boundary consumed by
the Phase 3B executor; it does not grant access to row values.

## DDL and Security Semantics

- Direct DML on `_vchordrq_maxsim_sources` is revoked from `PUBLIC`.
- Registration functions use definer rights with the fixed search path
  `pg_catalog, pg_temp`; ownership is checked against the session user and its
  role memberships.
- Attribute-number binding survives column renames.
- An `sql_drop` event trigger removes the binding when its index, heap relation,
  descriptor relation, or any bound column is dropped.
- A concurrent index replacement gets a new OID and therefore requires explicit
  re-registration. This is fail-closed behavior.
- Raw OID bindings are intentionally not portable pg_dump data. Restore tooling
  must register sources after restoring tables and indexes.
- Runtime metadata lookup revalidates the relation, opclass, and descriptor
  types. Type-altering DDL that does not drop an attribute therefore makes
  resolution fail closed until the binding is corrected or re-registered.
- The search API uses the caller's snapshot and permissions and preserves
  PostgreSQL row visibility. The registry itself grants no table or tensor
  access and does not define an application authorization or routing model.
