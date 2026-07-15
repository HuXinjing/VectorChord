-- Preserve the post-1.1.1 composite-type compatibility fix when upgrading.

CREATE OR REPLACE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_vector';

CREATE OR REPLACE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_halfvec';

CREATE OR REPLACE FUNCTION sphere(rabitq8, real) RETURNS sphere_rabitq8
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_rabitq8';

CREATE OR REPLACE FUNCTION sphere(rabitq4, real) RETURNS sphere_rabitq4
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_rabitq4';

-- Phase 3B tensor-source bindings

CREATE TABLE _vchordrq_maxsim_sources (
    index_oid oid PRIMARY KEY,
    heap_oid oid NOT NULL,
    model_contract_id text NOT NULL
        CHECK (
            model_contract_id OPERATOR(pg_catalog.<>) ''::text
            AND pg_catalog.length(model_contract_id) OPERATOR(pg_catalog.<=) 512
        ),
    storage text NOT NULL CHECK (
        storage OPERATOR(pg_catalog.=) ANY (
            ARRAY['heap_array', 'external_ref', 'external_relation']::text[]
        )
    ),
    model_contract_attnum smallint NOT NULL CHECK (
        model_contract_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    public_id_attnum smallint NOT NULL CHECK (
        public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    descriptor_oid oid,
    descriptor_public_id_attnum smallint,
    tensor_ref_attnum smallint,
    tensor_rows_attnum smallint,
    tensor_dim_attnum smallint,
    tensor_dtype_attnum smallint,
    tensor_checksum_attnum smallint,
    registered_by oid NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        (
            storage OPERATOR(pg_catalog.=) 'heap_array'::text
            AND descriptor_oid IS NULL
            AND descriptor_public_id_attnum IS NULL
            AND tensor_ref_attnum IS NULL
            AND tensor_rows_attnum IS NULL
            AND tensor_dim_attnum IS NULL
            AND tensor_dtype_attnum IS NULL
            AND tensor_checksum_attnum IS NULL
        )
        OR
        (
            storage OPERATOR(pg_catalog.=) 'external_ref'::text
            AND descriptor_oid IS NULL
            AND descriptor_public_id_attnum IS NULL
            AND tensor_ref_attnum IS NOT NULL
            AND tensor_ref_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_rows_attnum IS NOT NULL
            AND tensor_rows_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dim_attnum IS NOT NULL
            AND tensor_dim_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dtype_attnum IS NOT NULL
            AND tensor_dtype_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_checksum_attnum IS NOT NULL
            AND tensor_checksum_attnum OPERATOR(pg_catalog.>) 0::smallint
        )
        OR
        (
            storage OPERATOR(pg_catalog.=) 'external_relation'::text
            AND descriptor_oid IS NOT NULL
            AND descriptor_public_id_attnum IS NOT NULL
            AND descriptor_public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_ref_attnum IS NOT NULL
            AND tensor_ref_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_rows_attnum IS NOT NULL
            AND tensor_rows_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dim_attnum IS NOT NULL
            AND tensor_dim_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dtype_attnum IS NOT NULL
            AND tensor_dtype_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_checksum_attnum IS NOT NULL
            AND tensor_checksum_attnum OPERATOR(pg_catalog.>) 0::smallint
        )
    )
);

REVOKE ALL ON TABLE _vchordrq_maxsim_sources FROM PUBLIC;

CREATE FUNCTION vchordrq_register_maxsim_source(
    index_relation regclass,
    model_contract_id text,
    storage text,
    model_contract_column name,
    public_id_column name,
    tensor_ref_column name DEFAULT NULL,
    tensor_rows_column name DEFAULT NULL,
    tensor_dim_column name DEFAULT NULL,
    tensor_dtype_column name DEFAULT NULL,
    tensor_checksum_column name DEFAULT NULL,
    descriptor_relation regclass DEFAULT NULL,
    descriptor_public_id_column name DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    heap_oid oid;
    index_owner oid;
    descriptor_oid oid;
    descriptor_owner oid;
    tensor_relation_oid oid;
    normalized_storage text;
    attnum smallint;
    atttypid oid;
    attnotnull boolean;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    descriptor_id_is_unique boolean;
BEGIN
    IF index_relation IS NULL THEN
        RAISE EXCEPTION 'index_relation must not be NULL';
    END IF;
    IF model_contract_id IS NULL
       OR btrim(model_contract_id) = ''
       OR length(model_contract_id) > 512 THEN
        RAISE EXCEPTION 'model_contract_id must contain between 1 and 512 characters';
    END IF;
    model_contract_id := btrim(model_contract_id);
    IF model_contract_column IS NULL OR public_id_column IS NULL THEN
        RAISE EXCEPTION 'model_contract_column and public_id_column must not be NULL';
    END IF;

    normalized_storage := lower(btrim(storage));
    IF normalized_storage IS NULL
       OR normalized_storage NOT IN ('heap_array', 'external_ref', 'external_relation') THEN
        RAISE EXCEPTION 'storage must be heap_array, external_ref, or external_relation';
    END IF;

    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    IF caller_oid IS NULL THEN
        RAISE EXCEPTION 'could not resolve caller role';
    END IF;

    SELECT x.indrelid, i.relowner
    INTO heap_oid, index_owner
    FROM pg_catalog.pg_index AS x
    JOIN pg_catalog.pg_class AS i ON i.oid = x.indexrelid
    JOIN pg_catalog.pg_class AS h ON h.oid = x.indrelid
    JOIN pg_catalog.pg_am AS am ON am.oid = i.relam
    JOIN pg_catalog.pg_opclass AS opc ON opc.oid = x.indclass[0]
    WHERE x.indexrelid = index_relation::oid
      AND i.relkind = 'i'
      AND h.relkind IN ('r', 'm')
      AND am.amname = 'vchordrq'
      AND opc.opcname IN (
          'vector_maxsim_ops',
          'halfvec_maxsim_ops',
          'rabitq8_maxsim_ops',
          'rabitq4_maxsim_ops'
      )
      AND x.indisvalid
      AND x.indisready
      AND x.indnatts = 1
      AND x.indnkeyatts = 1;
    IF heap_oid IS NULL THEN
        RAISE EXCEPTION 'relation % is not a valid single-key vchordrq MaxSim index',
            index_relation;
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the index owner may register its MaxSim tensor source';
    END IF;

    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = heap_oid
      AND a.attname = model_contract_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'model contract column % must be a NOT NULL text column',
            model_contract_column;
    END IF;
    model_contract_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = heap_oid
      AND a.attname = public_id_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'bigint'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'public ID column % must be a NOT NULL bigint column',
            public_id_column;
    END IF;
    public_id_attnum := attnum;

    IF normalized_storage = 'external_relation' THEN
        IF descriptor_relation IS NULL OR descriptor_public_id_column IS NULL THEN
            RAISE EXCEPTION 'external_relation sources require descriptor_relation and descriptor_public_id_column';
        END IF;
        SELECT c.oid, c.relowner
        INTO descriptor_oid, descriptor_owner
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = descriptor_relation::oid
          AND c.relkind IN ('r', 'm');
        IF descriptor_oid IS NULL THEN
            RAISE EXCEPTION 'descriptor relation % must be a table or materialized view',
                descriptor_relation;
        END IF;
        IF NOT pg_catalog.pg_has_role(caller_oid, descriptor_owner, 'USAGE') THEN
            RAISE EXCEPTION 'only the descriptor relation owner may register it as a MaxSim tensor source';
        END IF;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = descriptor_oid
          AND a.attname = descriptor_public_id_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'bigint'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'descriptor public ID column % must be a NOT NULL bigint column',
                descriptor_public_id_column;
        END IF;
        descriptor_public_id_attnum := attnum;

        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index AS x
            WHERE x.indrelid = descriptor_oid
              AND x.indisunique
              AND x.indisvalid
              AND x.indisready
              AND x.indnkeyatts = 1
              AND x.indkey[0] = descriptor_public_id_attnum
              AND x.indexprs IS NULL
              AND x.indpred IS NULL
        ) INTO descriptor_id_is_unique;
        IF NOT descriptor_id_is_unique THEN
            RAISE EXCEPTION 'descriptor public ID column % must have a non-partial single-key unique index',
                descriptor_public_id_column;
        END IF;
        tensor_relation_oid := descriptor_oid;
    ELSE
        IF descriptor_relation IS NOT NULL OR descriptor_public_id_column IS NOT NULL THEN
            RAISE EXCEPTION '% sources must not specify a descriptor relation', normalized_storage;
        END IF;
        tensor_relation_oid := heap_oid;
    END IF;

    IF normalized_storage = 'heap_array' THEN
        IF tensor_ref_column IS NOT NULL
           OR tensor_rows_column IS NOT NULL
           OR tensor_dim_column IS NOT NULL
           OR tensor_dtype_column IS NOT NULL
           OR tensor_checksum_column IS NOT NULL THEN
            RAISE EXCEPTION 'heap_array sources must not specify external tensor columns';
        END IF;
    ELSE
        IF tensor_ref_column IS NULL
           OR tensor_rows_column IS NULL
           OR tensor_dim_column IS NULL
           OR tensor_dtype_column IS NULL
           OR tensor_checksum_column IS NULL THEN
            RAISE EXCEPTION 'external tensor sources require ref, rows, dim, dtype, and checksum columns';
        END IF;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_ref_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor ref column % must be a NOT NULL text column',
                tensor_ref_column;
        END IF;
        tensor_ref_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_rows_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor rows column % must be a NOT NULL integer column',
                tensor_rows_column;
        END IF;
        tensor_rows_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_dim_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor dim column % must be a NOT NULL integer column',
                tensor_dim_column;
        END IF;
        tensor_dim_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_dtype_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor dtype column % must be a NOT NULL text column',
                tensor_dtype_column;
        END IF;
        tensor_dtype_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_checksum_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor checksum column % must be a NOT NULL text column',
                tensor_checksum_column;
        END IF;
        tensor_checksum_attnum := attnum;

        IF (CASE WHEN normalized_storage = 'external_ref' THEN 7 ELSE 6 END) <> (
            SELECT count(DISTINCT u.attnum)
            FROM pg_catalog.unnest(ARRAY[
                CASE WHEN normalized_storage = 'external_ref' THEN model_contract_attnum ELSE descriptor_public_id_attnum END,
                CASE WHEN normalized_storage = 'external_ref' THEN public_id_attnum ELSE NULL END,
                tensor_ref_attnum,
                tensor_rows_attnum,
                tensor_dim_attnum,
                tensor_dtype_attnum,
                tensor_checksum_attnum
            ]) AS u(attnum)
        ) THEN
            RAISE EXCEPTION 'tensor source columns must be distinct';
        END IF;
    END IF;

    EXECUTE pg_catalog.format(
        'INSERT INTO %I._vchordrq_maxsim_sources (
             index_oid, heap_oid, model_contract_id, storage,
             model_contract_attnum, public_id_attnum,
             descriptor_oid, descriptor_public_id_attnum,
             tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
             tensor_dtype_attnum, tensor_checksum_attnum, registered_by
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
         ON CONFLICT (index_oid) DO UPDATE SET
             heap_oid = EXCLUDED.heap_oid,
             model_contract_id = EXCLUDED.model_contract_id,
             storage = EXCLUDED.storage,
             model_contract_attnum = EXCLUDED.model_contract_attnum,
             public_id_attnum = EXCLUDED.public_id_attnum,
             descriptor_oid = EXCLUDED.descriptor_oid,
             descriptor_public_id_attnum = EXCLUDED.descriptor_public_id_attnum,
             tensor_ref_attnum = EXCLUDED.tensor_ref_attnum,
             tensor_rows_attnum = EXCLUDED.tensor_rows_attnum,
             tensor_dim_attnum = EXCLUDED.tensor_dim_attnum,
             tensor_dtype_attnum = EXCLUDED.tensor_dtype_attnum,
             tensor_checksum_attnum = EXCLUDED.tensor_checksum_attnum,
             registered_by = EXCLUDED.registered_by,
             registered_at = pg_catalog.clock_timestamp()',
        ext_schema
    ) USING
        index_relation::oid,
        heap_oid,
        model_contract_id,
        normalized_storage,
        model_contract_attnum,
        public_id_attnum,
        descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum,
        caller_oid;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_register_maxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_register_maxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) TO PUBLIC;

CREATE FUNCTION vchordrq_unregister_maxsim_source(index_relation regclass)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    index_owner oid;
    removed_count bigint;
BEGIN
    IF index_relation IS NULL THEN
        RETURN false;
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    SELECT c.relowner INTO index_owner
    FROM pg_catalog.pg_class AS c
    WHERE c.oid = index_relation::oid AND c.relkind = 'i';
    IF index_owner IS NULL
       OR NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the index owner may unregister its MaxSim tensor source';
    END IF;

    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_maxsim_sources WHERE index_oid = $1',
        ext_schema
    ) USING index_relation::oid;
    GET DIAGNOSTICS removed_count = ROW_COUNT;
    RETURN removed_count > 0;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_unregister_maxsim_source(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_unregister_maxsim_source(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_maxsim_source_info(index_relation regclass)
RETURNS TABLE(
    registered_index regclass,
    heap_relation regclass,
    model_contract_id text,
    source_storage text,
    model_contract_column name,
    public_id_column name,
    descriptor_relation regclass,
    descriptor_public_id_column name,
    tensor_ref_column name,
    tensor_rows_column name,
    tensor_dim_column name,
    tensor_dtype_column name,
    tensor_checksum_column name
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    bound_heap_oid oid;
    live_heap_oid oid;
    index_owner oid;
    bound_model_contract_id text;
    bound_storage text;
    bound_descriptor_oid oid;
    tensor_relation_oid oid;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    valid_columns bigint;
    expected_columns bigint;
    resolved_model_contract_column name;
    resolved_public_id_column name;
    resolved_descriptor_public_id_column name;
    resolved_tensor_ref_column name;
    resolved_tensor_rows_column name;
    resolved_tensor_dim_column name;
    resolved_tensor_dtype_column name;
    resolved_tensor_checksum_column name;
BEGIN
    IF index_relation IS NULL THEN
        RAISE EXCEPTION 'index_relation must not be NULL';
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;
    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;

    EXECUTE pg_catalog.format(
        'SELECT heap_oid, model_contract_id, storage,
                model_contract_attnum, public_id_attnum,
                descriptor_oid, descriptor_public_id_attnum,
                tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
                tensor_dtype_attnum, tensor_checksum_attnum
         FROM %I._vchordrq_maxsim_sources
         WHERE index_oid = $1',
        ext_schema
    ) INTO
        bound_heap_oid,
        bound_model_contract_id,
        bound_storage,
        model_contract_attnum,
        public_id_attnum,
        bound_descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum
    USING index_relation::oid;
    IF bound_heap_oid IS NULL THEN
        RAISE EXCEPTION 'MaxSim tensor source is not registered for index %',
            index_relation;
    END IF;

    SELECT x.indrelid, i.relowner
    INTO live_heap_oid, index_owner
    FROM pg_catalog.pg_index AS x
    JOIN pg_catalog.pg_class AS i ON i.oid = x.indexrelid
    JOIN pg_catalog.pg_class AS h ON h.oid = x.indrelid
    JOIN pg_catalog.pg_am AS am ON am.oid = i.relam
    JOIN pg_catalog.pg_opclass AS opc ON opc.oid = x.indclass[0]
    WHERE x.indexrelid = index_relation::oid
      AND i.relkind = 'i'
      AND h.relkind IN ('r', 'm')
      AND am.amname = 'vchordrq'
      AND opc.opcname IN (
          'vector_maxsim_ops',
          'halfvec_maxsim_ops',
          'rabitq8_maxsim_ops',
          'rabitq4_maxsim_ops'
      )
      AND x.indisvalid
      AND x.indisready
      AND x.indnatts = 1
      AND x.indnkeyatts = 1;
    IF live_heap_oid IS NULL OR live_heap_oid <> bound_heap_oid THEN
        RAISE EXCEPTION 'registered MaxSim tensor source is stale or invalid';
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE')
       AND NOT pg_catalog.has_table_privilege(caller_oid, live_heap_oid, 'SELECT') THEN
        RAISE EXCEPTION 'permission denied for registered MaxSim tensor source';
    END IF;

    IF bound_storage NOT IN ('heap_array', 'external_ref', 'external_relation') THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid storage';
    END IF;

    SELECT count(*)
    INTO valid_columns
    FROM (
        VALUES
            (model_contract_attnum, 'text'::regtype),
            (public_id_attnum, 'bigint'::regtype)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = bound_heap_oid
     AND a.attnum = expected.attnum
     AND a.atttypid = expected.atttypid
     AND a.attnotnull
     AND NOT a.attisdropped;
    IF valid_columns <> 2 THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid heap columns';
    END IF;

    IF bound_storage = 'heap_array' THEN
        IF bound_descriptor_oid IS NOT NULL
           OR descriptor_public_id_attnum IS NOT NULL
           OR tensor_ref_attnum IS NOT NULL
           OR tensor_rows_attnum IS NOT NULL
           OR tensor_dim_attnum IS NOT NULL
           OR tensor_dtype_attnum IS NOT NULL
           OR tensor_checksum_attnum IS NOT NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid heap_array binding';
        END IF;
        tensor_relation_oid := NULL;
    ELSIF bound_storage = 'external_ref' THEN
        IF bound_descriptor_oid IS NOT NULL OR descriptor_public_id_attnum IS NOT NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid external_ref binding';
        END IF;
        tensor_relation_oid := bound_heap_oid;
    ELSE
        IF bound_descriptor_oid IS NULL OR descriptor_public_id_attnum IS NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid external_relation binding';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = bound_descriptor_oid
          AND c.relkind IN ('r', 'm');
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor relation is stale or invalid';
        END IF;
        IF NOT pg_catalog.has_table_privilege(caller_oid, bound_descriptor_oid, 'SELECT') THEN
            RAISE EXCEPTION 'SELECT privilege on the registered descriptor relation is required';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = bound_descriptor_oid
          AND a.attnum = descriptor_public_id_attnum
          AND a.atttypid = 'bigint'::regtype
          AND a.attnotnull
          AND NOT a.attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor public ID column is invalid';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_index AS x
        WHERE x.indrelid = bound_descriptor_oid
          AND x.indisunique
          AND x.indisvalid
          AND x.indisready
          AND x.indnkeyatts = 1
          AND x.indkey[0] = descriptor_public_id_attnum
          AND x.indexprs IS NULL
          AND x.indpred IS NULL;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor public ID is no longer unique';
        END IF;
        tensor_relation_oid := bound_descriptor_oid;
    END IF;

    expected_columns := CASE WHEN bound_storage = 'heap_array' THEN 0 ELSE 5 END;
    SELECT count(*)
    INTO valid_columns
    FROM (
        VALUES
            (tensor_ref_attnum, 'text'::regtype),
            (tensor_rows_attnum, 'integer'::regtype),
            (tensor_dim_attnum, 'integer'::regtype),
            (tensor_dtype_attnum, 'text'::regtype),
            (tensor_checksum_attnum, 'text'::regtype)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = tensor_relation_oid
     AND a.attnum = expected.attnum
     AND a.atttypid = expected.atttypid
     AND a.attnotnull
     AND NOT a.attisdropped
    WHERE expected.attnum IS NOT NULL;
    IF valid_columns <> expected_columns THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid descriptor columns';
    END IF;

    SELECT a.attname INTO resolved_model_contract_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_heap_oid AND a.attnum = model_contract_attnum;
    SELECT a.attname INTO resolved_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_heap_oid AND a.attnum = public_id_attnum;
    SELECT a.attname INTO resolved_descriptor_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_descriptor_oid AND a.attnum = descriptor_public_id_attnum;
    SELECT a.attname INTO resolved_tensor_ref_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_ref_attnum;
    SELECT a.attname INTO resolved_tensor_rows_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_rows_attnum;
    SELECT a.attname INTO resolved_tensor_dim_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dim_attnum;
    SELECT a.attname INTO resolved_tensor_dtype_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dtype_attnum;
    SELECT a.attname INTO resolved_tensor_checksum_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_checksum_attnum;

    RETURN QUERY SELECT
        index_relation,
        bound_heap_oid::regclass,
        bound_model_contract_id,
        bound_storage,
        resolved_model_contract_column,
        resolved_public_id_column,
        bound_descriptor_oid::regclass,
        resolved_descriptor_public_id_column,
        resolved_tensor_ref_column,
        resolved_tensor_rows_column,
        resolved_tensor_dim_column,
        resolved_tensor_dtype_column,
        resolved_tensor_checksum_column;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_maxsim_source_info(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_maxsim_source_info(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_maxsim_search(
    index_relation regclass,
    query anyarray,
    candidate_limit integer,
    top_k integer
)
RETURNS TABLE(public_id bigint, similarity real)
STRICT
VOLATILE
PARALLEL UNSAFE
LANGUAGE c
AS 'MODULE_PATHNAME', '_vchordrq_maxsim_search_external_wrapper';

COMMENT ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer)
IS 'Restricted Phase 3B external-tensor MaxSim search; returns exact similarity under caller MVCC, SELECT privileges, and PostgreSQL row visibility.';

REVOKE ALL ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer) TO PUBLIC;

CREATE FUNCTION _vchordrq_maxsim_source_sql_drop()
RETURNS event_trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    registry regclass;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RETURN;
    END IF;
    registry := pg_catalog.to_regclass(
        pg_catalog.format('%I._vchordrq_maxsim_sources', ext_schema)
    );
    IF registry IS NULL THEN
        RETURN;
    END IF;

    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_maxsim_sources AS s
         USING pg_catalog.pg_event_trigger_dropped_objects() AS d
         WHERE d.objid = s.index_oid
            OR (
                d.objid = s.heap_oid
                AND (
                    d.objsubid = 0
                    OR d.objsubid = s.model_contract_attnum
                    OR d.objsubid = s.public_id_attnum
                    OR (
                        s.storage = ''external_ref''
                        AND d.objsubid IN (
                            s.tensor_ref_attnum,
                            s.tensor_rows_attnum,
                            s.tensor_dim_attnum,
                            s.tensor_dtype_attnum,
                            s.tensor_checksum_attnum
                        )
                    )
                )
            )
            OR (
                d.objid = s.descriptor_oid
                AND (
                    d.objsubid = 0
                    OR d.objsubid = s.descriptor_public_id_attnum
                    OR d.objsubid IN (
                        s.tensor_ref_attnum,
                        s.tensor_rows_attnum,
                        s.tensor_dim_attnum,
                        s.tensor_dtype_attnum,
                        s.tensor_checksum_attnum
                    )
                )
            )',
        ext_schema
    );
END;
$$;

REVOKE ALL ON FUNCTION _vchordrq_maxsim_source_sql_drop() FROM PUBLIC;

CREATE EVENT TRIGGER _vchordrq_maxsim_source_sql_drop
ON sql_drop
EXECUTE FUNCTION _vchordrq_maxsim_source_sql_drop();

-- Exact TileMaxSim over a caller-scoped tensor set. This source registry is
-- deliberately relation-keyed rather than index-keyed: the caller owns ACL,
-- graph, and application filtering decisions. VectorChord accepts the full
-- visible ID set without an artificial count cap, loads its tensor pages, and
-- applies the configured GPU cache policy.

CREATE TABLE _vchordrq_tilemaxsim_sources (
    source_oid oid PRIMARY KEY,
    model_contract_id text NOT NULL CHECK (
        model_contract_id OPERATOR(pg_catalog.<>) ''::text
        AND pg_catalog.length(model_contract_id) OPERATOR(pg_catalog.<=) 512
    ),
    storage text NOT NULL CHECK (
        storage OPERATOR(pg_catalog.=) ANY (
            ARRAY['external_ref', 'external_relation']::text[]
        )
    ),
    model_contract_attnum smallint NOT NULL CHECK (
        model_contract_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    public_id_attnum smallint NOT NULL CHECK (
        public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    descriptor_oid oid,
    descriptor_public_id_attnum smallint,
    tensor_ref_attnum smallint NOT NULL CHECK (
        tensor_ref_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    tensor_rows_attnum smallint NOT NULL CHECK (
        tensor_rows_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    tensor_dim_attnum smallint NOT NULL CHECK (
        tensor_dim_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    tensor_dtype_attnum smallint NOT NULL CHECK (
        tensor_dtype_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    tensor_checksum_attnum smallint NOT NULL CHECK (
        tensor_checksum_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    registered_by oid NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        (
            storage OPERATOR(pg_catalog.=) 'external_ref'::text
            AND descriptor_oid IS NULL
            AND descriptor_public_id_attnum IS NULL
        )
        OR
        (
            storage OPERATOR(pg_catalog.=) 'external_relation'::text
            AND descriptor_oid IS NOT NULL
            AND descriptor_public_id_attnum IS NOT NULL
            AND descriptor_public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
        )
    )
);

REVOKE ALL ON TABLE _vchordrq_tilemaxsim_sources FROM PUBLIC;

CREATE FUNCTION vchordrq_register_tilemaxsim_source(
    source_relation regclass,
    model_contract_id text,
    storage text,
    model_contract_column name,
    public_id_column name,
    tensor_ref_column name,
    tensor_rows_column name,
    tensor_dim_column name,
    tensor_dtype_column name,
    tensor_checksum_column name,
    descriptor_relation regclass DEFAULT NULL,
    descriptor_public_id_column name DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    source_oid oid;
    source_owner oid;
    descriptor_oid oid;
    descriptor_owner oid;
    tensor_relation_oid oid;
    normalized_storage text;
    attnum smallint;
    atttypid oid;
    attnotnull boolean;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    public_id_is_unique boolean;
BEGIN
    IF source_relation IS NULL THEN
        RAISE EXCEPTION 'source_relation must not be NULL';
    END IF;
    IF model_contract_id IS NULL
       OR btrim(model_contract_id) = ''
       OR length(model_contract_id) > 512 THEN
        RAISE EXCEPTION 'model_contract_id must contain between 1 and 512 characters';
    END IF;
    model_contract_id := btrim(model_contract_id);
    IF model_contract_column IS NULL OR public_id_column IS NULL THEN
        RAISE EXCEPTION 'model_contract_column and public_id_column must not be NULL';
    END IF;
    IF tensor_ref_column IS NULL
       OR tensor_rows_column IS NULL
       OR tensor_dim_column IS NULL
       OR tensor_dtype_column IS NULL
       OR tensor_checksum_column IS NULL THEN
        RAISE EXCEPTION 'external tensor descriptor columns must not be NULL';
    END IF;

    normalized_storage := lower(btrim(storage));
    IF normalized_storage IS NULL
       OR normalized_storage NOT IN ('external_ref', 'external_relation') THEN
        RAISE EXCEPTION 'storage must be external_ref or external_relation';
    END IF;

    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    SELECT c.oid, c.relowner
    INTO source_oid, source_owner
    FROM pg_catalog.pg_class AS c
    WHERE c.oid = source_relation::oid
      AND c.relkind IN ('r', 'm');
    IF source_oid IS NULL THEN
        RAISE EXCEPTION 'source_relation % must be a table or materialized view', source_relation;
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, source_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the source relation owner may register a TileMaxSim tensor source';
    END IF;

    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = source_oid
      AND a.attname = model_contract_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'model contract column % must be a NOT NULL text column',
            model_contract_column;
    END IF;
    model_contract_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = source_oid
      AND a.attname = public_id_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL
       OR atttypid NOT IN ('integer'::regtype, 'bigint'::regtype)
       OR NOT attnotnull THEN
        RAISE EXCEPTION 'public ID column % must be a NOT NULL integer or bigint column',
            public_id_column;
    END IF;
    public_id_attnum := attnum;
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_index AS x
        WHERE x.indrelid = source_oid
          AND x.indisunique
          AND x.indisvalid
          AND x.indisready
          AND x.indnkeyatts = 1
          AND x.indkey[0] = public_id_attnum
          AND x.indexprs IS NULL
          AND x.indpred IS NULL
    ) INTO public_id_is_unique;
    IF NOT public_id_is_unique THEN
        RAISE EXCEPTION 'public ID column % must have a non-partial single-key unique index',
            public_id_column;
    END IF;

    IF normalized_storage = 'external_relation' THEN
        IF descriptor_relation IS NULL OR descriptor_public_id_column IS NULL THEN
            RAISE EXCEPTION 'external_relation sources require descriptor_relation and descriptor_public_id_column';
        END IF;
        SELECT c.oid, c.relowner
        INTO descriptor_oid, descriptor_owner
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = descriptor_relation::oid
          AND c.relkind IN ('r', 'm');
        IF descriptor_oid IS NULL THEN
            RAISE EXCEPTION 'descriptor relation % must be a table or materialized view',
                descriptor_relation;
        END IF;
        IF NOT pg_catalog.pg_has_role(caller_oid, descriptor_owner, 'USAGE') THEN
            RAISE EXCEPTION 'only the descriptor relation owner may register it as a TileMaxSim tensor source';
        END IF;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = descriptor_oid
          AND a.attname = descriptor_public_id_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL
           OR atttypid NOT IN ('integer'::regtype, 'bigint'::regtype)
           OR NOT attnotnull THEN
            RAISE EXCEPTION 'descriptor public ID column % must be a NOT NULL integer or bigint column',
                descriptor_public_id_column;
        END IF;
        descriptor_public_id_attnum := attnum;
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index AS x
            WHERE x.indrelid = descriptor_oid
              AND x.indisunique
              AND x.indisvalid
              AND x.indisready
              AND x.indnkeyatts = 1
              AND x.indkey[0] = descriptor_public_id_attnum
              AND x.indexprs IS NULL
              AND x.indpred IS NULL
        ) INTO public_id_is_unique;
        IF NOT public_id_is_unique THEN
            RAISE EXCEPTION 'descriptor public ID column % must have a non-partial single-key unique index',
                descriptor_public_id_column;
        END IF;
        tensor_relation_oid := descriptor_oid;
    ELSE
        IF descriptor_relation IS NOT NULL OR descriptor_public_id_column IS NOT NULL THEN
            RAISE EXCEPTION 'external_ref sources must not specify a descriptor relation';
        END IF;
        tensor_relation_oid := source_oid;
    END IF;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid
      AND a.attname = tensor_ref_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'tensor ref column % must be a NOT NULL text column', tensor_ref_column;
    END IF;
    tensor_ref_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid
      AND a.attname = tensor_rows_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'tensor rows column % must be a NOT NULL integer column', tensor_rows_column;
    END IF;
    tensor_rows_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid
      AND a.attname = tensor_dim_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'tensor dim column % must be a NOT NULL integer column', tensor_dim_column;
    END IF;
    tensor_dim_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid
      AND a.attname = tensor_dtype_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'tensor dtype column % must be a NOT NULL text column', tensor_dtype_column;
    END IF;
    tensor_dtype_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid
      AND a.attname = tensor_checksum_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'tensor checksum column % must be a NOT NULL text column',
            tensor_checksum_column;
    END IF;
    tensor_checksum_attnum := attnum;

    IF (CASE WHEN normalized_storage = 'external_ref' THEN 7 ELSE 6 END) <> (
        SELECT count(DISTINCT u.attnum)
        FROM pg_catalog.unnest(ARRAY[
            CASE WHEN normalized_storage = 'external_ref' THEN model_contract_attnum ELSE descriptor_public_id_attnum END,
            CASE WHEN normalized_storage = 'external_ref' THEN public_id_attnum ELSE NULL END,
            tensor_ref_attnum,
            tensor_rows_attnum,
            tensor_dim_attnum,
            tensor_dtype_attnum,
            tensor_checksum_attnum
        ]) AS u(attnum)
    ) THEN
        RAISE EXCEPTION 'TileMaxSim tensor source columns must be distinct';
    END IF;

    EXECUTE pg_catalog.format(
        'INSERT INTO %I._vchordrq_tilemaxsim_sources (
             source_oid, model_contract_id, storage,
             model_contract_attnum, public_id_attnum,
             descriptor_oid, descriptor_public_id_attnum,
             tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
             tensor_dtype_attnum, tensor_checksum_attnum, registered_by
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
         ON CONFLICT (source_oid) DO UPDATE SET
             model_contract_id = EXCLUDED.model_contract_id,
             storage = EXCLUDED.storage,
             model_contract_attnum = EXCLUDED.model_contract_attnum,
             public_id_attnum = EXCLUDED.public_id_attnum,
             descriptor_oid = EXCLUDED.descriptor_oid,
             descriptor_public_id_attnum = EXCLUDED.descriptor_public_id_attnum,
             tensor_ref_attnum = EXCLUDED.tensor_ref_attnum,
             tensor_rows_attnum = EXCLUDED.tensor_rows_attnum,
             tensor_dim_attnum = EXCLUDED.tensor_dim_attnum,
             tensor_dtype_attnum = EXCLUDED.tensor_dtype_attnum,
             tensor_checksum_attnum = EXCLUDED.tensor_checksum_attnum,
             registered_by = EXCLUDED.registered_by,
             registered_at = pg_catalog.clock_timestamp()',
        ext_schema
    ) USING
        source_oid,
        model_contract_id,
        normalized_storage,
        model_contract_attnum,
        public_id_attnum,
        descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum,
        caller_oid;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_register_tilemaxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_register_tilemaxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) TO PUBLIC;

CREATE FUNCTION vchordrq_unregister_tilemaxsim_source(source_relation regclass)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    source_owner oid;
    removed_count bigint;
BEGIN
    IF source_relation IS NULL THEN
        RETURN false;
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    SELECT c.relowner INTO source_owner
    FROM pg_catalog.pg_class AS c
    WHERE c.oid = source_relation::oid AND c.relkind IN ('r', 'm');
    IF ext_schema IS NULL
       OR source_owner IS NULL
       OR NOT pg_catalog.pg_has_role(caller_oid, source_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the source relation owner may unregister its TileMaxSim source';
    END IF;
    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_tilemaxsim_sources WHERE source_oid = $1',
        ext_schema
    ) USING source_relation::oid;
    GET DIAGNOSTICS removed_count = ROW_COUNT;
    RETURN removed_count > 0;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_unregister_tilemaxsim_source(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_unregister_tilemaxsim_source(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_tilemaxsim_source_info(source_relation regclass)
RETURNS TABLE(
    registered_source regclass,
    model_contract_id text,
    source_storage text,
    model_contract_column name,
    public_id_column name,
    descriptor_relation regclass,
    descriptor_public_id_column name,
    tensor_ref_column name,
    tensor_rows_column name,
    tensor_dim_column name,
    tensor_dtype_column name,
    tensor_checksum_column name
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    source_oid oid;
    source_owner oid;
    bound_model_contract_id text;
    bound_storage text;
    descriptor_oid oid;
    tensor_relation_oid oid;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    valid_columns bigint;
    resolved_model_contract_column name;
    resolved_public_id_column name;
    resolved_descriptor_public_id_column name;
    resolved_tensor_ref_column name;
    resolved_tensor_rows_column name;
    resolved_tensor_dim_column name;
    resolved_tensor_dtype_column name;
    resolved_tensor_checksum_column name;
BEGIN
    IF source_relation IS NULL THEN
        RAISE EXCEPTION 'source_relation must not be NULL';
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    EXECUTE pg_catalog.format(
        'SELECT source_oid, model_contract_id, storage,
                model_contract_attnum, public_id_attnum,
                descriptor_oid, descriptor_public_id_attnum,
                tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
                tensor_dtype_attnum, tensor_checksum_attnum
           FROM %I._vchordrq_tilemaxsim_sources
          WHERE source_oid = $1',
        ext_schema
    ) INTO
        source_oid,
        bound_model_contract_id,
        bound_storage,
        model_contract_attnum,
        public_id_attnum,
        descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum
    USING source_relation::oid;
    IF source_oid IS NULL THEN
        RAISE EXCEPTION 'TileMaxSim tensor source is not registered for relation %', source_relation;
    END IF;

    SELECT c.relowner INTO source_owner
    FROM pg_catalog.pg_class AS c
    WHERE c.oid = source_oid AND c.relkind IN ('r', 'm');
    IF source_owner IS NULL THEN
        RAISE EXCEPTION 'registered TileMaxSim source relation is stale or invalid';
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, source_owner, 'USAGE')
       AND NOT pg_catalog.has_table_privilege(caller_oid, source_oid, 'SELECT') THEN
        RAISE EXCEPTION 'permission denied for registered TileMaxSim source';
    END IF;

    SELECT count(*) INTO valid_columns
    FROM (
        VALUES
            (model_contract_attnum, 'text'::regtype),
            (public_id_attnum, NULL::oid)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = source_oid
     AND a.attnum = expected.attnum
     AND (expected.atttypid IS NULL OR a.atttypid = expected.atttypid)
     AND a.attnotnull
     AND NOT a.attisdropped
    WHERE expected.atttypid IS NOT NULL
       OR a.atttypid IN ('integer'::regtype, 'bigint'::regtype);
    IF valid_columns <> 2 THEN
        RAISE EXCEPTION 'registered TileMaxSim source columns are invalid';
    END IF;
    PERFORM 1
    FROM pg_catalog.pg_index AS x
    WHERE x.indrelid = source_oid
      AND x.indisunique
      AND x.indisvalid
      AND x.indisready
      AND x.indnkeyatts = 1
      AND x.indkey[0] = public_id_attnum
      AND x.indexprs IS NULL
      AND x.indpred IS NULL;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'registered TileMaxSim source public ID is no longer unique';
    END IF;

    IF bound_storage = 'external_ref' THEN
        IF descriptor_oid IS NOT NULL OR descriptor_public_id_attnum IS NOT NULL THEN
            RAISE EXCEPTION 'registered external_ref TileMaxSim binding is invalid';
        END IF;
        tensor_relation_oid := source_oid;
    ELSIF bound_storage = 'external_relation' THEN
        IF descriptor_oid IS NULL OR descriptor_public_id_attnum IS NULL THEN
            RAISE EXCEPTION 'registered external_relation TileMaxSim binding is invalid';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = descriptor_oid AND c.relkind IN ('r', 'm');
        IF NOT FOUND OR NOT pg_catalog.has_table_privilege(caller_oid, descriptor_oid, 'SELECT') THEN
            RAISE EXCEPTION 'registered TileMaxSim descriptor relation is unavailable';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = descriptor_oid
          AND a.attnum = descriptor_public_id_attnum
          AND a.atttypid IN ('integer'::regtype, 'bigint'::regtype)
          AND a.attnotnull
          AND NOT a.attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered TileMaxSim descriptor public ID is invalid';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_index AS x
        WHERE x.indrelid = descriptor_oid
          AND x.indisunique
          AND x.indisvalid
          AND x.indisready
          AND x.indnkeyatts = 1
          AND x.indkey[0] = descriptor_public_id_attnum
          AND x.indexprs IS NULL
          AND x.indpred IS NULL;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered TileMaxSim descriptor public ID is no longer unique';
        END IF;
        tensor_relation_oid := descriptor_oid;
    ELSE
        RAISE EXCEPTION 'registered TileMaxSim storage is invalid';
    END IF;

    SELECT count(*) INTO valid_columns
    FROM (
        VALUES
            (tensor_ref_attnum, 'text'::regtype),
            (tensor_rows_attnum, 'integer'::regtype),
            (tensor_dim_attnum, 'integer'::regtype),
            (tensor_dtype_attnum, 'text'::regtype),
            (tensor_checksum_attnum, 'text'::regtype)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = tensor_relation_oid
     AND a.attnum = expected.attnum
     AND a.atttypid = expected.atttypid
     AND a.attnotnull
     AND NOT a.attisdropped;
    IF valid_columns <> 5 THEN
        RAISE EXCEPTION 'registered TileMaxSim tensor descriptor columns are invalid';
    END IF;

    SELECT a.attname INTO resolved_model_contract_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = source_oid AND a.attnum = model_contract_attnum;
    SELECT a.attname INTO resolved_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = source_oid AND a.attnum = public_id_attnum;
    SELECT a.attname INTO resolved_descriptor_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = descriptor_oid AND a.attnum = descriptor_public_id_attnum;
    SELECT a.attname INTO resolved_tensor_ref_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_ref_attnum;
    SELECT a.attname INTO resolved_tensor_rows_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_rows_attnum;
    SELECT a.attname INTO resolved_tensor_dim_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dim_attnum;
    SELECT a.attname INTO resolved_tensor_dtype_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dtype_attnum;
    SELECT a.attname INTO resolved_tensor_checksum_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_checksum_attnum;

    RETURN QUERY SELECT
        source_oid::regclass,
        bound_model_contract_id,
        bound_storage,
        resolved_model_contract_column,
        resolved_public_id_column,
        descriptor_oid::regclass,
        resolved_descriptor_public_id_column,
        resolved_tensor_ref_column,
        resolved_tensor_rows_column,
        resolved_tensor_dim_column,
        resolved_tensor_dtype_column,
        resolved_tensor_checksum_column;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_tilemaxsim_source_info(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_tilemaxsim_source_info(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_tilemaxsim_rerank(
    source_relation regclass,
    query vector[],
    candidate_ids bigint[],
    top_k integer
)
RETURNS TABLE(public_id bigint, similarity real)
STRICT
VOLATILE
PARALLEL UNSAFE
LANGUAGE c
AS 'MODULE_PATHNAME', '_vchordrq_tilemaxsim_rerank_vector_wrapper';

CREATE FUNCTION vchordrq_tilemaxsim_rerank(
    source_relation regclass,
    query halfvec[],
    candidate_ids bigint[],
    top_k integer
)
RETURNS TABLE(public_id bigint, similarity real)
STRICT
VOLATILE
PARALLEL UNSAFE
LANGUAGE c
AS 'MODULE_PATHNAME', '_vchordrq_tilemaxsim_rerank_halfvec_wrapper';

COMMENT ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, vector[], bigint[], integer)
IS 'Exact TileMaxSim over a caller-scoped ID set without an artificial candidate-count cap. ACL and application filtering remain the caller responsibility.';
COMMENT ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, halfvec[], bigint[], integer)
IS 'Exact TileMaxSim over a caller-scoped ID set without an artificial candidate-count cap. ACL and application filtering remain the caller responsibility.';

REVOKE ALL ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, vector[], bigint[], integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, halfvec[], bigint[], integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, vector[], bigint[], integer) TO PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_tilemaxsim_rerank(regclass, halfvec[], bigint[], integer) TO PUBLIC;

CREATE FUNCTION _vchordrq_tilemaxsim_source_sql_drop()
RETURNS event_trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL OR pg_catalog.to_regclass(
        pg_catalog.format('%I._vchordrq_tilemaxsim_sources', ext_schema)
    ) IS NULL THEN
        RETURN;
    END IF;
    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_tilemaxsim_sources AS s
         USING pg_catalog.pg_event_trigger_dropped_objects() AS d
         WHERE (
             d.objid = s.source_oid
             AND (
                 d.objsubid = 0
                 OR d.objsubid IN (
                     s.model_contract_attnum,
                     s.public_id_attnum,
                     CASE WHEN s.storage = ''external_ref'' THEN s.tensor_ref_attnum END,
                     CASE WHEN s.storage = ''external_ref'' THEN s.tensor_rows_attnum END,
                     CASE WHEN s.storage = ''external_ref'' THEN s.tensor_dim_attnum END,
                     CASE WHEN s.storage = ''external_ref'' THEN s.tensor_dtype_attnum END,
                     CASE WHEN s.storage = ''external_ref'' THEN s.tensor_checksum_attnum END
                 )
             )
         )
         OR (
             d.objid = s.descriptor_oid
             AND (
                 d.objsubid = 0
                 OR d.objsubid IN (
                     s.descriptor_public_id_attnum,
                     s.tensor_ref_attnum,
                     s.tensor_rows_attnum,
                     s.tensor_dim_attnum,
                     s.tensor_dtype_attnum,
                     s.tensor_checksum_attnum
                 )
             )
         )',
        ext_schema
    );
END;
$$;

REVOKE ALL ON FUNCTION _vchordrq_tilemaxsim_source_sql_drop() FROM PUBLIC;

CREATE EVENT TRIGGER _vchordrq_tilemaxsim_source_sql_drop
ON sql_drop
EXECUTE FUNCTION _vchordrq_tilemaxsim_source_sql_drop();
