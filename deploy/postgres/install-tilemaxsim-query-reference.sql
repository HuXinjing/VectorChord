\set ON_ERROR_STOP on

BEGIN;
SET LOCAL lock_timeout = '3s';
SET LOCAL search_path = pg_catalog;

-- Existing appliances already have vchord 1.2.0 installed. The new library
-- must be in place before this script runs. Grant EXECUTE separately to the
-- NeoBrain search role before enabling its private encoder endpoint.
DO $install$
DECLARE
    extension_id oid;
    extension_schema name;
    function_id oid;
    signature constant text :=
        'vchordrq_tilemaxsim_rerank_ref(regclass, text, integer, integer, bigint[], integer)';
BEGIN
    SELECT ext.oid, ns.nspname INTO STRICT extension_id, extension_schema
      FROM pg_extension AS ext
      JOIN pg_namespace AS ns ON ns.oid = ext.extnamespace
     WHERE ext.extname = 'vchord' AND ext.extversion = '1.2.0';

    PERFORM pg_advisory_xact_lock(extension_id::bigint);
    function_id := to_regprocedure(format('%I.%s', extension_schema, signature));
    IF function_id IS NULL THEN
        EXECUTE format(
            'CREATE FUNCTION %I.%s
             RETURNS TABLE(public_id bigint, similarity real)
             STRICT VOLATILE PARALLEL UNSAFE LANGUAGE c
             AS %L, %L',
            extension_schema, signature, '$libdir/vchord',
            '_vchordrq_tilemaxsim_rerank_ref_wrapper'
        );
        EXECUTE format('ALTER EXTENSION vchord ADD FUNCTION %I.%s', extension_schema, signature);
        EXECUTE format(
            'COMMENT ON FUNCTION %I.%s IS %L',
            extension_schema, signature,
            'Exact FP16 over the current role-scoped candidates with an opaque CUDA IPC query reference.'
        );
        EXECUTE format('REVOKE ALL ON FUNCTION %I.%s FROM PUBLIC', extension_schema, signature);
    ELSIF NOT EXISTS (
        SELECT FROM pg_depend AS dependency
         WHERE dependency.classid = 'pg_proc'::regclass
           AND dependency.objid = function_id
           AND dependency.refclassid = 'pg_extension'::regclass
           AND dependency.refobjid = extension_id
           AND dependency.deptype = 'e'
    ) THEN
        RAISE EXCEPTION 'TileMaxSim query reference function already exists outside the vchord extension';
    END IF;
END
$install$;
COMMIT;
