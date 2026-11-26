# PostgreSQL runtime build

`docker build -f services/Dockerfile.postgres -t vectorchord-postgres:pg18 .` builds PostgreSQL 18.7 with pgvector 0.8.5 and this VectorChord source revision. The official PostgreSQL base is digest-pinned, and pgvector is pinned to its source commit. The multi-stage image includes no compiler toolchain.

The PG18 official entrypoint uses `/var/lib/postgresql/18/docker`; mount persistent storage at `/var/lib/postgresql`. A major image change does not upgrade an existing cluster. Verify an independent backup, rehearse an isolated restore or `pg_upgrade --copy`, preload vchord in both temporary servers and verify indexes, extension owners, grants and RLS before cutover. Never attach an old root-layout PG16 volume as though it were an empty PG18 volume. Deployment startup must reject that layout.

Publish reviewed builds to your registry and deploy by digest. No registry publication is implied by a local image tag. Existing Cargo PG14–PG18 feature support remains available; this file changes the packaged default, not the extension compatibility matrix. Override both build/runtime version inputs only with a matching pgvector build recipe.
