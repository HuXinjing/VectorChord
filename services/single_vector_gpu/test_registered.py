"""Opt-in tests on a disposable PostgreSQL database and a dedicated reader role.
The default CPU device is a functional test, never a GPU performance claim.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psycopg

from single_vector_gpu.registered import Broker, EPOCH_SQL, Proof


CONFIG = os.environ.get("NEOBRAIN_VECTOR_GPU_TEST_CONFIG")


@unittest.skipUnless(CONFIG, "disposable PostgreSQL configuration not provided")
class RegisteredTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(CONFIG).read_text())
        self.source = "gpu-fixture-"+uuid.uuid4().hex[:12]
        self.owner = psycopg.connect(self.config["admin"], autocommit=True, application_name="neobrain-schema-writer-v161")
        self.owner.execute("INSERT INTO sources(id,name)VALUES(%s,%s)", (self.source,self.source))
        page = self.owner.execute("INSERT INTO pages(source_id,slug,type,title)VALUES(%s,'fixture','document','Fixture')RETURNING id", (self.source,)).fetchone()[0]
        vector = '['+','.join(['1']+['0']*1535)+']'
        self.identity = self.owner.execute("""INSERT INTO content_chunks(page_id,chunk_index,chunk_text,embedding,model)
          VALUES(%s,0,'Private fixture',%s::vector,'encoder-v1')RETURNING id""", (page,vector)).fetchone()[0]
        definition = json.dumps({"version":1,"tenant":"fixture-tenant","sources":[self.source],"model":"encoder-v1",
            "dimensions":1536,"candidate_k":1,"select":"SELECT cc.id,cc.embedding FROM content_chunks cc JOIN pages p ON p.id=cc.page_id WHERE p.source_id=$1",
            "params":[self.source],"search":{},"ranking":"","settings":{}},separators=(',',':'))
        self.manifest = hashlib.sha256(definition.encode()).hexdigest()
        self.owner.execute("""INSERT INTO knowledge_vector_gpu_manifests(manifest_id,tenant_key,definition,source_ids)
          VALUES(%s,'fixture-tenant',%s,%s)""", (self.manifest,definition,[self.source]))
        self.owner.execute("""INSERT INTO retrieval_capacity_groups VALUES(%s,16,16) ON CONFLICT DO NOTHING""", (self.config['resource_group'],))
        self.token = str(uuid.uuid4())
        self.owner.execute("""INSERT INTO retrieval_capacity_leases(token,resource_group,tenant_key,weight,expires_at,gpu_borrowable)
          VALUES(%s,%s,'fixture-tenant',1,now()+interval '1 minute',true)""", (self.token,self.config['resource_group']))
        self.epoch = self.owner.execute(EPOCH_SQL, ([self.source],)).fetchone()[0]
        self.broker = Broker(self.config)
        manifest = self.broker.manifest(self.manifest)
        end = time.monotonic()+8
        while not self.broker.prepare(manifest,self.epoch):
            if time.monotonic()>end: self.fail("snapshot did not load")
            time.sleep(.02)
        self.metadata = {"version":1,"manifestId":self.manifest,"epoch":self.epoch,"leases":[self.token],
                         "transport":self.config['transport'],"rows":1,"k":1}
        self.query = np.array([[1]+[0]*1535], dtype='<f4').tobytes()

    def tearDown(self):
        self.broker.close()
        self.owner.execute("DELETE FROM retrieval_capacity_leases WHERE token=%s", (self.token,))
        self.owner.execute("DELETE FROM knowledge_vector_gpu_manifests WHERE manifest_id=%s", (self.manifest,))
        self.owner.execute("DELETE FROM sources WHERE id=%s", (self.source,))
        self.owner.close()

    def test_loaded_snapshot_uses_colocated_vectors_and_real_borrowed_lease(self):
        result = self.broker.search(self.metadata,self.query,force=True)
        self.assertEqual(result['results'][0]['backend'],'gpu')
        self.assertEqual(result['results'][0]['candidates'][0]['id'],self.identity)
        self.assertEqual(self.owner.execute('SELECT count(*) FROM knowledge_vector_gpu_borrows WHERE parent_token=%s',(self.token,)).fetchone()[0],0)

    def test_reader_cannot_read_document_text_or_mutate_native_vectors(self):
        with psycopg.connect(self.config['database'],autocommit=True, application_name="neobrain-schema-writer-v161")as reader:
            with self.assertRaises(psycopg.errors.InsufficientPrivilege):reader.execute('SELECT chunk_text FROM content_chunks')
            with self.assertRaises(psycopg.errors.InsufficientPrivilege):reader.execute('DELETE FROM content_chunks')
            with self.assertRaises(psycopg.errors.InsufficientPrivilege):
                reader.execute("SELECT public.neobrain_retrieval_capacity_claim('forbidden','tenant',ARRAY['new-token'],1,1,1,1000,false,true)")

    def test_old_or_foreign_parent_is_not_borrowable(self):
        for sql in ["UPDATE retrieval_capacity_leases SET gpu_borrowable=false WHERE token=%s",
                    "UPDATE retrieval_capacity_leases SET gpu_borrowable=true,tenant_key='other-tenant' WHERE token=%s"]:
            self.owner.execute(sql,(self.token,))
            with self.assertRaises(PermissionError):self.broker.search(self.metadata,self.query,force=True)

    def test_partial_foreign_batch_rolls_back_all_child_borrows(self):
        metadata={**self.metadata,'rows':2,'leases':[self.token,str(uuid.uuid4())]}
        with self.assertRaises(PermissionError):self.broker.search(metadata,self.query*2,force=True)
        self.assertEqual(self.owner.execute('SELECT count(*) FROM knowledge_vector_gpu_borrows WHERE parent_token=%s',(self.token,)).fetchone()[0],0)

    def test_final_delivery_rechecks_even_with_a_cached_cohort_receipt(self):
        manifest=self.broker.manifest(self.manifest)
        proof=Proof(self.broker.pool,manifest,self.epoch,[self.token],self.config['resource_group'])
        try:
            self.owner.execute('UPDATE retrieval_capacity_leases SET closing=true WHERE token=%s',(self.token,))
            with self.assertRaises(PermissionError):proof.check(force=True)
        finally:proof.release()

    def test_upstream_change_blocks_old_snapshot_even_before_reload(self):
        self.owner.execute('DELETE FROM content_chunks WHERE id=%s',(self.identity,))
        with self.assertRaises(RuntimeError):self.broker.search(self.metadata,self.query,force=True)

    def test_parent_release_retains_capacity_until_child_computation_finishes(self):
        entered,release=threading.Event(),threading.Event();original=self.broker.dispatch.gpu
        def blocked(*args):
            entered.set();release.wait(3);return original(*args)
        self.broker.dispatch.gpu=blocked
        with ThreadPoolExecutor(max_workers=1)as executor:
            running=executor.submit(self.broker.search,self.metadata,self.query,True)
            self.assertTrue(entered.wait(2))
            try:
                with self.owner.transaction():
                    self.owner.execute('UPDATE retrieval_capacity_leases SET closing=true WHERE token=%s',(self.token,))
                    self.owner.execute('''DELETE FROM retrieval_capacity_leases p WHERE token=%s AND
                      NOT EXISTS(SELECT 1 FROM knowledge_vector_gpu_borrows b WHERE b.parent_token=p.token)''',(self.token,))
                self.assertEqual(self.owner.execute('SELECT count(*) FROM retrieval_capacity_leases WHERE token=%s',(self.token,)).fetchone()[0],1)
            finally:release.set()
            with self.assertRaises(PermissionError):running.result(3)
        self.assertEqual(self.owner.execute('SELECT count(*) FROM retrieval_capacity_leases WHERE token=%s',(self.token,)).fetchone()[0],0)

    def test_expired_manifest_cleanup_retains_inflight_references(self):
        manifest=self.broker.manifest(self.manifest)
        proof=Proof(self.broker.pool,manifest,self.epoch,[self.token],self.config['resource_group'])
        self.owner.execute("UPDATE knowledge_vector_gpu_manifests SET expires_at=now()-interval '1 second' WHERE manifest_id=%s",(self.manifest,))
        try:
            with self.broker.pool.connection()as conn:conn.execute('SELECT knowledge_vector_gpu_compact()')
            self.assertEqual(self.owner.execute('SELECT count(*) FROM knowledge_vector_gpu_manifests WHERE manifest_id=%s',(self.manifest,)).fetchone()[0],1)
        finally:proof.release()
        with self.broker.pool.connection()as conn:conn.execute('SELECT knowledge_vector_gpu_compact()')
        self.assertEqual(self.owner.execute('SELECT count(*) FROM knowledge_vector_gpu_manifests WHERE manifest_id=%s',(self.manifest,)).fetchone()[0],0)

    def test_witnesses_see_out_of_order_commits_and_compaction_preserves_epoch(self):
        before=self.owner.execute(EPOCH_SQL,([self.source],)).fetchone()[0]
        with psycopg.connect(self.config['admin'], application_name="neobrain-schema-writer-v161")as first,psycopg.connect(self.config['admin'], application_name="neobrain-schema-writer-v161")as second:
            for conn in [first,second]:conn.execute('INSERT INTO knowledge_vector_gpu_changes VALUES(%s,txid_current(),1)',(self.source,))
            second.commit()
            middle=self.owner.execute(EPOCH_SQL,([self.source],)).fetchone()[0]
            first.commit()
        after=self.owner.execute(EPOCH_SQL,([self.source],)).fetchone()[0]
        self.assertNotEqual(before,middle);self.assertNotEqual(middle,after)
        with psycopg.connect(self.config['database'],autocommit=True, application_name="neobrain-schema-writer-v161")as reader:reader.execute('SELECT knowledge_vector_gpu_compact()')
        self.assertEqual(self.owner.execute(EPOCH_SQL,([self.source],)).fetchone()[0],after)


if __name__ == '__main__':unittest.main()
