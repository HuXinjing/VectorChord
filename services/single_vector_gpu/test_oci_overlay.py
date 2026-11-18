import gzip
import io
import json
import tarfile
import unittest

from single_vector_gpu.oci_overlay import build, digest, encoded


class OverlayTest(unittest.TestCase):
    def base(self):
        config = encoded({'os':'linux','architecture':'amd64','rootfs':{'type':'layers','diff_ids':['sha256:'+'1'*64]},
            'config':{'User':'10001:10001','Entrypoint':['python3','-m','single_vector_gpu.registered'],
                      'Env':['OMP_NUM_THREADS=4'],'Healthcheck':{'Test':['CMD','health']}},
            'history':[{'created_by':'base'}]})
        manifest = encoded({'schemaVersion':2,'mediaType':'application/vnd.oci.image.manifest.v1+json',
            'config':{'digest':digest(config),'size':len(config)},
            'layers':[{'digest':'sha256:'+'2'*64,'size':3,'mediaType':'application/vnd.oci.image.layer.v1.tar+gzip'}]})
        return manifest,config

    def test_preserves_base_runtime_and_layers_and_adds_only_the_code_layer(self):
        manifest,config = self.base()
        result = build(manifest,config,b'new source\n','a'*40,'example/broker:release')
        self.assertEqual(result,build(manifest,config,b'new source\n','a'*40,'example/broker:release'))
        with tarfile.open(fileobj=io.BytesIO(result)) as archive:
            read=lambda name:archive.extractfile(name).read()
            index=json.loads(read('index.json'))
            descriptor=index['manifests'][0]
            updated_manifest_bytes=read('blobs/sha256/'+descriptor['digest'].split(':')[1])
            self.assertEqual(digest(updated_manifest_bytes),descriptor['digest'])
            updated_manifest=json.loads(updated_manifest_bytes)
            self.assertEqual(updated_manifest['layers'][:-1],json.loads(manifest)['layers'])
            updated_config=json.loads(read('blobs/sha256/'+updated_manifest['config']['digest'].split(':')[1]))
            for key,value in json.loads(config)['config'].items():
                self.assertEqual(updated_config['config'][key],value)
            layer=read('blobs/sha256/'+updated_manifest['layers'][-1]['digest'].split(':')[1])
            self.assertEqual(updated_config['rootfs']['diff_ids'][-1],digest(gzip.decompress(layer)))
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(layer))) as patch:
                self.assertEqual(patch.getnames(),['app/single_vector_gpu/registered.py'])
                self.assertEqual(patch.extractfile(patch.getnames()[0]).read(),b'new source\n')

    def test_rejects_unverified_config_revision_and_platform(self):
        manifest,config=self.base()
        with self.assertRaises(ValueError):build(manifest,config+b' ',b'x','a'*40,'example/broker:release')
        with self.assertRaises(ValueError):build(manifest,config,b'x','latest','example/broker:release')
        changed=json.loads(config);changed['architecture']='arm64';changed=encoded(changed)
        parent=json.loads(manifest);parent['config']['digest']=digest(changed)
        with self.assertRaises(ValueError):build(encoded(parent),changed,b'x','a'*40,'example/broker:release')
