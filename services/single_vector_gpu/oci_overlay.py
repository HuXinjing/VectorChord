"""Build a deterministic code layer against an immutable, already-cached OCI base.

The resulting archive requires base blobs to exist in the importing content store.
It deliberately carries no credentials, dependencies or replacement CUDA layers.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return 'sha256:'+hashlib.sha256(value).hexdigest()


def member(archive, name, value):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(value), 0o644, 0
    archive.addfile(info, io.BytesIO(value))


def build(base_manifest, base_config, source, revision, reference):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('full_source_revision_required')
    manifest = json.loads(base_manifest)
    if digest(base_config) != manifest['config']['digest']:
        raise ValueError('base_config_digest_mismatch')
    config = copy.deepcopy(json.loads(base_config))
    if config.get('os') != 'linux' or config.get('architecture') != 'amd64':
        raise ValueError('unsupported_base_platform')
    if len(config['rootfs']['diff_ids']) != len(manifest['layers']):
        raise ValueError('base_layer_count_mismatch')
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        member(archive, 'app/single_vector_gpu/registered.py', source)
    uncompressed = raw.getvalue()
    layer = gzip.compress(uncompressed, mtime=0)
    config['rootfs']['diff_ids'].append(digest(uncompressed))
    config.setdefault('history', []).append({'created_by':'COPY single_vector_gpu/registered.py /app/single_vector_gpu/registered.py'})
    labels = config.setdefault('config', {}).setdefault('Labels', {})
    labels.update({'org.opencontainers.image.revision':revision,
                   'org.opencontainers.image.base.digest':digest(base_manifest),
                   'neoclaw.broker.source.sha256':hashlib.sha256(source).hexdigest()})
    config_bytes = encoded(config)
    manifest = copy.deepcopy(manifest)
    manifest['config'] = {'mediaType':'application/vnd.oci.image.config.v1+json',
                          'digest':digest(config_bytes),'size':len(config_bytes)}
    manifest['layers'].append({'mediaType':'application/vnd.oci.image.layer.v1.tar+gzip',
                               'digest':digest(layer),'size':len(layer)})
    manifest_bytes = encoded(manifest)
    index = {'schemaVersion':2,'manifests':[{'mediaType':manifest['mediaType'],
             'digest':digest(manifest_bytes),'size':len(manifest_bytes),
             'platform':{'architecture':'amd64','os':'linux'},
             'annotations':{'io.containerd.image.name':reference,'org.opencontainers.image.ref.name':reference}}]}
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode='w') as archive:
        member(archive, 'oci-layout', encoded({'imageLayoutVersion':'1.0.0'}))
        member(archive, 'index.json', encoded(index))
        for value in (config_bytes, layer, manifest_bytes):
            member(archive, 'blobs/sha256/'+digest(value).split(':')[1], value)
    return result.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','config','source','revision','reference','output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    data = build(Path(args.manifest).read_bytes(), Path(args.config).read_bytes(),
                 Path(args.source).read_bytes(), args.revision, args.reference)
    # Refuse to overwrite an existing release artifact.
    with open(args.output, 'xb') as target:
        target.write(data)


if __name__ == '__main__':
    main()
