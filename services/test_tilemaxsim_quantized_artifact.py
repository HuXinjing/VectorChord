import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.tilemaxsim_quantized_artifact import (
    finalize_artifact,
    inspect_quantizer,
    write_codes,
    write_quantizer,
)


class ProductionQuantizedArtifactTest(unittest.TestCase):
    def test_writes_atomic_quantizer_and_content_addressed_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "quantizer.npz"
            np.savez(
                archive,
                stage_count=np.asarray([2], dtype=np.int32),
                codebooks_0=np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
                has_rotation_0=np.asarray([True]),
                rotation_0=np.eye(2, dtype=np.float32),
                codebooks_1=np.asarray([[[0.5, 0.0], [0.0, 0.5]]], dtype=np.float32),
                has_rotation_1=np.asarray([False]),
            )
            contract = "qtc1-" + "a" * 64
            inspected = inspect_quantizer(archive)
            metadata = write_quantizer(contract, archive, root / "quantizer.vctq")
            self.assertEqual(inspected, metadata)
            self.assertEqual(metadata["dimension"], 2)
            self.assertEqual(metadata["stages"], 2)
            self.assertEqual(metadata["subspaces"], 1)
            self.assertEqual(metadata["centroids"], 2)
            self.assertEqual(metadata["rotation_mask"], 1)
            quantizer = (root / "quantizer.vctq").read_bytes()
            self.assertEqual(quantizer[:4], b"VCTQ")
            self.assertEqual(quantizer[28:60], bytes.fromhex("a" * 64))
            self.assertEqual(quantizer[60:92], hashlib.sha256(quantizer[:60] + quantizer[128:]).digest())
            self.assertEqual(metadata["quantizer_checksum"], hashlib.sha256(quantizer[128:]).hexdigest())

            source = hashlib.sha256(b"source").hexdigest()
            codes = np.asarray([[[0], [0]], [[1], [1]]], dtype=np.uint8)
            write_codes(contract, codes, [source], np.asarray([2]), 2, root)
            target = root / "codes" / source[:2] / f"{source}.vctc"
            payload = target.read_bytes()
            self.assertEqual(payload[:4], b"VCTC")
            self.assertEqual(payload[60:92], bytes.fromhex(source))
            self.assertEqual(payload[92:124], hashlib.sha256(payload[:92] + payload[128:]).digest())
            tree = finalize_artifact(root, hashlib.sha256(b"manifest").hexdigest())
            self.assertEqual(len(tree), 64)
            self.assertTrue((root / "metadata.json").is_file())
            self.assertFalse(any(root.rglob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
