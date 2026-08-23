# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from services.tilemaxsim_quantization_contract import (
    QuantizationContract,
    QuantizationContractRegistry,
    artifact_tree_sha256,
)


def contract(source: str, artifact: str) -> QuantizationContract:
    return QuantizationContract(
        model_contract="model@1",
        source_manifest_checksum=source,
        encoding="pq",
        dimension=320,
        pooling=2,
        normalize_pooling=True,
        subspaces=16,
        centroids=256,
        residual_stages=2,
        opq_iterations=4,
        artifact_checksum=artifact,
    )


class QuantizationContractRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.source = hashlib.sha256(b"source").hexdigest()
        (self.artifact / "metadata.json").write_text(
            json.dumps({"complete": True, "source_manifest_checksum": self.source})
        )
        (self.artifact / "codes.bin").write_bytes(b"first")
        self.registry = QuantizationContractRegistry(self.root / "registry")

    def tearDown(self):
        self.temporary.cleanup()

    def test_stage_activate_upgrade_and_rollback(self):
        first_contract = contract(self.source, artifact_tree_sha256(self.artifact))
        first = self.registry.stage(first_contract, self.artifact)
        self.assertEqual(self.registry.activate(first, expected_active=None), 1)
        second_artifact = self.root / "artifact-2"
        second_artifact.mkdir()
        (second_artifact / "metadata.json").write_text(
            json.dumps({"complete": True, "source_manifest_checksum": self.source})
        )
        (second_artifact / "codes.bin").write_bytes(b"second")
        second = self.registry.stage(
            contract(self.source, artifact_tree_sha256(second_artifact)),
            second_artifact,
        )
        self.assertEqual(self.registry.activate(second, expected_active=first), 2)
        self.assertEqual(self.registry.resolve_active()["contract_id"], second)
        self.assertEqual(self.registry.rollback(expected_active=second), 3)
        self.assertEqual(self.registry.resolve_active()["contract_id"], first)

    def test_compare_and_swap_rejects_stale_controller(self):
        identifier = self.registry.stage(
            contract(self.source, artifact_tree_sha256(self.artifact)), self.artifact
        )
        self.registry.activate(identifier, expected_active=None)
        with self.assertRaisesRegex(RuntimeError, "changed concurrently"):
            self.registry.activate(identifier, expected_active=None)

    def test_incomplete_or_wrong_source_fails_closed(self):
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        (incomplete / "metadata.json").write_text(json.dumps({"complete": False}))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.registry.stage(contract(self.source, "a" * 64), incomplete)
        with self.assertRaisesRegex(ValueError, "source checksum"):
            self.registry.stage(
                contract("0" * 64, artifact_tree_sha256(self.artifact)), self.artifact
            )

    def test_payload_tampering_fails_closed(self):
        expected = artifact_tree_sha256(self.artifact)
        (self.artifact / "codes.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "content checksum"):
            self.registry.stage(contract(self.source, expected), self.artifact)

    def test_contract_id_is_content_addressed(self):
        first = contract(self.source, "a" * 64)
        self.assertEqual(first.contract_id, contract(self.source, "a" * 64).contract_id)
        self.assertNotEqual(first.contract_id, contract(self.source, "b" * 64).contract_id)

    def test_invalid_contract_and_symlink_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            contract(self.source, "not-a-checksum")
        link = self.artifact / "outside-link"
        link.symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            self.registry.stage(
                contract(self.source, artifact_tree_sha256(self.artifact)), self.artifact
            )

    def test_artifact_changed_after_staging_cannot_activate(self):
        identifier = self.registry.stage(
            contract(self.source, artifact_tree_sha256(self.artifact)), self.artifact
        )
        (self.artifact / "codes.bin").write_bytes(b"changed-after-stage")
        with self.assertRaisesRegex(ValueError, "changed before activation"):
            self.registry.activate(identifier, expected_active=None)

    def test_rollback_revalidates_previous_artifact(self):
        first = self.registry.stage(
            contract(self.source, artifact_tree_sha256(self.artifact)), self.artifact
        )
        self.registry.activate(first, expected_active=None)
        second_artifact = self.root / "artifact-2"
        second_artifact.mkdir()
        (second_artifact / "metadata.json").write_text(
            json.dumps({"complete": True, "source_manifest_checksum": self.source})
        )
        (second_artifact / "codes.bin").write_bytes(b"second")
        second = self.registry.stage(
            contract(self.source, artifact_tree_sha256(second_artifact)), second_artifact
        )
        self.registry.activate(second, expected_active=first)
        (self.artifact / "codes.bin").write_bytes(b"broken-previous")
        with self.assertRaisesRegex(ValueError, "changed before activation"):
            self.registry.rollback(expected_active=second)
        self.assertEqual(self.registry.resolve_active()["contract_id"], second)


if __name__ == "__main__":
    unittest.main()
