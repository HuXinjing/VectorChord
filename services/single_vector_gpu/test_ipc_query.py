import hashlib
from pathlib import Path
import socket
import tempfile
import unittest

from .ipc_query import FRAME, IpcInbox, IpcUnavailable


class IpcInboxTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "ipc.sock")
        self.inbox = IpcInbox(self.path)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.inbox.close)

    def publish(self, reference, nbytes=12):
        producer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(producer.close)
        producer.settimeout(1)
        producer.connect(self.path)
        producer.sendall(FRAME.pack(
            b"SVQ1", reference, 1, 3, nbytes, bytes(64), bytes(64),
            hashlib.sha256(b"tenant-lease").digest(),
        ))
        return producer

    def test_claim_checks_original_lease_and_prevents_reuse(self):
        ref = b"1" * 16
        producer = self.publish(ref)
        for references, leases, dimension, error in (
            ([ref], ["wrong-lease"], 3, PermissionError),
            ([ref], ["tenant-lease"], 4, ValueError),
            ([ref, ref], ["tenant-lease", "tenant-lease"], 3, ValueError),
        ):
            with self.assertRaises(error):
                with self.inbox.claim(references, leases, dimension):
                    self.fail("Invalid claim was admitted")
        with self.inbox.claim([ref], ["tenant-lease"], 3) as items:
            self.assertEqual(items[0].dimension, 3)
        self.assertEqual(producer.recv(1), b"R")
        with self.assertRaises(IpcUnavailable):
            with self.inbox.claim([ref], ["tenant-lease"], 3):
                self.fail("Reference was reusable")

    def test_invalid_frame_is_released_without_a_mapping(self):
        producer = self.publish(b"2" * 16, nbytes=6)
        self.assertEqual(producer.recv(1), b"R")
        self.assertEqual(self.inbox.entries, {})

    def test_cannot_replace_a_live_inbox(self):
        with self.assertRaises(BlockingIOError):
            IpcInbox(self.path)
        producer = self.publish(b"3" * 16)
        with self.inbox.claim([b"3" * 16], ["tenant-lease"], 3):
            pass
        self.assertEqual(producer.recv(1), b"R")

    def test_can_recover_a_stale_socket(self):
        path = str(Path(self.directory.name) / "stale.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(path)
        recovered = IpcInbox(path)
        recovered.close()
