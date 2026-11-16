"""Exercise durable cache recovery with an explicitly supplied CPU daemon binary.

Run from the repository root. This tests process lifecycle, not GPU performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from devtools.test_tilemaxsim_reference_sidecar import (  # noqa: E402
    scheduled_external_request_frame, decode_response,
)
from devtools import tilemaxsim_reference_sidecar as protocol  # noqa: E402


def http(path, target):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(5)
        sock.connect(str(path))
        sock.sendall(f"GET {target} HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n".encode())
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    header, body = b"".join(chunks).split(b"\r\n\r\n", 1)
    assert b"200 OK" in header, header
    return json.loads(body)


def run(binary):
    with tempfile.TemporaryDirectory(prefix="tilemaxsim-recovery-process-") as directory:
        root = Path(directory)
        tensors = root / "tensors"
        tensors.mkdir()
        state = root / "cache.json"
        score = root / "score.sock"
        status = root / "status.sock"
        args = [str(binary), "--socket", str(score), "--status-socket", str(status),
                "--gpu-memory-gb", "0=0.01", "--gpu-workspace-gb", "0.001",
                "--host-cache-gb", "0.001", "--contract-root", f"model@1={tensors}",
                "--cache-recovery-state", str(state), "--cache-log-interval", "1"]
        log = (root / "process.log").open("w")
        process = None

        def start():
            nonlocal process
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError((root / "process.log").read_text())
                try:
                    if http(status, "/healthz")["ready"]:
                        cache = http(status, "/v1/cache")["cache"]
                        assert cache["devices"][0]["backend"]["backend"] == "cpu", cache
                        return cache
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
            raise AssertionError("daemon did not restore and become ready")

        def query(tenant, number, expected_status=0):
            values = [[float(number), 0.25]]
            payload = struct.pack("<2e", *values[0])
            digest = hashlib.sha256(payload).hexdigest()
            location = tensors / "objects" / digest[:2] / f"{digest}.tensor"
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_bytes(payload)
            frame, _ = scheduled_external_request_frame(
                number, 2, [[1.0, 0.0]], "model@1",
                [(number, f"sha256://{digest}", values)], tenant, 0, 10000)
            with socket.socket(socket.AF_UNIX) as sock:
                sock.settimeout(10)
                sock.connect(str(score))
                sock.sendall(frame)
                header = protocol.receive_exact(sock, 24)
                body = protocol.receive_exact(sock, protocol.HEADER.unpack(header)[4])
            result = decode_response(header + body)
            assert result[1] == expected_status, result
            return result

        try:
            start()
            expected = [query("tenant-a", 1), query("tenant-b", 2)]
            envelope = json.loads(state.read_text())
            checkpoint = json.loads(envelope["payload"])
            assert {e["tenant"] for e in checkpoint["entries"]} == {"tenant-a", "tenant-b"}
            assert len(checkpoint["entries"]) == 2
            assert state.stat().st_mode & 0o777 == 0o600
            process.kill()  # SIGKILL: no shutdown hook can write the state.
            process.wait(timeout=5)
            before = start()
            assert before["devices"][0]["entries"] == 2, before
            before_state = state.read_bytes()
            assert [query("tenant-a", 1), query("tenant-b", 2)] == expected
            after = http(status, "/v1/cache")["cache"]
            assert after["batch_read_bytes"] == before["batch_read_bytes"]
            assert after["devices"][0]["h2d_bytes"] == before["devices"][0]["h2d_bytes"]
            assert state.read_bytes() == before_state  # Cache hits do not rewrite checkpoints.
            committed = state.read_bytes()
            # A failed checkpoint must not acknowledge a successful scoring response.
            state.write_text("{corrupted")
            query("tenant-c", 3, expected_status=4)
            assert process.wait(timeout=5) != 0
            assert state.read_text() == "{corrupted"
            state.write_bytes(committed)
            envelope = json.loads(state.read_text())
            envelope["payload"] = envelope["payload"].replace("tenant-a", "tenant-c")
            state.write_text(json.dumps(envelope))
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            assert process.wait(timeout=15) != 0, "tampered checkpoint must fail before ready"
            print(json.dumps({"result": "passed", "backend": "cpu", "tenants": 2,
                              "restored_entries": 2, "forced_exit": "SIGKILL",
                              "post_restore_storage_reads": 0, "post_restore_h2d_bytes": 0,
                              "tampering_rejected": True, "checkpoint_failure_rejects_success": True}))
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-binary", required=True, type=Path)
    run(parser.parse_args().cpu_binary.resolve())
