# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Versioned, atomic activation registry for quantized TileMaxSim artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import string
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class QuantizationContract:
    model_contract: str
    source_manifest_checksum: str
    encoding: str
    dimension: int
    pooling: int
    normalize_pooling: bool
    subspaces: int
    centroids: int
    residual_stages: int
    opq_iterations: int
    # Digest of the trained encoder/codebook material before VCTQ framing.
    # This participates in v2 identity, unlike the post-framing artifact tree.
    quantizer_checksum: str
    artifact_checksum: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not self.model_contract
            or len(self.model_contract) > 256
            or any(not character.isprintable() for character in self.model_contract)
        ):
            raise ValueError("invalid model contract ID")
        for name, value in (
            ("source manifest", self.source_manifest_checksum),
            ("quantizer", self.quantizer_checksum),
            ("artifact", self.artifact_checksum),
        ):
            if len(value) != 64 or any(character not in string.hexdigits for character in value):
                raise ValueError(f"invalid {name} checksum")
        if self.schema_version not in (1, 2) or self.dimension <= 0 or self.pooling < 0:
            raise ValueError("invalid quantization contract dimensions or version")
        if self.encoding not in {"fp16", "int8", "fp8", "pq"}:
            raise ValueError("unsupported quantization encoding")
        if min(self.subspaces, self.centroids, self.residual_stages, self.opq_iterations) < 0:
            raise ValueError("quantization parameters must be nonnegative")
        if self.encoding == "pq" and min(
            self.subspaces, self.centroids, self.residual_stages
        ) <= 0:
            raise ValueError("PQ contracts require positive codebook parameters")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def contract_id(self) -> str:
        if self.schema_version == 1:
            identity = self.canonical_bytes()
        else:
            # v2 breaks the circular dependency between an immutable artifact
            # header (which embeds this ID) and the separately verified artifact
            # checksum. The checksum remains signed by the registry record and
            # is revalidated at stage, activation, rollback, and resolve time.
            fields = asdict(self)
            del fields["artifact_checksum"]
            identity = json.dumps(
                fields, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        return "qtc1-" + hashlib.sha256(identity).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_tree_sha256(root: Path) -> str:
    """Hash artifact names, lengths, and bytes without trusting metadata."""

    digest = hashlib.sha256()
    discovered = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in discovered):
        raise ValueError("quantized artifact must not contain symbolic links")
    files = [
        path for path in discovered if path.is_file() and path.name != "metadata.json"
    ]
    if not files:
        raise ValueError("quantized artifact contains no payload files")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class QuantizationContractRegistry:
    """Keep staged and active contracts without mutating artifact directories."""

    def __init__(self, root: Path):
        self.root = root
        self.contracts = root / "contracts"
        self.state_path = root / "state.json"
        self.lock_path = root / ".registry.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {"version": 2, "generation": 0, "scopes": {}}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(state.get("generation"), int):
            raise ValueError("invalid quantization registry state")
        if state.get("version") == 1:
            return self._migrate_v1_state(state)
        if state.get("version") != 2 or not isinstance(state.get("scopes"), dict):
            raise ValueError("invalid quantization registry state")
        return state

    def _record(self, contract_id: str) -> dict[str, object]:
        path = self.contracts / f"{contract_id}.json"
        if not path.is_file():
            raise ValueError("contract has not been staged")
        return json.loads(path.read_text(encoding="utf-8"))

    def _scope_for_record(self, record: dict[str, object]) -> str:
        contract = QuantizationContract(**record["contract"])
        return f"{contract.model_contract}\x1f{contract.encoding}"

    def _migrate_v1_state(self, state: dict[str, object]) -> dict[str, object]:
        """Translate the old singleton state without mutating it during reads."""

        scopes: dict[str, dict[str, str | None]] = {}
        active = state.get("active")
        previous = state.get("previous")
        if active is not None:
            if not isinstance(active, str):
                raise ValueError("invalid v1 quantization registry state")
            active_record = self._record(active)
            scope = self._scope_for_record(active_record)
            if previous is not None:
                if not isinstance(previous, str):
                    raise ValueError("invalid v1 quantization registry state")
                if self._scope_for_record(self._record(previous)) != scope:
                    # A singleton v1 registry could switch between encodings.
                    # Keeping the old contract as another scope's active value
                    # preserves rollback data without inventing cross-format
                    # rollback semantics in v2.
                    previous_scope = self._scope_for_record(self._record(previous))
                    scopes[previous_scope] = {"active": previous, "previous": None}
                    previous = None
            scopes[scope] = {"active": active, "previous": previous}
        return {"version": 2, "generation": int(state["generation"]), "scopes": scopes}

    def _write_state(self, state: dict[str, object]) -> None:
        temporary = self.state_path.with_suffix(f".tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def stage(self, contract: QuantizationContract, artifact_root: Path) -> str:
        metadata = artifact_root / "metadata.json"
        if not metadata.is_file():
            raise ValueError("quantized artifact has no metadata.json")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise ValueError("quantized artifact is incomplete")
        if payload.get("source_manifest_checksum") != contract.source_manifest_checksum:
            raise ValueError("artifact source checksum disagrees with contract")
        if artifact_tree_sha256(artifact_root) != contract.artifact_checksum:
            raise ValueError("artifact content checksum disagrees with contract")
        contract_id = contract.contract_id
        record = {
            "version": 1,
            "contract_id": contract_id,
            "contract": asdict(contract),
            "artifact_root": os.fspath(artifact_root.resolve()),
        }
        with self._locked():
            self.contracts.mkdir(parents=True, exist_ok=True)
            destination = self.contracts / f"{contract_id}.json"
            if destination.exists():
                if json.loads(destination.read_text(encoding="utf-8")) != record:
                    raise ValueError("immutable contract ID collision")
            else:
                temporary = destination.with_suffix(f".tmp.{os.getpid()}")
                with temporary.open("w", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                directory = os.open(self.contracts, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        return contract_id

    def _verify_staged(self, contract_id: str) -> None:
        path = self.contracts / f"{contract_id}.json"
        if not path.is_file():
            raise ValueError("contract has not been staged")
        record = json.loads(path.read_text(encoding="utf-8"))
        contract = QuantizationContract(**record["contract"])
        if contract.contract_id != contract_id:
            raise ValueError("staged contract ID disagrees with its content")
        artifact_root = Path(record["artifact_root"])
        if artifact_tree_sha256(artifact_root) != contract.artifact_checksum:
            raise ValueError("staged artifact changed before activation")

    def activate(self, contract_id: str, *, expected_active: str | None) -> int:
        with self._locked():
            self._verify_staged(contract_id)
            scope = self._scope_for_record(self._record(contract_id))
            state = self._state()
            scopes = state["scopes"]
            current = scopes.get(scope, {"active": None, "previous": None})
            if current.get("active") != expected_active:
                raise RuntimeError("active contract changed concurrently")
            if contract_id == expected_active:
                return int(state["generation"])
            scopes[scope] = {"previous": current.get("active"), "active": contract_id}
            state["generation"] = int(state["generation"]) + 1
            self._write_state(state)
            return int(state["generation"])

    def rollback(self, *, expected_active: str) -> int:
        with self._locked():
            scope = self._scope_for_record(self._record(expected_active))
            state = self._state()
            current = state["scopes"].get(scope, {"active": None, "previous": None})
            if current.get("active") != expected_active:
                raise RuntimeError("active contract changed concurrently")
            previous = current.get("previous")
            if not isinstance(previous, str):
                raise ValueError("no previous contract is available")
            self._verify_staged(previous)
            current["active"], current["previous"] = previous, current["active"]
            state["scopes"][scope] = current
            state["generation"] = int(state["generation"]) + 1
            self._write_state(state)
            return int(state["generation"])

    def resolve_active(
        self, *, model_contract: str | None = None, encoding: str | None = None
    ) -> dict[str, object] | None:
        with self._locked():
            state = self._state()
            if (model_contract is None) != (encoding is None):
                raise ValueError("model_contract and encoding must be supplied together")
            if model_contract is None:
                active_scopes = [
                    value for value in state["scopes"].values() if value.get("active")
                ]
                if not active_scopes:
                    return None
                if len(active_scopes) != 1:
                    raise ValueError("multiple active scopes require an explicit model and encoding")
                active = active_scopes[0]["active"]
            else:
                scope = f"{model_contract}\x1f{encoding}"
                active = state["scopes"].get(scope, {}).get("active")
            if active is None:
                return None
            return self._record(active)
