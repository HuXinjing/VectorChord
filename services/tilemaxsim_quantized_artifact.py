"""Production VCTQ/VCTC writer for immutable PQ-family TileMaxSim artifacts."""

from __future__ import annotations

import hashlib
import os
import struct
import json
from pathlib import Path

import numpy as np

from services.tilemaxsim_quantization_contract import artifact_tree_sha256

HEADER_BYTES = 128
VERSION = 1


def _digest(contract_id: str) -> bytes:
    if not contract_id.startswith("qtc1-") or len(contract_id) != 69:
        raise ValueError("invalid qtc1 contract ID")
    try:
        return bytes.fromhex(contract_id[5:])
    except ValueError as error:
        raise ValueError("invalid qtc1 contract ID") from error


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _frame(
    magic: bytes,
    payload: bytes,
    *,
    dimension: int,
    contract_digest: bytes,
    stages: int,
    subspaces: int,
    centroids: int = 0,
    rotation_mask: int = 0,
    source_digest: bytes | None = None,
    rows: int = 0,
) -> bytes:
    header = bytearray(HEADER_BYTES)
    header[:4] = magic
    struct.pack_into("<H", header, 4, VERSION)
    if magic == b"VCTQ":
        struct.pack_into("<IHHQHH", header, 8, dimension, stages, subspaces, len(payload), centroids, rotation_mask)
        header[28:60] = contract_digest
        header[60:92] = hashlib.sha256(bytes(header[:60]) + payload).digest()
    elif magic == b"VCTC":
        struct.pack_into("<IIQHH", header, 8, rows, dimension, len(payload), stages, subspaces)
        header[28:60] = contract_digest
        if source_digest is None or len(source_digest) != 32:
            raise ValueError("VCTC requires a SHA-256 source digest")
        header[60:92] = source_digest
        header[92:124] = hashlib.sha256(bytes(header[:92]) + payload).digest()
    else:
        raise ValueError("unsupported artifact magic")
    return bytes(header) + payload


def _quantizer_payload(archive: Path) -> tuple[bytes, dict[str, int | str]]:
    with np.load(archive, allow_pickle=False) as values:
        stages = int(values["stage_count"][0])
        if not 1 <= stages <= 16:
            raise ValueError("invalid quantizer stage count")
        rotations: list[np.ndarray] = []
        books: list[np.ndarray] = []
        rotation_mask = 0
        shape: tuple[int, int, int] | None = None
        for stage in range(stages):
            book = np.asarray(values[f"codebooks_{stage}"], dtype="<f4")
            if book.ndim != 3 or not np.isfinite(book).all():
                raise ValueError("invalid PQ codebook")
            current = (book.shape[0], book.shape[1], book.shape[0] * book.shape[2])
            if shape is not None and current != shape:
                raise ValueError("PQ stages have inconsistent shapes")
            shape = current
            if bool(values[f"has_rotation_{stage}"][0]):
                rotation = np.asarray(values[f"rotation_{stage}"], dtype="<f4")
                if rotation.shape != (current[2], current[2]) or not np.isfinite(rotation).all():
                    raise ValueError("invalid OPQ rotation")
                rotation_mask |= 1 << stage
                rotations.append(rotation)
            books.append(book)
    assert shape is not None
    payload = b"".join(value.tobytes(order="C") for value in (*rotations, *books))
    metadata: dict[str, int | str] = {
        "dimension": shape[2], "stages": stages, "subspaces": shape[0],
        "centroids": shape[1], "rotation_mask": rotation_mask,
        "quantizer_checksum": hashlib.sha256(payload).hexdigest(),
    }
    return payload, metadata


def inspect_quantizer(archive: Path) -> dict[str, int | str]:
    """Return the identity-bearing shape and payload digest before framing."""

    return _quantizer_payload(archive)[1]


def write_quantizer(contract_id: str, archive: Path, destination: Path) -> dict[str, int | str]:
    contract_digest = _digest(contract_id)
    payload, metadata = _quantizer_payload(archive)
    frame = _frame(b"VCTQ", payload, dimension=int(metadata["dimension"]), contract_digest=contract_digest,
                   stages=int(metadata["stages"]), subspaces=int(metadata["subspaces"]),
                   centroids=int(metadata["centroids"]), rotation_mask=int(metadata["rotation_mask"]))
    _atomic_write(destination, frame)
    return metadata


def write_codes(
    contract_id: str,
    codes: np.ndarray,
    source_digests: list[str],
    rows: np.ndarray,
    dimension: int,
    destination_root: Path,
) -> None:
    contract_digest = _digest(contract_id)
    codes = np.asarray(codes, dtype=np.uint8)
    rows = np.asarray(rows, dtype=np.int64)
    if codes.ndim != 3 or len(rows) != len(source_digests) or int(rows.sum()) != codes.shape[0]:
        raise ValueError("PQ codes, rows, and source digests disagree")
    position = 0
    for count, source in zip(rows, source_digests, strict=True):
        source_digest = bytes.fromhex(source.removeprefix("sha256:"))
        if len(source_digest) != 32 or count <= 0:
            raise ValueError("invalid source digest or row count")
        payload = codes[position : position + int(count)].tobytes(order="C")
        frame = _frame(b"VCTC", payload, dimension=dimension, contract_digest=contract_digest,
                       stages=codes.shape[1], subspaces=codes.shape[2], source_digest=source_digest,
                       rows=int(count))
        hexadecimal = source_digest.hex()
        _atomic_write(destination_root / "codes" / hexadecimal[:2] / f"{hexadecimal}.vctc", frame)
        position += int(count)


def finalize_artifact(destination_root: Path, source_manifest_checksum: str) -> str:
    """Durably mark a fully written immutable tree complete and return its digest.

    metadata.json is deliberately excluded from the tree digest, allowing the
    caller to compute the v2 contract ID before framing files and then bind the
    resulting tree checksum without an identity cycle.
    """

    if len(source_manifest_checksum) != 64:
        raise ValueError("invalid source manifest checksum")
    try:
        bytes.fromhex(source_manifest_checksum)
    except ValueError as error:
        raise ValueError("invalid source manifest checksum") from error
    if not (destination_root / "quantizer.vctq").is_file():
        raise ValueError("quantized artifact has no quantizer.vctq")
    if not any((destination_root / "codes").rglob("*.vctc")):
        raise ValueError("quantized artifact has no VCTC codes")
    metadata = json.dumps(
        {"complete": True, "source_manifest_checksum": source_manifest_checksum},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    _atomic_write(destination_root / "metadata.json", metadata)
    return artifact_tree_sha256(destination_root)
