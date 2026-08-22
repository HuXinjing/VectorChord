# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Build a complete, auditable visual-TileMaxSim quantization dataset.

The command reuses compatible content-addressed page tensors, accepts previous
generated-page manifests as read-only supplements, and encodes only genuinely
missing PDF pages.  It fails closed unless every corpus document, every qrel,
and every query tensor is represented in the resulting execution manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility with older environments
    import fitz as pymupdf


MODEL = "colqwen3.5-4.5B-v3"
MODEL_CONTRACT = "colqwen35@pdf-page-visual-v1"
DIMENSION = 320
DTYPE = np.dtype("<f2")
MAX_PIXELS = 786_432


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def post_json(
    url: str, payload: dict[str, Any], timeout: float = 240.0
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def corpus_bindings(
    corpus: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_path = {item["canonical_path"]: item for item in corpus}
    by_alias = {alias: item for item in corpus for alias in item.get("aliases", [])}
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in corpus:
        for name in item.get("filenames", []):
            by_name[name].append(item)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        document = by_path.get(page["rel_path"]) or by_alias.get(page["rel_path"])
        if document is None:
            candidates = by_name.get(Path(page["rel_path"]).name, ())
            if len(candidates) == 1:
                document = candidates[0]
        if document is not None:
            grouped[document["doc_id"]].append(page)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["page_no"]))
    return dict(grouped)


def load_supplements(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("model") != MODEL
            or int(payload.get("dimension", 0)) != DIMENSION
        ):
            raise ValueError(f"incompatible supplement: {path}")
        root = path.parent
        for doc_id, pages in payload.get("generated_documents", {}).items():
            converted = []
            for page in pages:
                source = root / page["tensor_file"]
                rows = int(page["tensor_rows"])
                dimension = int(page["tensor_dim"])
                if (
                    not source.is_file()
                    or dimension != DIMENSION
                    or rows <= 0
                    or source.stat().st_size != rows * dimension * DTYPE.itemsize
                ):
                    raise ValueError(
                        f"missing or malformed supplement tensor: {source}"
                    )
                converted.append(
                    {
                        "page_key": page["page_key"],
                        "page_no": int(page["page_no"]),
                        "tensor_rows": rows,
                        "tensor_dim": DIMENSION,
                        "tensor_dtype": "float16",
                        "source": {
                            "kind": "file",
                            "path": str(source.resolve()),
                            "checksum": file_sha256(source),
                        },
                    }
                )
            converted.sort(key=lambda item: item["page_no"])
            previous = result.setdefault(doc_id, converted)
            if previous != converted:
                raise ValueError(f"conflicting supplement pages for {doc_id}")
    return result


def render_page(path: Path, page_number: int) -> bytes:
    document = pymupdf.open(path)
    try:
        pixmap = document[page_number - 1].get_pixmap(
            matrix=pymupdf.Matrix(2.0, 2.0), alpha=False
        )
        return pixmap.tobytes("png")
    finally:
        document.close()


def encode_image(pooling_url: str, png: bytes) -> np.ndarray:
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    response = post_json(
        pooling_url,
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            "mm_processor_kwargs": {
                "min_pixels": 65_536,
                "max_pixels": MAX_PIXELS,
            },
        },
    )
    result = np.asarray(response["data"][0]["data"], dtype=DTYPE)
    if (
        result.ndim != 2
        or result.shape[0] <= 0
        or result.shape[1] != DIMENSION
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"unexpected image tensor shape or values: {result.shape}")
    return result


def locate_pdf(root: Path, document: dict[str, Any]) -> Path:
    exact = root / document["canonical_path"]
    if exact.is_file():
        return exact
    matches: set[Path] = set()
    for filename in document.get("filenames", []):
        matches.update(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one PDF for {document['doc_id']}, found {len(matches)}"
        )
    return matches.pop()


def existing_generated(output: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = output / "generated-pages.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not manifest.exists():
        return {}
    for item in read_jsonl(manifest):
        tensor = output / item["tensor_file"]
        rows = int(item["tensor_rows"])
        dimension = int(item["tensor_dim"])
        if (
            tensor.is_file()
            and rows > 0
            and dimension == DIMENSION
            and tensor.stat().st_size == rows * dimension * DTYPE.itemsize
        ):
            actual_checksum = file_sha256(tensor)
            registered_checksum = item.get("tensor_checksum")
            if (
                registered_checksum is not None
                and registered_checksum != actual_checksum
            ):
                continue
            grouped[item["doc_id"]].append(
                {
                    "page_key": item["page_key"],
                    "page_no": int(item["page_no"]),
                    "tensor_rows": rows,
                    "tensor_dim": DIMENSION,
                    "tensor_dtype": "float16",
                    "source": {
                        "kind": "file",
                        "path": str(tensor.resolve()),
                        "checksum": actual_checksum,
                    },
                }
            )
    for pages in grouped.values():
        pages.sort(key=lambda item: item["page_no"])
    return dict(grouped)


def encode_missing_documents(
    corpus: list[dict[str, Any]],
    missing_ids: set[str],
    pdf_root: Path,
    output: Path,
    pooling_url: str,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    completed = existing_generated(output)
    by_id = {item["doc_id"]: item for item in corpus}
    tasks: list[tuple[dict[str, Any], Path, int, str]] = []
    for doc_id in sorted(missing_ids):
        document = by_id[doc_id]
        pdf = locate_pdf(pdf_root, document)
        with pymupdf.open(pdf) as opened:
            page_count = len(opened)
        known = {page["page_no"] for page in completed.get(doc_id, ())}
        for page_number in range(1, page_count + 1):
            if page_number in known:
                continue
            page_key = hashlib.sha1(
                f"{document['sha256']}:{page_number}".encode()
            ).hexdigest()
            tasks.append((document, pdf, page_number, page_key))

    print(
        json.dumps(
            {
                "event": "encode_plan",
                "missing_documents": len(missing_ids),
                "remaining_pages": len(tasks),
                "concurrency": concurrency,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tensor_root = output / "page-tensors"
    tensor_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generated-pages.jsonl"

    def work(task: tuple[dict[str, Any], Path, int, str]) -> dict[str, Any]:
        document, pdf, page_number, page_key = task
        error: Exception | None = None
        for attempt in range(5):
            try:
                tensor = encode_image(pooling_url, render_page(pdf, page_number))
                relative = Path("page-tensors") / page_key[:2] / f"{page_key}.f16"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(f".tmp-{os.getpid()}")
                tensor.tofile(temporary)
                os.replace(temporary, target)
                return {
                    "doc_id": document["doc_id"],
                    "page_key": page_key,
                    "page_no": page_number,
                    "tensor_file": relative.as_posix(),
                    "tensor_rows": int(tensor.shape[0]),
                    "tensor_dim": DIMENSION,
                    "tensor_dtype": "float16",
                    "tensor_checksum": "sha256:"
                    + hashlib.sha256(tensor.tobytes()).hexdigest(),
                }
            except Exception as caught:  # transient HTTP or renderer failure
                error = caught
                time.sleep(min(2 ** (attempt + 1), 15))
        raise RuntimeError(f"failed {document['doc_id']} page {page_number}: {error}")

    started = time.perf_counter()
    if tasks:
        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [executor.submit(work, task) for task in tasks]
            with manifest_path.open("a", encoding="utf-8") as stream:
                for index, future in enumerate(as_completed(futures), 1):
                    item = future.result()
                    stream.write(
                        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    stream.flush()
                    if index % 25 == 0 or index == len(tasks):
                        elapsed = time.perf_counter() - started
                        print(
                            json.dumps(
                                {
                                    "event": "encode_progress",
                                    "completed": index,
                                    "total": len(tasks),
                                    "pages_per_second": index / elapsed,
                                }
                            ),
                            flush=True,
                        )
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return existing_generated(output)


def encode_queries(qrels: list[dict[str, Any]], output: Path, pooling_url: str) -> Path:
    destination = output / "query-tensors.npz"
    expected = {item["id"] for item in qrels}
    if destination.exists():
        with np.load(destination, allow_pickle=False) as archive:
            if set(archive.files) == expected and all(
                archive[key].ndim == 2 and archive[key].shape[1] == DIMENSION
                for key in archive.files
            ):
                return destination
    response = post_json(
        pooling_url,
        {"model": MODEL, "input": [item["query"] for item in qrels]},
    )
    if len(response.get("data", [])) != len(qrels):
        raise ValueError("pooling response omitted query tensors")
    by_index = {int(item["index"]): item for item in response["data"]}
    if set(by_index) != set(range(len(qrels))):
        raise ValueError("pooling response has duplicate or invalid query indexes")
    tensors = {}
    for index, item in enumerate(qrels):
        encoded = by_index[index]
        tensor = np.asarray(encoded["data"], dtype=DTYPE)
        if (
            tensor.ndim != 2
            or tensor.shape[1] != DIMENSION
            or not np.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid query tensor for {item['id']}: {tensor.shape}")
        tensors[item["id"]] = tensor
    np.savez(destination, **tensors)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--page-text", required=True, type=Path)
    parser.add_argument("--descriptors", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--supplement", action="append", default=[], type=Path)
    parser.add_argument("--pooling-url", default="http://127.0.0.1:18019/pooling")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="ignore historical tensors and encode every corpus page with the current model",
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(args.corpus)
    qrels_payload = json.loads(args.qrels.read_text(encoding="utf-8"))
    qrels = qrels_payload.get("queries", [])
    if len(corpus) != 1124 or len(qrels) != 81:
        raise ValueError(
            f"expected the frozen 1124-document/81-query dataset, got {len(corpus)}/{len(qrels)}"
        )
    corpus_ids = {item["doc_id"] for item in corpus}
    relevant_ids = {doc_id for item in qrels for doc_id in item["relevant"]}
    unknown = relevant_ids - corpus_ids
    if unknown:
        raise ValueError(f"qrels reference unknown documents: {sorted(unknown)}")

    old_pages = read_jsonl(args.page_text)
    old_descriptors = {item["page_key"]: item for item in read_jsonl(args.descriptors)}
    old_by_doc = {} if args.regenerate_all else corpus_bindings(corpus, old_pages)
    for pages in old_by_doc.values():
        for page in pages:
            if page["page_key"] not in old_descriptors:
                raise ValueError(f"page has no tensor descriptor: {page['page_key']}")
    supplements = {} if args.regenerate_all else load_supplements(args.supplement)
    covered = set(old_by_doc) | set(supplements)
    generated = encode_missing_documents(
        corpus,
        corpus_ids - covered,
        args.pdf_root,
        args.output,
        args.pooling_url,
        args.concurrency,
    )
    overlap = (set(generated) | set(supplements)) & set(old_by_doc)
    if overlap:
        raise ValueError(
            f"generated pages overlap canonical cache: {sorted(overlap)[:5]}"
        )
    all_ids = set(old_by_doc) | set(supplements) | set(generated)
    if all_ids != corpus_ids:
        raise RuntimeError(
            f"tensor coverage incomplete: {len(corpus_ids - all_ids)} documents"
        )
    query_tensors = encode_queries(qrels, args.output, args.pooling_url)

    documents: dict[str, list[dict[str, Any]]] = {}
    for doc_id, pages in old_by_doc.items():
        documents[doc_id] = [
            {
                "page_key": page["page_key"],
                "page_no": int(page["page_no"]),
                "tensor_rows": int(old_descriptors[page["page_key"]]["tensor_rows"]),
                "tensor_dim": int(old_descriptors[page["page_key"]]["tensor_dim"]),
                "tensor_dtype": old_descriptors[page["page_key"]]["tensor_dtype"],
                "source": {
                    "kind": "shard",
                    "digest": old_descriptors[page["page_key"]][
                        "tensor_checksum"
                    ].removeprefix("sha256:"),
                },
            }
            for page in pages
        ]
    documents.update(supplements)
    documents.update(generated)
    manifest = {
        "version": 1,
        "model": MODEL,
        "model_contract": MODEL_CONTRACT,
        "encoding_mode": "regenerated-all"
        if args.regenerate_all
        else "reuse-compatible",
        "dimension": DIMENSION,
        "preprocess": {
            "renderer": "PyMuPDF",
            "scale": 2.0,
            "min_pixels": 65_536,
            "max_pixels": MAX_PIXELS,
            "image_prompt": "Describe the image.",
        },
        "input_checksums": {
            "corpus": file_sha256(args.corpus),
            "qrels": file_sha256(args.qrels),
            "page_text": file_sha256(args.page_text),
            "descriptors": file_sha256(args.descriptors),
        },
        "corpus": str(args.corpus.resolve()),
        "qrels": str(args.qrels.resolve()),
        "query_tensors": str(query_tensors.resolve()),
        "shard_root": str(args.shard_root.resolve()),
        "documents": documents,
        "coverage": {
            "corpus_documents": len(corpus),
            "documents": len(documents),
            "queries": len(qrels),
            "relevant_documents": len(relevant_ids),
            "relevant_documents_covered": len(relevant_ids & set(documents)),
            "old_documents": len(old_by_doc),
            "supplement_documents": len(supplements),
            "generated_documents": len(generated),
        },
    }
    destination = args.output / "execution-manifest.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "event": "dataset_ready",
                "manifest": str(destination),
                **manifest["coverage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
