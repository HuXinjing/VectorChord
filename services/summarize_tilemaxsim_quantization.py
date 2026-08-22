# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Summarize qrel recall and exact-TileMaxSim fidelity for quantization ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def agreement(expected: list[str], actual: list[str], k: int) -> float:
    return len(set(expected[:k]) & set(actual[:k])) / k


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--baseline", default="exact-fp16")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    output_json = args.output_json.resolve()
    paths = sorted(
        path
        for path in args.report_root.glob("*.json")
        if path.resolve() != output_json
    )
    reports = {}
    for path in paths:
        payload = load(path)
        if "variant" in payload and "cases" in payload:
            reports[path.stem] = payload
    if args.baseline not in reports:
        raise ValueError(f"missing baseline report {args.baseline!r}")
    baseline = reports[args.baseline]
    baseline_cases = {item["query_id"]: item for item in baseline["cases"]}
    baseline_latency = float(baseline["latency_ms"]["mean"])
    baseline_bytes = int(baseline["cache"]["artifact_bytes"])
    rows = []
    for name, report in reports.items():
        cases = report["cases"]
        if {item["query_id"] for item in cases} != set(baseline_cases):
            raise ValueError(f"query set differs from baseline for {name}")
        latency = float(report["latency_ms"]["mean"])
        artifact_bytes = int(report["cache"]["artifact_bytes"])
        rows.append(
            {
                "variant": name,
                **report["variant"],
                **report["quality"],
                "delta_recall_at_10": float(report["quality"]["recall_at_10"])
                - float(baseline["quality"]["recall_at_10"]),
                "exact_top1_agreement": sum(
                    agreement(
                        baseline_cases[item["query_id"]]["top_20"], item["top_20"], 1
                    )
                    for item in cases
                )
                / len(cases),
                "exact_top5_overlap": sum(
                    agreement(
                        baseline_cases[item["query_id"]]["top_20"], item["top_20"], 5
                    )
                    for item in cases
                )
                / len(cases),
                "exact_top10_overlap": sum(
                    agreement(
                        baseline_cases[item["query_id"]]["top_20"], item["top_20"], 10
                    )
                    for item in cases
                )
                / len(cases),
                "mean_latency_ms": latency,
                "speedup": baseline_latency / latency,
                "artifact_bytes": artifact_bytes,
                "storage_ratio": artifact_bytes / baseline_bytes,
                "mean_transfer_ms": float(report["transfer_ms"]["mean"]),
                "mean_kernel_ms": float(report["kernel_ms"]["mean"]),
                "mean_query_prep_ms": float(report["query_prep_ms"]["mean"]),
                "padding_row_ratio": float(report["padding_row_ratio"]),
            }
        )
    rows.sort(key=lambda item: item["variant"])
    payload = {
        "version": 1,
        "scope": {
            "retrieval": "visual TileMaxSim only",
            "excluded": ["FTS", "graph", "fact store", "reranker", "query expansion"],
            "baseline": args.baseline,
            "documents": baseline["dataset"]["documents"],
            "queries": baseline["dataset"]["queries"],
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TileMaxSim quantization ablation",
        "",
        "This report compares only visual TileMaxSim on the frozen 1,124-document, 81-query qrels. FTS, graph, Fact Store, reranking, and query expansion are excluded.",
        "",
        "| Variant | R@1 | R@5 | R@10 | Hit@10 | ΔR@10 | Exact top-10 overlap | Mean ms | Speedup | Storage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {recall_at_1:.4f} | {recall_at_5:.4f} | {recall_at_10:.4f} | {hit_at_10:.4f} | {delta_recall_at_10:+.4f} | {exact_top10_overlap:.4f} | {mean_latency_ms:.2f} | {speedup:.2f}× | {storage_ratio:.3f}× |".format(
                **row
            )
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
