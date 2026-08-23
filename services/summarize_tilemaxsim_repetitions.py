# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Aggregate repeated TileMaxSim runs with deterministic bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def bootstrap_interval(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not values or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(statistic(resampled))
    tail = (1 - confidence) / 2
    return percentile(estimates, tail), percentile(estimates, 1 - tail)


def hierarchical_bootstrap_interval(
    runs: list[list[float]],
    statistic: Callable[[list[float]], float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not runs or any(not run for run in runs):
        raise ValueError("hierarchical bootstrap runs must be nonempty")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        combined = []
        for _ in runs:
            run = runs[generator.randrange(len(runs))]
            combined.extend(run[generator.randrange(len(run))] for _ in run)
        estimates.append(statistic(combined))
    tail = (1 - confidence) / 2
    return percentile(estimates, tail), percentile(estimates, 1 - tail)


def latency_samples(report: dict[str, Any], field: str) -> list[float]:
    cases = report.get("cases", [])
    if not cases:
        raise ValueError("report does not contain per-query cases")
    warmups = int(report.get("warmup_queries", 0))
    if any("measured" in case for case in cases):
        selected = [case for case in cases if case.get("measured", False)]
    else:
        selected = cases[warmups:]
    return [float(case[field]) for case in selected]


def summarize(
    reports: list[dict[str, Any]], *, samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    variant = reports[0]["variant"]
    quality = reports[0]["quality"]
    for report in reports[1:]:
        if report["variant"] != variant:
            raise ValueError("variant definitions differ across repetitions")
        if report["quality"] != quality:
            raise ValueError("quality metrics differ across deterministic repetitions")
    metrics = {}
    for output_name, case_field in (
        ("latency_ms", "latency_ms"),
        ("transfer_ms", "transfer_ms"),
        ("kernel_ms", "kernel_ms"),
        ("query_prep_ms", "query_prep_ms"),
    ):
        runs = [latency_samples(report, case_field) for report in reports]
        values = [value for run in runs for value in run]
        mean_ci = hierarchical_bootstrap_interval(
            runs,
            statistics.fmean,
            samples=samples,
            confidence=confidence,
            seed=seed,
        )
        p95_ci = hierarchical_bootstrap_interval(
            runs,
            lambda sample: percentile(sample, 0.95),
            samples=samples,
            confidence=confidence,
            seed=seed + 1,
        )
        metrics[output_name] = {
            "observations": len(values),
            "mean": statistics.fmean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "mean_confidence_interval": list(mean_ci),
            "p95_confidence_interval": list(p95_ci),
        }
    return {
        "variant": variant,
        "quality": quality,
        "repetitions": len(reports),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.run_root.glob("run-*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "variant" in report and "cases" in report:
            grouped[report["variant"]["name"]].append(report)
    if not grouped:
        raise ValueError("no run-*/variant.json reports found")
    payload = {
        "version": 1,
        "results": {
            name: summarize(
                reports,
                samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed,
            )
            for name, reports in sorted(grouped.items())
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Repeated TileMaxSim benchmark",
        "",
        "Intervals use deterministic hierarchical bootstrap resampling over runs and measured queries.",
        "",
        "| Variant | Runs | Observations | Mean ms (CI) | p50 | p95 (CI) | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["results"].items():
        metric = result["metrics"]["latency_ms"]
        mean_low, mean_high = metric["mean_confidence_interval"]
        p95_low, p95_high = metric["p95_confidence_interval"]
        lines.append(
            f"| {name} | {result['repetitions']} | {metric['observations']} | "
            f"{metric['mean']:.2f} [{mean_low:.2f}, {mean_high:.2f}] | "
            f"{metric['p50']:.2f} | {metric['p95']:.2f} "
            f"[{p95_low:.2f}, {p95_high:.2f}] | {metric['p99']:.2f} |"
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
