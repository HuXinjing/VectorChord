"""Estimate the stable CPU/GPU burst crossover from measured arrival latency."""
from __future__ import annotations

import math
import statistics


def estimate_crossover(samples, *, margin=1.2):
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("invalid_crossover_margin")
    groups = {}
    for row in samples:
        if row["arm"] not in ("cpu", "gpu"): continue
        n, elapsed = row["count"], row["arrival_p95_ms"]
        if type(n) is not int or n < 1 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid_calibration_sample")
        groups.setdefault(n, {"cpu": [], "gpu": []})[row["arm"]].append(elapsed)
    points = []
    for n in sorted(groups):
        if min(map(len, groups[n].values())) < 5:
            raise ValueError("insufficient_calibration_repeats")
        cpu, gpu = (statistics.median(groups[n][a]) for a in ("cpu", "gpu"))
        points.append({"count": n, "cpu_ms": cpu, "gpu_ms": gpu, "difference_ms": cpu-gpu})
    # Do not extrapolate beyond observed counts or claim a crossing from a
    # transient win that reverses again at a larger measured batch.
    for i in range(1, len(points)):
        left, right = points[i-1], points[i]
        if (left["difference_ms"] <= 0 < right["difference_ms"]
            and len(points)-i >= 2 and all(p["difference_ms"] > 0 for p in points[i:])):
            fraction = -left["difference_ms"] / (right["difference_ms"]-left["difference_ms"])
            crossing = left["count"] + fraction * (right["count"]-left["count"])
            return {"crossover": crossing, "margin": margin,
                    "threshold": math.floor(crossing*margin)+1, "points": points}
    raise ValueError("stable_crossover_not_observed")
