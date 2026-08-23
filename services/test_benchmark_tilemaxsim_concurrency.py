# This software is licensed under a dual license model:
# GNU Affero General Public License v3 (AGPLv3) or Elastic License v2 (ELv2).
# Copyright (c) 2026 Hu Xinjing

import unittest

from services.benchmark_tilemaxsim_concurrency import (
    crossed_priority,
    latency_summary,
    maximum_consecutive,
)


class ConcurrencyBenchmarkTest(unittest.TestCase):
    def test_latency_summary_includes_tail_percentiles(self):
        result = latency_summary([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(result["p50"], 2.0)
        self.assertEqual(result["p95"], 100.0)
        self.assertEqual(result["p99"], 100.0)

    def test_maximum_consecutive_completion_domain(self):
        self.assertEqual(maximum_consecutive(["a", "b", "b", "a"]), 2)
        self.assertEqual(maximum_consecutive([]), 0)

    def test_priorities_are_crossed_with_domains(self):
        priorities = [-10, 0, 10, 50]
        assignments = [crossed_priority(index, 4, priorities) for index in range(16)]
        self.assertEqual(assignments, [-10] * 4 + [0] * 4 + [10] * 4 + [50] * 4)


if __name__ == "__main__":
    unittest.main()
