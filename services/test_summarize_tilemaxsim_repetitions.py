# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

import unittest

from services.summarize_tilemaxsim_repetitions import (
    bootstrap_interval,
    hierarchical_bootstrap_interval,
    latency_samples,
    summarize,
)


def report(latencies, *, quality=0.75):
    return {
        "variant": {"name": "int8", "encoding": "int8"},
        "quality": {"recall_at_10": quality},
        "warmup_queries": 1,
        "cases": [
            {
                "measured": index > 0,
                "latency_ms": value,
                "transfer_ms": value / 2,
                "kernel_ms": value / 4,
                "query_prep_ms": value / 8,
            }
            for index, value in enumerate(latencies)
        ],
    }


class RepeatedSummaryTest(unittest.TestCase):
    def test_bootstrap_is_seeded_and_ordered(self):
        first = bootstrap_interval(
            [1.0, 2.0, 3.0], sum, samples=100, confidence=0.95, seed=7
        )
        second = bootstrap_interval(
            [1.0, 2.0, 3.0], sum, samples=100, confidence=0.95, seed=7
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_hierarchical_bootstrap_is_seeded(self):
        first = hierarchical_bootstrap_interval(
            [[1.0, 2.0], [10.0, 20.0]],
            sum,
            samples=100,
            confidence=0.95,
            seed=9,
        )
        second = hierarchical_bootstrap_interval(
            [[1.0, 2.0], [10.0, 20.0]],
            sum,
            samples=100,
            confidence=0.95,
            seed=9,
        )
        self.assertEqual(first, second)

    def test_warmups_are_excluded_from_samples(self):
        self.assertEqual(latency_samples(report([1000.0, 2.0, 3.0]), "latency_ms"), [2.0, 3.0])

    def test_repetitions_are_pooled_without_changing_quality(self):
        result = summarize(
            [report([1000.0, 2.0, 4.0]), report([1000.0, 6.0, 8.0])],
            samples=200,
            confidence=0.95,
            seed=11,
        )
        self.assertEqual(result["repetitions"], 2)
        self.assertEqual(result["metrics"]["latency_ms"]["observations"], 4)
        self.assertEqual(result["metrics"]["latency_ms"]["mean"], 5.0)

    def test_quality_drift_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "quality metrics differ"):
            summarize(
                [report([1.0, 2.0]), report([1.0, 2.0], quality=0.5)],
                samples=20,
                confidence=0.95,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
