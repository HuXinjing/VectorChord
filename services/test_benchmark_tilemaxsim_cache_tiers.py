# This software is licensed under a dual license model:
# GNU Affero General Public License v3 (AGPLv3) or Elastic License v2 (ELv2).
# Copyright (c) 2026 Hu Xinjing

import unittest

from services.benchmark_tilemaxsim_cache_tiers import delta, labeled_delta


class CacheTierBenchmarkTest(unittest.TestCase):
    def test_metric_deltas_are_scoped_by_prefix_and_label(self):
        before = {
            'cache{event="hit"}': 2.0,
            'cache{event="miss"}': 3.0,
            "other": 100.0,
        }
        after = {
            'cache{event="hit"}': 7.0,
            'cache{event="miss"}': 4.0,
            "other": 200.0,
        }
        self.assertEqual(delta(before, after, "cache"), 6.0)
        self.assertEqual(labeled_delta(before, after, "cache", 'event="hit"'), 5.0)


if __name__ == "__main__":
    unittest.main()
