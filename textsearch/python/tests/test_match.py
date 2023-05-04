#!/usr/bin/env python3
#
# Copyright      2023  Xiaomi Corp.       (authors: Wei Kang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R match_test_py

import unittest
import numpy as np

from textsearch import get_longest_increasing_pairs


class TestMatch(unittest.TestCase):
    def test_get_longest_increasing_pairs(self):
        for dtype in [np.int32, np.int64]:
            seq1 = np.array([0, 1, 1, 2, 2, 3, 4, 5, 6], dtype=dtype)
            seq2 = np.array([9, 7, 8, 9, 6, 7, 10, 12, 8], dtype=dtype)
            expected = [(1, 7), (1, 8), (2, 9), (4, 10), (5, 12)]
            result = get_longest_increasing_pairs(seq1=seq1, seq2=seq2)
            for i, r in enumerate(result):
                self.assertTrue(r == expected[i])


if __name__ == "__main__":
    unittest.main()
