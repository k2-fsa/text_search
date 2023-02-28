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
#  ctest --verbose -R levenshtein_distance_test_py

import unittest
import numpy as np

from textsearch import levenshtein_distance


class TestLevenshtein(unittest.TestCase):
    def test_levenshtein_distance(self):
        query = np.array([1,2,3,4], dtype=np.int32)
        target = np.array([1,5,3,4,6,7,1,2,4], dtype=np.int32)
        distance, alignments = levenshtein_distance(query, target)
        self.assertTrue(distance == 1)
        self.assertTrue(alignments[0] == (3, '01010101'))
        self.assertTrue(alignments[1] == (8, '0101101'))

if __name__ == "__main__":
    unittest.main()
