#!/usr/bin/env python3
#
# Copyright      2024  Xiaomi Corp.       (authors: Wei Kang)
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

from textsearch.utils import is_overlap


class TestOverlap(unittest.TestCase):
    def test_is_overlap(self):
        candidates = [
            [20, 30],
            [15, 25],
            [10, 21.1],
            [1, 10],
            [60, 70],
            [65, 73],
            [68.5, 85],
            [25, 35],
            [45, 55],
            [20, 25],
            [21, 25],
            [34.5, 46.5],
            [35, 46.1],
            [25, 35],
            [26, 34],
            [44, 70.5],
        ]
        selected_ranges: List[Tuple[float, float]] = []
        selected_indexes: List[int] = []
        segments = []
        overlapped_segments = []
        for r in candidates:
            status, index = is_overlap(
                selected_ranges,
                selected_indexes,
                query=(r[0], r[1]),
                segment_index=len(segments),
                overlap_ratio=0.1,
            )
            if status:
                if index is not None:
                    overlapped_segments.append(index)
                    segments.append(r)
            else:
                segments.append(r)
        for index in sorted(overlapped_segments, reverse=True):
            segments.pop(index)
        expected_segments = [
            [10, 21.1],
            [1, 10],
            [68.5, 85],
            [25, 35],
            [21, 25],
            [35, 46.1],
        ]
        assert segments == expected_segments


if __name__ == "__main__":
    unittest.main()
