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
#  ctest --verbose -R suffix_array_test_py

import unittest
import numpy as np

from textsearch import create_suffix_array


class TestSuffixArray(unittest.TestCase):
    def test_create_suffix_array(self):
        for dtype in [np.uint8, np.int8, np.uint16, np.int16]:
            array = np.array([3, 2, 1], dtype=dtype)
            suffix_array = create_suffix_array(array)
            expected_array = np.array([2, 1, 0, 3], dtype=np.int32)
            self.assertTrue((suffix_array == expected_array).all())
            self.assertTrue(suffix_array.dtype == np.int32)


if __name__ == "__main__":
    unittest.main()
