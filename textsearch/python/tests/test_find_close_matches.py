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
#  ctest --verbose -R find_close_matches_test_py

import unittest
import numpy as np

from textsearch import create_suffix_array, find_close_matches


class TestFindCloseMatches(unittest.TestCase):
    def test_find_close_matches(self):
        """
        The suffix array contains the information below, the first column is
        the index into the original texts (the only thing suffix array store),
        the following column tell the suffix and which type (query or reference)
        this token belongs to.

        6, type : query, suffix : alloiholloyouyouhellome
        1, type : query, suffix : ellohalloiholloyouyouhellome
        23, type : reference, suffix : ellome
        28, type : reference, suffix : e
        5, type : query, suffix : halloiholloyouyouhellome
        0, type : query, suffix : hellohalloiholloyouyouhellome
        22, type : reference, suffix : hellome
        11, type : reference, suffix : holloyouyouhellome
        10, type : reference, suffix : iholloyouyouhellome
        2, type : query, suffix : llohalloiholloyouyouhellome
        7, type : query, suffix : lloiholloyouyouhellome
        24, type : reference, suffix : llome
        13, type : reference, suffix : lloyouyouhellome
        3, type : query, suffix : lohalloiholloyouyouhellome
        8, type : query, suffix : loiholloyouyouhellome
        25, type : reference, suffix : lome
        14, type : reference, suffix : loyouyouhellome
        27, type : reference, suffix : me
        4, type : query, suffix : ohalloiholloyouyouhellome
        9, type : query, suffix : oiholloyouyouhellome
        12, type : reference, suffix : olloyouyouhellome
        26, type : reference, suffix : ome
        20, type : reference, suffix : ouhellome
        17, type : reference, suffix : ouyouhellome
        15, type : reference, suffix : oyouyouhellome
        21, type : reference, suffix : uhellome
        18, type : reference, suffix : uyouhellome
        19, type : reference, suffix : youhellome
        16, type : reference, suffix : youyouhellome
        29, type : reference, suffix :
        """
        queries = ["hello", "hallo"]
        documents = ["iholloyou", "youhellome"]

        # texts will be : "hellohalloiholloyouyouhellome"
        texts = "".join(queries) + "".join(documents)
        texts_array = np.fromstring(texts, dtype=np.int8)
        suffix_array = create_suffix_array(texts_array)

        query_len = len("".join(queries))
        output = find_close_matches(suffix_array, query_len, num_close_matches=2)

        # Take the first token of query as an example, it is 'h', first we
        # will find the element 0 in suffix_array, from the comment lines above,
        # it's the sixth element of suffix_array, the nearest
        # reference before and after it are the fourth and seventh element, so
        # its close match references are [28, 22].
        # You can check the other tokens the same way.
        expected_output = np.array(
            [
                [28, 22],
                [28, 23],
                [10, 24],
                [13, 25],
                [27, 12],
                [28, 22],
                [28, 23],
                [10, 24],
                [13, 25],
                [27, 12],
            ],
            dtype=np.int32,
        )
        self.assertTrue((output == expected_output).all())

        output = find_close_matches(suffix_array, query_len, num_close_matches=4)
        expected_output = np.array(
            [
                [23, 28, 22, 11],
                [28, 28, 23, 28],
                [11, 10, 24, 13],
                [24, 13, 25, 14],
                [14, 27, 12, 26],
                [23, 28, 22, 11],
                [28, 28, 23, 28],
                [11, 10, 24, 13],
                [24, 13, 25, 14],
                [14, 27, 12, 26],
            ],
            dtype=np.int32,
        )
        self.assertTrue((output == expected_output).all())


if __name__ == "__main__":
    unittest.main()
