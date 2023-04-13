# Copyright      2023   Xiaomi Corp.       (author: Wei Kang)
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

import numpy as np
from typing import List, Tuple
from _fasttextsearch import (
    get_longest_increasing_pairs as _get_longest_increasing_pairs,
)


def get_longest_increasing_pairs(
    seq1: np.ndarray, seq2: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Get the longest increasing pairs for given sequences.
    See https://github.com/danpovey/text_search/issues/21 for more details.

    Suppose seq1 is [i1, i2, i3... iN] and seq2 is [j1, j2, j3... jN], this
    function returns the  longest increasing pairs: (i1, j1), (i2, j2), ... (iN, jN)
    such that i1 <= i2 <= ... <= iN, and j1 <= j2 <= ... <= jN.

    Args:
      seq1:
        The first sequence.
      seq2:
        The second sequence.

    >>> import numpy as np
    >>> from textsearch import get_longest_increasing_pairs
    >>> seq1 = np.array([0, 1, 1, 2, 2, 3, 4, 5, 6], dtype=np.int32)
    >>> seq2 = np.array([9, 7, 8, 9, 6, 7, 10, 12, 8], dtype=np.int64)
    >>> get_longest_increasing_pairs(seq1=seq1, seq2=seq2)
    [(1, 7), (1, 8), (2, 9), (4, 10), (5, 12)]

    """
    assert seq1.ndim == 1, seq1.ndim
    assert seq2.ndim == 1, seq2.ndim
    assert seq1.size == seq2.size, (seq1.size, seq2.size)

    # The sequences are required to be contiguous int32 array in C++.
    seq1_int32 = np.ascontiguousarray(seq1, dtype=np.int32)
    seq2_int32 = np.ascontiguousarray(seq2, dtype=np.int32)

    return _get_longest_increasing_pairs(seq1_int32, seq2_int32)
