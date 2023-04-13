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
    assert seq1.ndim == 1, seq1.ndim
    assert seq2.ndim == 1, seq2.ndim

    seq1_int32 = np.ascontiguousarray(seq1, dtype=np.int32)
    seq2_int32 = np.ascontiguousarray(seq2, dtype=np.int32)

    return _get_longest_increasing_pairs(seq1_int32, seq2_int32)
