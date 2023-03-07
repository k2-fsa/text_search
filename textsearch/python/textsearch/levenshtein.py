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


def get_nice_alignments(
    alignments: List[Tuple[int, str]], query: np.ndarray, target: np.ndarray
) -> List[str]:
    """
    Get the alignment of the matched segments.

    Args:
      alignments:
        The alignments information returned by the `levenshtein_distance`.
      query:
        The query sequence.
      target:
        The target sequence.

    Returns:
      Return a list of alignment string, it has the same length as alignments.
      See the following example for more details, the first line of the alignment
      is query, the second line is the error types, the third line is the target
      segment. '*' means empty, '+' means insertion, '-' means deletion, '#' means
      replacement, '|' means equal.

    >>> from textsearch import get_nice_alignments, levenshtein_distance
    >>> import numpy as np
    >>> query = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> target = np.array([1, 5, 3, 4, 6, 7, 1, 2, 4], dtype=np.int32)
    >>> distance, alignments = levenshtein_distance(query, target)
    >>> aligns = get_nice_alignments(alignments, a, b)
    >>> print (aligns[0])
    1 2 3 4
    | # | |
    1 5 3 4
    >>> print (aligns[1])
    1 2 3 4
    | | + |
    1 2 * 4
    """
    results = []
    for align in alignments:
        j = align[0]
        i = len(query) - 1
        k = len(align[1]) - 1
        qs = ""
        ms = ""
        ts = ""
        while k - 1 >= 0:
            if align[1][k] == "0":
                # a deletion error
                ts_ = f"{target[j]} "
                sl = len(ts_)

                qs += f"{'*':>{sl}}"
                ms += f"{'-':>{sl}}"
                ts += f"{ts_[::-1]:>{sl}}"
                j = j - 1
                k = k - 1
            elif align[1][k] == "1" and align[1][k - 1] == "0":
                # correct or a substitution error
                qs_ = f"{query[i]} "
                ts_ = f"{target[j]} "
                sl = max(len(qs_), len(ts_))

                qs += f"{qs_[::-1]:>{sl}}"
                ts += f"{ts_[::-1]:>{sl}}"
                if query[i] == target[j]:
                    ms += f"{'|':>{sl}}"
                else:
                    ms += f"{'#':>{sl}}"
                j = j - 1
                i = i - 1
                k = k - 2
            else:
                # an insertion error
                assert align[1][k] == "1", align[1][k]
                qs_ = f"{query[i]} "
                sl = len(qs_)

                qs += f"{qs_[::-1]:>{sl}}"
                ms += f"{'+':>{sl}}"
                ts += f"{'*':>{sl}}"
                i = i - 1
                k = k - 1
        if k > 0:
            assert k == 1, k
            if align[1][k] == "0":
                # a deletion error
                ts_ = f"{target[j]} "
                sl = len(ts_)

                qs += f"{'*':>{sl}}"
                ms += f"{'-':>{sl}}"
                ts += f"{ts_[::-1]:>{sl}}"
            else:
                # an insertion error
                assert align[1][k] == "1", align[1][k]
                qs_ = f"{query[i]} "
                sl = len(qs_)

                qs += f"{qs_[::-1]:>{sl}}"
                ms += f"{'+':>{sl}}"
                ts += f"{'*':>{sl}}"
        results.append("\n".join([qs[::-1], ms[::-1], ts[::-1]]))
    return results
