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
    alignments: List[Tuple[int, int, str]],
    query: np.ndarray,
    target: np.ndarray,
) -> List[str]:
    """
    Get the alignment of the matched segments.

    Args:
      alignments:
        The alignments information returned by the :func:`levenshtein_distance`.
      query:
        The query sequence.
      target:
        The target sequence.

    Returns:
      Return a list of alignment string, it has the same length as alignments.
      See the following example for more details, the first line of the alignment
      is query, the second line is the error types, the third line is the target
      segment. The following symbols are used to represent the error types:

        - ``*``: empty
        - ``+``: insertion
        - ``-``: deletion
        - ``#``: substitution
        - ``|``: correct.

    **Example 1/2**

    >>> from textsearch import get_nice_alignments, levenshtein_distance
    >>> import numpy as np
    >>> query = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> target = np.array([1, 5, 3, 4, 6, 7, 1, 2, 4], dtype=np.int32)
    >>> distance, alignments = levenshtein_distance(query, target)
    >>> distance
    1
    >>> alignments
    [(0, 3, 'CSCC'), (6, 8, 'CCIC')]
    >>> aligns = get_nice_alignments(alignments, query, target)
    >>> repr(aligns)
    "['1 2 3 4 \\n| # | | \\n1 5 3 4 ', '1 2 3 4 \\n| | + | \\n1 2 * 4 ']"
    >>> print (aligns[0])
    1 2 3 4
    | # | |
    1 5 3 4
    >>> print (aligns[1])
    1 2 3 4
    | | + |
    1 2 * 4

    **Example 2/2**

        .. literalinclude:: ./code/edit-distance.py
    """
    results = []
    for align in alignments:
        j = align[0]
        i = 0
        ali_seq = align[2]
        qs = ""
        ms = ""
        ts = ""
        for k in ali_seq:
            if k == "D":
                # a deletion error
                ts_ = f"{target[j]} "
                sl = len(ts_)

                qs += f"{'*':{sl}}"
                ms += f"{'-':{sl}}"
                ts += f"{ts_:{sl}}"
                j = j + 1
            elif k == "S" or k == "C":
                # correct or a substitution error
                qs_ = f"{query[i]} "
                ts_ = f"{target[j]} "
                sl = max(len(qs_), len(ts_))

                qs += f"{qs_:{sl}}"
                ts += f"{ts_:{sl}}"
                if k == "C":
                    ms += f"{'|':{sl}}"
                else:
                    ms += f"{'#':{sl}}"
                j = j + 1
                i = i + 1
            else:
                # an insertion error
                assert k == "I", k
                qs_ = f"{query[i]} "
                sl = len(qs_)

                qs += f"{qs_:{sl}}"
                ms += f"{'+':{sl}}"
                ts += f"{'*':{sl}}"
                i = i + 1
        results.append("\n".join([qs, ms, ts]))
    return results
