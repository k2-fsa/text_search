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

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
from .suffix_array import create_suffix_array
from .datatypes import SourcedText


def create_suffix_array_from_sourced_text(text: SourcedText) -> np.ndarray:
    extend_text = np.concatenate(
        (
            text.binary_text,
            np.array(
                [np.max(text.binary_text) + 1, 0, 0, 0], dtype=text.binary_text.dtype
            ),
        ),
        axis=0,
    )
    return create_suffix_array(extend_text)


def find_candidate_matches(
    close_matches: np.ndarray,
    text: SourcedText,
    length_ratio: float = 1.5,
    num_candidates: int = 1,
) -> List[List[Tuple[int, int]]]:
    """
    Find candidate regions in reference document that could be good matches for
    each query document.

    Args:
       close_matches: an np.ndarray of shape (2*tot_query_symbols,) as returned by
          find_close_matches(), indicating two close matches within the reference
          text for each symbol in the query documents.
       text:  The SourcedText corresponding to the query and reference documents
          combined; needed for its `doc` member which can be assumed to be an np.ndarray.
       length_ratio: indicates the maximum candidate-region length, which will be reduced
          if the matching reference document was shorter than this.
       num_candidates:  the number of candidate regions to find for each query
          document.
    Returns:
       A list, one per query document, of close matches, where each close match
       is a (begin, end) position within `text`.
    """
    assert close_matches.ndim == 2, close_matches.ndim
    tot_query_symbols, num_close_matches = close_matches.shape
    assert text.binary_text.size > tot_query_symbols, (text.binary_text.size, tot_query_symbols)

    # TODO:can we assert the query docs are sorted by doc ids ?
    num_query_docs = np.unique(text.doc[0:tot_query_symbols]).size

    assert num_query_docs == text.doc[tot_query_symbols], (
        num_query_docs,
        text.doc[tot_query_symbols],
    )

    num_query_docs = text.doc[tot_query_symbols]

    row_splits = text.doc_splits

    candidate_matches = []

    for q in range(num_query_docs):
        matches_start_pos = row_splits[q]
        matches_end_pos = row_splits[q + 1]
        current_query_len = row_splits[q + 1] - row_splits[q]
        reference_chunk_length = current_query_len * length_ratio

        current_close_matches = np.sort(
            close_matches[matches_start_pos:matches_end_pos, :].flatten()
        )

        # (start pos in reference, end pos in reference, hits)
        current_candidates = [(0, 0, 0)] * num_candidates
        j = 1
        for i in range(current_close_matches.size):
            doc_id = text.doc[current_close_matches[i]]
            pos_id = text.pos[current_close_matches[i]]
            while (
                j < current_close_matches.size
                and text.doc[current_close_matches[j]] == doc_id
                and text.pos[current_close_matches[j]]
                < (pos_id + reference_chunk_length)
            ):
                j += 1

            candidate = (
                current_close_matches[i],
                current_close_matches[j - 1] + 1,
                j - i,
            )

            min_score = candidate[2]
            min_index = None
            overlap = False
            for i, c in enumerate(current_candidates):
                if c[2] < min_score:
                    min_score = c[2]
                    min_index = i
                if candidate[0] < c[1] and candidate[2] > c[2]:
                    current_candidates[i] = candidate
                    overlap = True
                    break
            if not overlap:
                if len(current_candidates) < num_candidates:
                    current_candidates.append(candidate)
                else:
                    if min_index is not None:
                        current_candidates[min_index] = candidate
        candidate_matches.append([(c[0], c[1]) for c in current_candidates])
    return candidate_matches
