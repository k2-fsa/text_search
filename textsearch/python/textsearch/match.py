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
import os
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Union

import numpy as np

from _fasttextsearch import (
    get_longest_increasing_pairs as _get_longest_increasing_pairs,
    levenshtein_distance,
)
from .suffix_array import create_suffix_array
from .datatypes import SourcedText


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


def _break_trace(
    sourced_text: SourcedText,
    trace: List[Tuple[int, int]],
    query_length: int,
    max_segment_length: int = 5000,
    length_ratio: float = 1.5,
) -> List[Tuple[int, int, int, int]]:
    prev_break_point = 0  # index in trace
    candidate_segments: List[Tuple[int, int]] = []  # start, end index in trace
    for i in range(1, len(trace)):
        # The query and reference distance between current trace and preceding
        # trance.
        gap = trace[i][0] - trace[i - 1][0], trace[i][1] - trace[i - 1][1]
        if gap[0] < gap[1]:
            # +1 for smoothing (avoid div zero error)
            ratio = (trace[i][1] - trace[prev_break_point][1] + 1) / (
                trace[i][0] - trace[prev_break_point][0] + 1
            )
            # break on current trace
            if ratio > length_ratio:
                candidate_segments.append((prev_break_point, i))
                prev_break_point = i

    candidate_segments.append((prev_break_point, len(trace)))
    # Find the longest query sequence
    max_length = -1
    max_item = (0, len(trace))
    for c in candidate_segments:
        segment_length = trace[c[1] - 1][0] - trace[c[0]][0]
        if segment_length > max_length:
            max_length = segment_length
            max_item = c

    # start, end index in query and reference
    # [(start_query, end_query, start_reference, end_reference)]
    segments: List[Tuple[int, int, int, int]] = []

    def add_segments(
        query_start, query_end, target_start, target_end, max_segment_length, segments
    ):
        num_chunk = (query_end - query_start) // max_segment_length
        if num_chunk >= 2:
            for i in range(num_chunk - 1):
                segments.append(
                    (
                        query_start,
                        query_start + max_segment_length,
                        target_start,
                        target_start + max_segment_length,
                    )
                )
                query_start += max_segment_length
                target_start += max_segment_length
        segments.append((query_start, query_end, target_start, target_end))

    # index in query and reference
    doc_id = sourced_text.doc[trace[max_item[0]][1]]
    base = sourced_text.doc_splits[doc_id]

    prev_target = trace[max_item[0]][1] - trace[max_item[0]][0]
    prev_break_point = (
        0,
        prev_target if prev_target >= base else base,
    )
    for ind in range(max_item[0], max_item[1]):
        if trace[ind][0] - prev_break_point[0] > max_segment_length:
            if ind == max_item[0]:
                continue
            else:
                query_start = prev_break_point[0]
                query_end = trace[ind - 1][0]
                target_start = prev_break_point[1]
                target_end = trace[ind - 1][1]

                ratio = (target_end - target_start) / (query_end - query_start)
                half = (length_ratio - 1) / 2
                if ratio < 1 - half or ratio > 1 + half:
                    continue

                prev_break_point = (query_end, target_end)
                add_segments(
                    query_start,
                    query_end,
                    target_start,
                    target_end,
                    max_segment_length,
                    segments,
                )

    query_start, target_start = prev_break_point
    query_end = query_length
    target_end = target_start + (query_length - query_start)
    next_base = sourced_text.doc_splits[doc_id + 1]
    target_end = target_end if target_end <= next_base else next_base
    if query_end - query_start < max_segment_length:
        if segments:
            segments[-1] = (
                segments[-1][0],
                query_end,
                segments[-1][2],
                target_end,
            )
        else:
            segments.append(
                (
                    query_start,
                    query_end,
                    target_start,
                    target_end,
                )
            )
    else:
        add_segments(
            query_start,
            query_end,
            target_start,
            target_end,
            max_segment_length,
            segments,
        )
    return segments


def get_alignments(
    sourced_text: SourcedText,
    close_matches: np.ndarray,
    segment_length: int = 5000,
    target_length_ratio: float = 1.5,
    num_threads: int = -1,
) -> List[Tuple[int, int, str]]:
    """
    Get levenshtein alignment for each query document.


    """
    tot_query_symbols, num_close_matches = close_matches.shape
    num_queries = sourced_text.doc[tot_query_symbols]

    row_splits = sourced_text.doc_splits

    extra_target_length = int(segment_length * (target_length_ratio - 1) / 2)

    logging.info("Getting matching trace.")
    arguments = []
    for q in range(num_queries):
        query_start = row_splits[q]
        query_end = row_splits[q + 1]
        query_len = row_splits[q + 1] - row_splits[q]

        seq1 = np.arange(query_len).reshape(-1, 1)
        seq1 = np.tile(seq1, num_close_matches).flatten()

        seq2 = close_matches[query_start:query_end, :].flatten()
        trace = get_longest_increasing_pairs(seq1, seq2)

        # In the algorithm of `find_close_matches`, `sourced_text.binary_text.size - 1`
        # means no close_matches
        trim_pos = len(trace) - 1
        while trace[trim_pos][1] == sourced_text.binary_text.size - 1:
            trim_pos -= 1
        trace = trace[0:trim_pos]

        left = 0
        right = len(trace) - 1

        skip_query = False
        while sourced_text.doc[trace[left][1]] != sourced_text.doc[trace[right][1]]:
            left += 1
            right -= 1
            if left >= right:
                skip_query = True
                break
        if skip_query:
            logging.warning(
                f"Skipping query, as the close matched segment is too short."
            )
            continue

        doc_id = sourced_text.doc[left]
        while left >= 1 and sourced_text.doc[trace[left - 1][1]] == doc_id:
            left -= 1
        while (
            right < len(trace) - 1 and sourced_text.doc[trace[right + 1][1]] == doc_id
        ):
            right += 1

        logging.info(f"trace length : {len(trace)}, left : {left}, right : {right}")

        if trace[right][0] - trace[left][0] < 0.5 * query_len:
            logging.warning(
                f"Skipping query, less than half of query matched by close_matches."
            )
            continue
        if trace[right][1] - trace[left][1] < 0.5 * query_len:
            logging.warning(
                f"Skipping query, less than half of reference matched by close_matches."
            )
            continue

        trace = trace[left : right + 1]

        if q == 0:
            gaps = []
            for i in range(1, len(trace)):
                gaps.append(
                    (trace[i][0] - trace[i - 1][0], trace[i][1] - trace[i - 1][1])
                )
            logging.info(f"gaps : {gaps}")

        segments = _break_trace(
            sourced_text, trace, query_len, segment_length, target_length_ratio
        )
        # logging.info(f"segments : {segments}")

        for i, seg in enumerate(segments):
            arguments.append((sourced_text, q, seg, extra_target_length))
    logging.info("Getting matching trace done.")

    def levenshtein_worker(sourced_text, query_index, segment, extra_target_length):
        row_splits = sourced_text.doc_splits
        query_start = row_splits[query_index]
        query = sourced_text.binary_text[
            query_start + segment[0] : query_start + segment[1]
        ]
        doc_id = sourced_text.doc[segment[2]]
        base = row_splits[doc_id]
        next_base = row_splits[doc_id + 1]
        target_start = segment[2] - extra_target_length
        target_start = target_start if target_start >= base else base
        target_end = segment[3] + extra_target_length
        target_end = target_end if target_end <= next_base else next_base
        target = sourced_text.binary_text[target_start:target_end]
        alignment = levenshtein_distance(query=query, target=target)
        return {
            "query_index": query_index,
            "target_pos": (target_start, target_end),
            "alignment": alignment,
        }

    real_num_threads = (
        min(len(arguments), os.cpu_count()) if num_threads <= 0 else num_threads
    )
    with ThreadPool(real_num_threads) as pool:
        logging.info("Matching with levenshtein.")
        async_results = pool.starmap_async(levenshtein_worker, arguments)
        results = async_results.get()

        # combining the alignments together
        alignments = [None] * num_queries
        prev_target_end = 0
        for i, res in enumerate(results):
            target_align_start, target_align_end, align_str = res["alignment"][1][0]
            if alignments[res["query_index"]] is None:
                prev_target_end = res["target_pos"][0] + target_align_end + 1
                alignments[res["query_index"]] = [
                    res["target_pos"][0] + target_align_start,
                    prev_target_end,
                    align_str,
                ]
            else:
                assert alignments[res["query_index"]] is not None

                diff = res["target_pos"][0] + target_align_start - prev_target_end
                if abs(diff) > segment_length * 0.01:
                    logging.warning(
                        f"Too many deletions or insertions on the boundaries of segments,"
                        f"please try a large target_length_ratio, current ratio "
                        f"{target_length_ratio}. "
                        f"diff : {diff} "
                        f"query_index : {res['query_index']}"
                    )
                if diff == 0:
                    alignments[res["query_index"]][2] += align_str
                elif diff > 0:
                    alignments[res["query_index"]][2] += "D" * diff + align_str
                else:
                    align_str = "I" * (-diff) + align_str[-diff:]
                    alignments[res["query_index"]][2] += align_str
                prev_target_end = res["target_pos"][0] + target_align_end + 1
                alignments[res["query_index"]][1] = prev_target_end
        logging.info("Matching with levenshtein done.")
        return alignments


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
    assert text.binary_text.size > tot_query_symbols, (
        text.binary_text.size,
        tot_query_symbols,
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
