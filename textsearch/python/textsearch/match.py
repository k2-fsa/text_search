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
from bisect import bisect_left
from dataclasses import dataclass
from heapq import heappush, heappop
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Tuple, Set, Union

import numpy as np

from _fasttextsearch import (
    get_longest_increasing_pairs as _get_longest_increasing_pairs,
    levenshtein_distance,
)
from .suffix_array import create_suffix_array
from .datatypes import SourcedText, Transcript
from .utils import is_overlap, is_punctuation, row_ids_to_row_splits

PUNCTUATION_END = set([ord(i) for i in ".!?"])


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
    max_segment_length: int = 5000,
    length_ratio: float = 1.1,
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
            half = (length_ratio - 1) / 2
            if ratio < 1 - half or ratio > 1 + half:
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
        if num_chunk > 0:
            for i in range(num_chunk):
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
        if segments and query_end - query_start < max_segment_length // 4:
            segments[-1] = (
                segments[-1][0],
                query_end,
                segments[-1][2],
                target_end,
            )
        else:
            segments.append((query_start, query_end, target_start, target_end))

    target_doc_id = sourced_text.doc[trace[max_item[0]][1]]
    target_base = sourced_text.doc_splits[target_doc_id]
    next_target_base = sourced_text.doc_splits[target_doc_id + 1]

    query_doc_id = sourced_text.doc[trace[max_item[0]][0]]
    query_base = sourced_text.doc_splits[query_doc_id]
    next_query_base = sourced_text.doc_splits[query_doc_id + 1]

    prev_target = trace[max_item[0]][1] - (trace[max_item[0]][0] - query_base)
    # index in query and reference
    prev_break_point = (
        query_base,
        prev_target if prev_target >= target_base else target_base,
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
    query_end = next_query_base

    target_end = target_start + (query_end - query_start)
    target_end = target_end if target_end <= next_target_base else next_target_base

    if query_end - query_start < max_segment_length // 4:
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


def _combine_sub_alignments(
    sourced_text: SourcedText, sub_alignments: List[Dict[str, Any]], num_queries: int
) -> List[Tuple[Tuple[int, int], List[Dict[str, Any]]]]:
    # combining the alignments together
    alignments = [None] * num_queries
    prev_target_end = 0
    for sub in sub_alignments:
        # for the global mode of levenshtein, there is only one alignment
        if sub["alignment"][0] == -1:
            logging.warning(f"Skipping empty sub segment.")
            continue
        target_align_start, target_align_end, align_str = sub["alignment"][1][0]
        query_start, query_end, target_start, target_end = sub["segment"]
        query_id = sourced_text.doc[query_start]

        if alignments[query_id] is None:
            alignments[query_id] = (
                (query_start, target_start + target_align_start),
                [],
            )
        query_doc_id = sourced_text.doc[query_start]
        query_base = sourced_text.doc_splits[query_doc_id]
        query_source = sourced_text.sources[query_doc_id]

        # aligns : [{"ref": , "hyp": , "ref_pos": , "hyp_pos": , "hyp_time":}]
        aligns = alignments[query_id][1]
        times = query_source.times if isinstance(query_source, Transcript) else None
        time_stride = 1 if query_source.binary_text.dtype == np.uint8 else 4

        query_local_index = query_start - query_base
        query_index = query_start
        target_index = target_start + target_align_start

        hyp_time = 0 if times is None else float(times[query_local_index * time_stride])
        for ali in align_str:
            if ali != "D":
                hyp_time = (
                    0
                    if times is None
                    else float(times[query_local_index * time_stride])
                )
            if ali == "I":
                aligns.append(
                    {
                        "ref": "",
                        "hyp": chr(sourced_text.binary_text[query_index]),
                        "ref_pos": int(sourced_text.pos[target_index]),
                        "hyp_pos": query_local_index,
                        "hyp_time": hyp_time,
                    }
                )
                query_local_index += 1
                query_index += 1
            elif ali == "D":
                aligns.append(
                    {
                        "ref": chr(sourced_text.binary_text[target_index]),
                        "hyp": "",
                        "ref_pos": int(sourced_text.pos[target_index]),
                        "hyp_pos": query_local_index,
                        "hyp_time": hyp_time,
                    }
                )
                target_index += 1
            else:
                assert ali == "C" or ali == "S"
                ref = chr(sourced_text.binary_text[target_index])
                hyp = chr(sourced_text.binary_text[query_index])
                # The following two asserts guarantee the levenshtein alignment
                # is correct. So DO NOT delete it.
                if ali == "C":
                    assert ref == hyp
                else:
                    assert ref != hyp
                aligns.append(
                    {
                        "ref": ref,
                        "hyp": hyp,
                        "ref_pos": int(sourced_text.pos[target_index]),
                        "hyp_pos": query_local_index,
                        "hyp_time": hyp_time,
                    }
                )
                query_local_index += 1
                query_index += 1
                target_index += 1
    return alignments


def get_alignments(
    sourced_text: SourcedText,
    close_matches: np.ndarray,
    segment_length: int = 5000,
    target_length_ratio: float = 1.1,
    num_threads: int = -1,
) -> List[Tuple[int, int, str]]:
    """
    Get levenshtein alignment for each query document.
    """
    tot_query_symbols, num_close_matches = close_matches.shape
    num_queries = sourced_text.doc[tot_query_symbols]

    row_splits = sourced_text.doc_splits

    logging.info("Getting matching trace.")
    arguments = []
    for q in range(num_queries):
        query_start = row_splits[q]
        query_end = row_splits[q + 1]
        query_len = row_splits[q + 1] - row_splits[q]

        seq1 = np.arange(query_start, query_end).reshape(-1, 1)
        seq1 = np.tile(seq1, num_close_matches).flatten()

        seq2 = close_matches[query_start:query_end, :].flatten()
        trace = get_longest_increasing_pairs(seq1, seq2)

        # In the algorithm of `find_close_matches`, `sourced_text.binary_text.size - 1`
        # means no close_matches
        trim_pos = len(trace) - 1
        while trace[trim_pos][1] == sourced_text.binary_text.size - 1:
            trim_pos -= 1
        trace = trace[0:trim_pos]

        doc_ids = sourced_text.doc[np.array([x[1] for x in trace])]
        doc_ids = doc_ids - doc_ids[0]
        doc_splits = row_ids_to_row_splits(doc_ids)
        max_num_matches = -1
        max_ranges = (0, len(trace))
        for i in range(doc_splits.size - 1):
            if doc_splits[i + 1] - doc_splits[i] < 2:
                continue
            num_matches = trace[doc_splits[i + 1] - 1][0] - trace[doc_splits[i]][0]
            if num_matches > max_num_matches:
                max_num_matches = num_matches
                max_ranges = (doc_splits[i], doc_splits[i + 1])

        if max_num_matches < 0.5 * query_len:
            logging.warning(
                f"Skipping query, less than half of query matched by close_matches."
            )
            continue

        trace = trace[max_ranges[0] : max_ranges[1]]

        segments = _break_trace(
            sourced_text, trace, segment_length, target_length_ratio
        )

        for i, seg in enumerate(segments):
            arguments.append((sourced_text, seg))
    logging.info("Getting matching trace done.")

    def levenshtein_worker(sourced_text, segment):
        query = sourced_text.binary_text[segment[0] : segment[1]]
        target = sourced_text.binary_text[segment[2] : segment[3]]
        alignment = levenshtein_distance(query=query, target=target, model="global")
        return {
            "segment": segment,
            "alignment": alignment,
        }

    real_num_threads = (
        min(len(arguments), os.cpu_count()) if num_threads <= 0 else num_threads
    )
    with ThreadPool(real_num_threads) as pool:
        logging.info("Matching with levenshtein.")
        async_results = pool.starmap_async(levenshtein_worker, arguments)
        results = async_results.get()
        logging.info("Matching with levenshtein done.")

        return _combine_sub_alignments(sourced_text, results, num_queries)


def _get_segment_candidates(
    sourced_text: SourcedText, alignment
) -> List[Tuple[int, int, float]]:
    """
    Split the long aligned sequence into smaller segments.

    we create scores for each position in the alignment, corresponding to how good
    a position it is to begin or end a split.

     - begin a split (i.e. this is first position in a segment)
        - plus score equal to log(num silence frames this
          follows, up to some limit like 4.0 sec), i.e. this element
          of the Transcript's time minus the previous element's time; or some default (4.0 sec)
          if this is the first element of the Transcript.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if this reference position follows a whitespace character.
        - good if the previous non-whitespace character was a punctuation
          character.  (Lists of whitespace and punctuation characters can probably
          be passed in, or we can use some kind of isspace for utf-8.).

     - end a split (i.e. this is the last position in a segment).
        - good if more silence follows this position.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if this reference position precedes a whitespace character.
        - good if this position is a punctuation character.


    We then create a rule to assign scores to potential segments. This consist of
    the begin-scores, plus the end-scores, plus:
      - Some kind of penalty related to the duration of the segment, e.g.
        infinity if it's over some max-duration like 30 seconds or less than a
        min-duration like 2 seconds; else, one that encourages a duration between
        5 to 20 seconds.
      - A bonus for the number of matches in the alignment.
      - A penalty for the number of errors in the alignment (could multiply this by
        some scale depending how much we don't want to have errors, but some errors
        are expected due to both ASR errors and normalization differences.)

    Next, we can do a search for a good segmentation.  You could define the problem
    as getting the highest-scoring set of segments that do not overlap.  One
    possible way to do it is as follows:
       For each begin_position in the top 10% of scores, find the 4 best-scoring end_positions
       For each end_position in the top 10% of scores, find the 4 best-scoring begin_positions
    Append the preceding 2 sets of segments to get a list of candidate segments.

    Args:
      target_source:
        A TextSource containing the matched reference.
      sourced_text:
        The SourcedText containing queries and references.
      aligns:
        Alignment information generated by `get_aligns`.

    Returns:
      Returns a list of tuple, each tuple contains the start position, end position and score of
      current segment, start position and end position are indexes in aligns.
    """
    (query_start, target_start), aligns = alignment
    target_source = sourced_text.sources[sourced_text.doc[target_start]]

    # [(index, score)]
    begin_scores: List[Tuple[int, float]] = []
    end_scores: List[Tuple[int, float]] = []

    # levenshtein errors in given regin size
    half_regin_size: int = 20
    errors_in_regin: int = sum(
        [
            1 if align["ref"] != align["hyp"] else 0
            for align in aligns[0:half_regin_size]
        ]
    )

    max_silence = 4  # seconds
    space_score = 3  # score for preceding(begin) and following(end) space
    punctuation_score = 8  # score for preceding (begin) and following(end) punctuation

    # Use cumsum to get number of matches and errors in a range efficiently
    cumsum_match = [0] * len(aligns)
    cumsum_error = [0] * len(aligns)

    selected_index = 0
    for i, align in enumerate(aligns):
        matched = align["ref"] == align["hyp"]
        cumsum_match[i] = (
            int(matched) if i == 0 else (cumsum_match[i - 1] + int(matched))
        )
        cumsum_error[i] = (
            int(not matched) if i == 0 else (cumsum_error[i - 1] + int(not matched))
        )

        # erros in regin
        if i - half_regin_size >= 0:
            if aligns[i - half_regin_size]["ref"] != aligns[i - half_regin_size]["hyp"]:
                errors_in_regin -= 1
        if i + half_regin_size < len(aligns):
            if aligns[i + half_regin_size]["ref"] != aligns[i + half_regin_size]["hyp"]:
                errors_in_regin += 1

        # only split at space position
        if align["ref"] == " " and align["hyp"] == " ":
            # silence
            prev_silence = (
                max_silence
                if i == 0
                else (align["hyp_time"] - aligns[i - 1]["hyp_time"])
            )
            prev_silence = max_silence if prev_silence > max_silence else prev_silence
            post_silence = (
                max_silence
                if i == len(aligns) - 1
                else (aligns[i + 1]["hyp_time"] - align["hyp_time"])
            )
            post_silence = max_silence if post_silence > max_silence else post_silence

            # white space
            prev_space = space_score if i == 0 or aligns[i - 1]["ref"] == " " else 0
            post_space = (
                space_score
                if i == len(aligns) - 1 or aligns[i + 1]["ref"] == " "
                else 0
            )

            # punctuation
            prev_punctuation = 0
            j = align["ref_pos"] - 1
            while j >= 0:
                if target_source.binary_text[j] == ord(" "):
                    j -= 1
                elif is_punctuation(chr(target_source.binary_text[j]), eos_only=True):
                    prev_punctuation = punctuation_score
                    break
                else:
                    break

            post_punctuation = 0
            j = align["ref_pos"] + 1
            while j < target_source.binary_text.size:
                if target_source.binary_text[j] == ord(" "):
                    j += 1
                elif is_punctuation(chr(target_source.binary_text[j]), eos_only=True):
                    post_punctuation = punctuation_score
                    break
                else:
                    break
            begin_scores.append(
                (
                    selected_index,
                    i,
                    prev_silence
                    + prev_space
                    + prev_punctuation
                    - 2 * errors_in_regin / half_regin_size,
                )
            )
            end_scores.append(
                (
                    selected_index,
                    i,
                    post_silence
                    + post_space
                    + post_punctuation
                    - 2 * errors_in_regin / half_regin_size,
                )
            )
            selected_index += 1

    sorted_begin_scores = sorted(begin_scores, key=lambda x: x[2], reverse=True)
    sorted_end_scores = sorted(end_scores, key=lambda x: x[2], reverse=True)

    top_ratio = 0.5
    top_begin = sorted_begin_scores[0 : int(len(begin_scores) * top_ratio)]
    top_end = sorted_end_scores[0 : int(len(end_scores) * top_ratio)]

    # (start, end, score)
    begin_list: List[Tuple[int, int, float]] = []
    end_list: List[Tuple[int, int, float]] = []

    num_of_best_position = 8
    max_errors_ratio = 0.25
    max_text_length = 2000
    init_duration_score = 5  # duration_score for segment between 5 ~ 20 seconds

    for item in top_begin:
        # Caution: Can only be modified with heappush and heappop, it is used as
        # the container of a heap.
        item_q = []
        ind = item[0] + 1
        while ind < len(end_scores) and end_scores[ind][1] - item[1] < max_text_length:
            score = begin_scores[item[0]][2] + end_scores[ind][2]
            # matching scores
            score += (
                3
                * (cumsum_match[end_scores[ind][1]] - cumsum_match[item[1]])
                / (end_scores[ind][1] - item[1])
            )

            # errors penalties
            total_errors = cumsum_error[end_scores[ind][1]] - cumsum_error[item[1]]
            # skipping segment with too much matching errors.
            if total_errors >= (end_scores[ind][1] - item[1]) * max_errors_ratio:
                ind += 1
                continue
            score -= 3 * (total_errors) / (end_scores[ind][1] - item[1])

            # duration scores
            duration = (
                aligns[end_scores[ind][1]]["hyp_time"] - aligns[item[1]]["hyp_time"]
            )
            duration_score = (
                float("-inf")
                if duration >= 30 or duration <= 2
                else init_duration_score
            )
            duration_score = (
                duration_score - (duration - 2)
                if duration < 5 and duration > 2
                else duration_score
            )
            duration_score = (
                duration_score - (30 - duration) / 3
                if duration > 20 and duration < 30
                else duration_score
            )

            heappush(item_q, (score + duration_score, (item[1], end_scores[ind][1])))
            if len(item_q) > num_of_best_position:
                heappop(item_q)
            ind += 1
        while item_q:
            x = heappop(item_q)
            if x[0] != float("-inf"):
                begin_list.append((x[1][0], x[1][1], x[0]))

    for item in top_end:
        # Caution: Can only be modified with heappush and heappop, it is used as
        # the container of a heap.
        item_q = []
        ind = item[0] - 1
        while ind >= 0 and item[1] - begin_scores[ind][1] < max_text_length:
            score = begin_scores[ind][2] + end_scores[item[0]][2]
            # matching scores
            score += (
                3
                * (cumsum_match[item[1]] - cumsum_match[begin_scores[ind][1]])
                / (item[1] - begin_scores[ind][1])
            )

            # errors penalties
            total_errors = cumsum_error[begin_scores[ind][1]] - cumsum_error[item[1]]
            # skipping segment with too much matching errors.
            if total_errors >= (begin_scores[ind][1] - item[1]) * max_errors_ratio:
                ind -= 1
                continue
            score -= 3 * (total_errors) / (item[1] - begin_scores[ind][1])

            # duration scores
            duration = (
                aligns[item[1]]["hyp_time"] - aligns[begin_scores[ind][1]]["hyp_time"]
            )
            duration_score = (
                float("-inf")
                if duration >= 30 or duration <= 2
                else init_duration_score
            )
            duration_score = (
                duration_score - (duration - 2)
                if duration < 5 and duration > 2
                else duration_score
            )
            duration_score = (
                duration_score - (30 - duration) / 3
                if duration > 20 and duration < 30
                else duration_score
            )

            heappush(item_q, (score + duration_score, (begin_scores[ind][1], item[1])))
            if len(item_q) > num_of_best_position:
                heappop(item_q)
            ind -= 1
        while item_q:
            x = heappop(item_q)
            if x[0] != float("-inf"):
                end_list.append((x[1][0], x[1][1], x[0]))

    candidates = begin_list + end_list
    return candidates


def split_into_segments(
    sourced_text: SourcedText, alignment
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Split a long sequence into smaller segments.

    We will create scores for each position in the alignment, corresponding to how good
    a position it is to begin or end a split. We can then create a rule to assign scores
    to potential segments. The scores would consist of the begin-scores, plus the end-scores,
    plus some kind of scores of given segment (duration, matching errors .etc).

    Args:
      sourced_text:
        The sourced_text containing queries and references.
      alignment:
    """
    (query_start, target_start), aligns = alignment

    target_source = sourced_text.sources[sourced_text.doc[target_start]]
    query_source = sourced_text.sources[sourced_text.doc[query_start]]

    # candidates : (start, end, score), start and end are indexes in aligns
    candidates = _get_segment_candidates(sourced_text=sourced_text, alignment=alignment)

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    # Handle the overlapping
    # Note: Don't modified selected_ranges, it will be manipulated in `is_overlap`
    # and will be always kept sorted.
    selected_ranges: List[Tuple[int, int]] = []
    segments = []
    for r in candidates:
        if not is_overlap(selected_ranges, (r[0], r[1])):
            segments.append(r)

    results = []
    min_text_length = 100

    preceding_context_length = 500
    for seg in segments:
        begin_pos = aligns[seg[0]]["ref_pos"]
        end_pos = aligns[seg[1]]["ref_pos"]
        start = aligns[seg[0]]["hyp_time"]
        duration = aligns[seg[1]]["hyp_time"] - aligns[seg[0]]["hyp_time"]

        hyp_begin_pos = aligns[seg[0]]["hyp_pos"]
        hyp_end_pos = aligns[seg[1]]["hyp_pos"]
        hyp = "".join(
            [chr(i) for i in query_source.binary_text[hyp_begin_pos:hyp_end_pos]]
        )

        # skipping segment that has too short text
        if len(hyp) < min_text_length:
            continue

        ref = "".join([chr(i) for i in target_source.binary_text[begin_pos:end_pos]])

        pre_index = (
            begin_pos - preceding_context_length
            if begin_pos - preceding_context_length >= 0
            else 0
        )

        pre_ref = "".join(
            [chr(i) for i in target_source.binary_text[pre_index:begin_pos]]
        )

        pre_index = (
            hyp_begin_pos - preceding_context_length
            if hyp_begin_pos - preceding_context_length > 0
            else 0
        )
        pre_hyp = "".join(
            [chr(i) for i in query_source.binary_text[pre_index:hyp_begin_pos]]
        )

        results.append(
            {
                "begin_byte": begin_pos,
                "end_byte": end_pos,
                "start_time": start,
                "duration": duration,
                "hyp": hyp,
                "ref": ref,
                "pre_ref": pre_ref,
                "pre_hyp": pre_hyp,
            }
        )
    return results


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
