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
import math
from bisect import bisect_left
from dataclasses import dataclass
from heapq import heappush, heappop
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from _textsearch import (
    get_longest_increasing_pairs as _get_longest_increasing_pairs,
    levenshtein_distance,
)
from .suffix_array import create_suffix_array
from .datatypes import SourcedText, TextSource, Transcript
from .utils import is_overlap, is_punctuation, row_ids_to_row_splits


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


def _break_query(
    sourced_text: SourcedText,
    matched_points: List[Tuple[int, int]],
    segment_length: int = 5000,
    reference_length_difference: float = 0.1,
) -> List[Tuple[int, int, int, int]]:
    """
    Break long query into small segments at the matched points with the `segment_length`
    constraint.

    Args:
      sourced_text:
        The SourcedText containing the queries and references.
      matched_points:
        A list of matched points, each item is a pair of (query index, target index)
        in sourced_text.
      segment_length:
        The expected length of the segmented piece. Note: this is not really the
        length of the segments, the segment length might be a little longer
        than `segment_length`, like `segment_length` * 1.25
      reference_length_difference:
        Because of the insertion or deletion errors, the reference sequence might be
        shorter or longer than the query, so the reference segment length can be from
        `len(query) * (1 - reference_length_difference / 2)` to
        `len(query) * (1 + reference_length_difference / 2)`.
    """
    # Firstly we will find the longest range that satisfies
    # 1 - reference_length_difference / 2 <= len(reference) / len(query)
    # <= 1 + reference_length_difference / 2
    prev_break_point = 0  # index in matched_points
    candidate_ranges: List[Tuple[int, int]] = []  # start, end indexes in matched_points
    for i in range(1, len(matched_points)):
        # The query and reference distance between current matched_point and preceding
        # matched_point.
        gap = (
            matched_points[i][0] - matched_points[i - 1][0],
            matched_points[i][1] - matched_points[i - 1][1],
        )
        if gap[0] < gap[1]:
            # +1 for smoothing (avoid div zero error)
            ratio = (matched_points[i][1] - matched_points[prev_break_point][1] + 1) / (
                matched_points[i][0] - matched_points[prev_break_point][0] + 1
            )
            # break on current matched_point, we will finally choose the longest
            # segment.
            half = reference_length_difference / 2
            if ratio < 1 - half or ratio > 1 + half:
                candidate_ranges.append((prev_break_point, i))
                prev_break_point = i

    candidate_ranges.append((prev_break_point, len(matched_points)))

    # Find the range containing the longest query sequence
    # max_item contains the matched points we choose : matched_points[max_item[0], max_item[1]]
    max_length = -1
    max_item = (0, len(matched_points))
    for c in candidate_ranges:
        current_length = matched_points[c[1] - 1][0] - matched_points[c[0]][0]
        if current_length > max_length:
            max_length = current_length
            max_item = c

    # start, end indexes of query and reference in sourced_text
    # [(query_start, query_end, target_start, target_end)]
    segments: List[Tuple[int, int, int, int]] = []

    def add_segments(
        query_start, query_end, target_start, target_end, segment_length, segments
    ):
        num_chunk = (query_end - query_start) // segment_length
        if num_chunk > 0:
            for i in range(num_chunk):
                real_target_end = (
                    target_start + segment_length
                    if target_start + segment_length < target_end
                    else target_end
                )
                segments.append(
                    (
                        query_start,
                        query_start + segment_length,
                        target_start,
                        real_target_end,
                    )
                )
                query_start += segment_length
                target_start += segment_length
        # if the remaining part is smaller than segment_length // 4, we will append
        # it to the last segment rather than creating a new segment.
        if segments and query_end - query_start < segment_length // 4:
            segments[-1] = (
                segments[-1][0],
                query_end,
                segments[-1][2],
                target_end,
            )
        else:
            segments.append((query_start, query_end, target_start, target_end))

    target_doc_id = sourced_text.doc[matched_points[max_item[0]][1]]
    target_base = sourced_text.doc_splits[target_doc_id]
    next_target_base = sourced_text.doc_splits[target_doc_id + 1]

    query_doc_id = sourced_text.doc[matched_points[max_item[0]][0]]
    query_base = sourced_text.doc_splits[query_doc_id]
    next_query_base = sourced_text.doc_splits[query_doc_id + 1]

    # indexes of query and reference in sourced_text
    # Guarantee the target index is in the same reference document
    prev_target = matched_points[max_item[0]][1] - (
        matched_points[max_item[0]][0] - query_base
    )
    prev_break_point = (
        query_base,
        prev_target if prev_target >= target_base else target_base,
    )

    # Break the chosen matched_points into smaller segment, we might need to
    # extend on both side, as the first and last chosen points are sometimes
    # not the real start point and end point of query.
    for ind in range(max_item[0], max_item[1]):
        if matched_points[ind][0] - prev_break_point[0] > segment_length:
            if ind == max_item[0]:
                continue
            else:
                query_start = prev_break_point[0]
                query_end = matched_points[ind - 1][0]
                target_start = prev_break_point[1]
                target_end = matched_points[ind - 1][1]

                ratio = (target_end - target_start) / (query_end - query_start)
                half = reference_length_difference / 2
                if ratio < 1 - half or ratio > 1 + half:
                    continue

                prev_break_point = (query_end, target_end)
                add_segments(
                    query_start,
                    query_end,
                    target_start,
                    target_end,
                    segment_length,
                    segments,
                )

    query_start, target_start = prev_break_point
    query_end = next_query_base

    target_end = target_start + (query_end - query_start)
    target_end = target_end if target_end <= next_target_base else next_target_base

    if query_end - query_start < segment_length // 4:
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
            segment_length,
            segments,
        )
    return segments


def _combine_sub_alignments(
    sourced_text: SourcedText, sub_alignments: List[Dict[str, Any]], num_queries: int
) -> List[Tuple[Tuple[int, int], List[Dict[str, Any]]]]:
    """
    Combine the segments broken by `_break_query` together to get a long query.

    Note: The segments here contain the levenshtein alignment.

    Args:
      sourced_text:
        The SourcedText containing the queries and references.
      sub_alignments:
        A list of segments, each item (returned by levenshtein_worker) contains
        the segment it belongs to and the levenshtein alignment.
      num_queries:
        The number of queries, this argument is for efficiency, if we know number
        of queries we can preallocate the list.
    Returns:
      Return a list of tuple containing ((query_start, target_start), [alignment item]).
      The `query_start` and `target_start` are indexes in sourced_text, the `alignment item`
      is a list containing {"ref": ref, "hyp": hyp, "ref_pos": ref_pos, "hyp_pos": hyp_pos,
      "hyp_time": hyp_time}, ref is the token from reference, hyp is the token from query,
      ref_pos is local index in reference document, hyp_pos is local index in query document,
      hyp_time is the timestamp of hyp.
    """
    # The container to store the returning results.
    alignments = [None] * num_queries
    prev_target_end = 0
    for sub in sub_alignments:
        if sub["alignment"][0] == -1:
            logging.warning(f"Skipping empty sub segment.")
            continue
        # for the global mode of levenshtein, there is only one alignment
        # See docs in `levenshtein_distance` for the meaning of target_align_start,
        # target_align_end and align_str
        target_align_start, target_align_end, align_str = sub["alignment"][1][0]
        query_start, query_end, target_start, target_end = sub["segment"]
        # we are doing levenshtein_distance in "global" mode
        assert target_align_start == 0, target_align_start
        assert target_end == (target_start + target_align_end + 1), (
            target_end,
            target_start + target_align_end + 1,
        )

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
        # Note: query_source could be TextSource or Transcript, only Transcript
        # has times.
        times = query_source.times if isinstance(query_source, Transcript) else None
        # The times is in byte level.
        time_stride = 1 if query_source.binary_text.dtype == np.uint8 else 4

        query_length = query_end - query_start
        query_local_index = query_start - query_base
        query_index = query_start
        target_index = target_start + target_align_start

        hyp_time = 0 if times is None else float(times[query_local_index * time_stride])
        for ali in align_str:
            query_local_index = (
                query_local_index
                if query_local_index < query_length
                else query_length - 1
            )
            hyp_time = (
                0 if times is None else float(times[query_local_index * time_stride])
            )
            target_index = target_index if target_index < target_end else target_end - 1
            ref_pos = int(sourced_text.pos[target_index])
            query_index = query_index if query_index < query_end else query_end - 1
            if ali == "I":
                aligns.append(
                    {
                        "ref": "",
                        "hyp": chr(sourced_text.binary_text[query_index]),
                        "ref_pos": ref_pos,
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
                        "ref_pos": ref_pos,
                        "hyp_pos": query_local_index,
                        "hyp_time": hyp_time,
                    }
                )
                target_index += 1
            else:
                assert ali == "C" or ali == "S"
                ref = chr(sourced_text.binary_text[target_index])
                hyp = chr(sourced_text.binary_text[query_index])
                # The following two asserts guarantee the correctness of
                # levenshtein alignment. So *DO NOT* delete it.
                if ali == "C":
                    assert ref == hyp
                else:
                    assert ref != hyp
                aligns.append(
                    {
                        "ref": ref,
                        "hyp": hyp,
                        "ref_pos": ref_pos,
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
    reference_length_difference: float = 0.1,
    thread_pool: Optional[ThreadPool] = None,
) -> List[Tuple[Tuple[int, int], List[Dict[str, Any]]]]:
    """
    Get levenshtein alignment for each query document in sourced_text.

    For each query document we will locate its corresponding segment in the reference
    document and return the levenshtein alignment between the query and its corresponding
    reference segment.

    Args:
      sourced_text:
        The SourcedText containing the queries and references, the first N (we can
        get it from close_matches) documents are queries and the others are references.
      close_matches:
        Generated from sourced_text with the function `find_candidate_matches`.
        It contains the close matched reference token for each query token.
      segment_length:
        The query length might be long, in order to accelerate the levenshtein
        algorithm we will split the long query into smaller segments, `segment_length`
        is the expected length of the smaller segments.
      reference_length_difference:
        Because of the insertion or deletion errors, the reference sequence might be
        shorter or longer than the query, so the reference segment length can be from
        `len(query) * (1 - reference_length_difference / 2)` to
        `len(query) * (1 + reference_length_difference / 2)`.
      thread_pool:
        The ThreadPool to run levenshtein.
    Returns:
      Return a list of tuple containing ((query_start, target_start), [alignment item]).
      The `query_start` and `target_start` are indexes in sourced_text, the `alignment item`
      is a list containing {"ref": ref, "hyp": hyp, "ref_pos": ref_pos, "hyp_pos": hyp_pos,
      "hyp_time": hyp_time}, ref is the token from reference, hyp is the token from query,
      ref_pos is local index in reference document, hyp_pos is local index in query document,
      hyp_time is the timestamp of hyp.
    """
    tot_query_symbols, num_close_matches = close_matches.shape
    num_queries = sourced_text.doc[tot_query_symbols]

    row_splits = sourced_text.doc_splits

    logging.info("Getting matching points.")
    arguments = []
    for q in range(num_queries):
        query_start = row_splits[q]
        query_end = row_splits[q + 1]
        query_len = row_splits[q + 1] - row_splits[q]

        # seq1 contains the query index in sourced_text.
        # seq2 contains the reference index in sourced_text.
        seq1 = np.arange(query_start, query_end).reshape(-1, 1)
        seq1 = np.tile(seq1, num_close_matches).flatten()

        seq2 = close_matches[query_start:query_end, :].flatten()
        # matched_points is a list of (query index, target index), global index in sourced_text
        matched_points = get_longest_increasing_pairs(seq1, seq2)

        if len(matched_points) == 0:
            continue

        # In the algorithm of `find_close_matches`, `sourced_text.binary_text.size - 1`
        # means no close_matches
        trim_pos = len(matched_points) - 1
        while matched_points[trim_pos][1] == sourced_text.binary_text.size - 1:
            trim_pos -= 1
        matched_points = matched_points[0:trim_pos]

        # The following code guarantees the matched points are in the same reference
        # document. We will choose the reference document that matches the most number
        # of query tokens.
        doc_ids = sourced_text.doc[np.array([x[1] for x in matched_points])]
        # we don't really care about the real doc id, we just want to know the
        # index in matched_points, minus doc_ids[0] here for efficiency.
        doc_ids = doc_ids - doc_ids[0]
        doc_splits = row_ids_to_row_splits(doc_ids)
        max_num_matches = -1
        max_ranges = (0, len(matched_points))
        for i in range(doc_splits.size - 1):
            if doc_splits[i + 1] - doc_splits[i] < 2:
                continue
            num_matches = (
                matched_points[doc_splits[i + 1] - 1][0]
                - matched_points[doc_splits[i]][0]
            )
            if num_matches > max_num_matches:
                max_num_matches = num_matches
                max_ranges = (doc_splits[i], doc_splits[i + 1])

        if max_num_matches < 0.33 * query_len:
            logging.warning(
                f"Skipping query {q}, less than 1 / 3 of query matched by close_matches."
            )
            continue

        matched_points = matched_points[max_ranges[0] : max_ranges[1]]

        # segments is a list of (query_start, query_end, target_start, target_end)
        # they are the indexes in sourced_text.
        segments = _break_query(
            sourced_text=sourced_text,
            matched_points=matched_points,
            segment_length=segment_length,
            reference_length_difference=reference_length_difference,
        )

        # prepare arguments list for levenshtein_worker
        for seg in segments:
            arguments.append((sourced_text, seg))

    logging.info("Getting matching points done.")

    def levenshtein_worker(sourced_text, segment):
        query = sourced_text.binary_text[segment[0] : segment[1]]
        target = sourced_text.binary_text[segment[2] : segment[3]]
        # Using global matching mode here, thanks to `get_longest_increasing_pairs`
        # we have very good matching points.
        alignment = levenshtein_distance(query=query, target=target, mode="global")
        return {
            "segment": segment,
            "alignment": alignment,
            "target": target,
        }

    pool = ThreadPool() if thread_pool is None else thread_pool
    logging.info("Matching with levenshtein.")
    async_results = pool.starmap_async(levenshtein_worker, arguments)
    results = async_results.get()
    logging.info("Matching with levenshtein done.")

    return _combine_sub_alignments(sourced_text, results, num_queries)


def _get_segment_candidates(
    target_source: TextSource, alignment
) -> List[Tuple[int, int, float]]:
    """
    Split the long aligned query into smaller segments.

    we create scores for each position in the alignment, corresponding to how good
    a position it is to begin or end a split.

     - begin a split (i.e. this is first position in a segment)
        - plus score equal to num silence seconds this
          follows, up to some limit 3 seconds, i.e. this element
          of the Transcript's time minus the previous element's time; or some default (3.0 sec)
          if this is the first element of the Transcript.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if the previous non-whitespace character was a punctuation
          character.  (Lists of whitespace and punctuation characters will
          be passed in)

     - end a split (i.e. this is the last position in a segment).
        - plus score equal to number of silence seconds follows this position, up to some
          limit 3 seconds, i.e next element of the Transcript's time minus this element's
          time; or some default (3.0 sec) if this is the last element of the Transcript.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if the following non-whitespace character is a punctuation character.


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
      alignment:
        Alignment information, one item of the returned alignments from `get_alignments`.

    Returns:
      Returns a list of tuple, each tuple contains the start position, end position and score of
      current segment, start position and end position are indexes in aligns.
    """
    (query_start, target_start), aligns = alignment

    # [(index, score)]
    begin_scores: List[Tuple[int, float]] = []
    end_scores: List[Tuple[int, float]] = []

    # levenshtein errors in given region size
    half_region_size: int = 20

    errors_in_region: int = sum(
        [
            1 if align["ref"] != align["hyp"] else 0
            for align in aligns[0:half_region_size]
        ]
    )

    # will be used as max_silence, punctuation_score, init_duration_score
    # we need to make these scores similar to avoiding one of them dominating
    # the total score. Any value is ok, choose 3 here mainly based on the silence length.
    base_score = 3
    max_silence = base_score  # seconds
    punctuation_score = (
        base_score  # score for preceding (begin) and following(end) punctuation
    )

    # Use cumsum to get number of matches and errors in a range efficiently
    cumsum_match = [0] * len(aligns)
    cumsum_error = [0] * len(aligns)

    for i, align in enumerate(aligns):
        matched = align["ref"] == align["hyp"]
        cumsum_match[i] = (
            int(matched) if i == 0 else (cumsum_match[i - 1] + int(matched))
        )
        cumsum_error[i] = (
            int(not matched) if i == 0 else (cumsum_error[i - 1] + int(not matched))
        )

        # calculate the errors in a region with a sliding window.
        if i - half_region_size >= 0:
            if (
                aligns[i - half_region_size]["ref"]
                != aligns[i - half_region_size]["hyp"]
            ):
                errors_in_region -= 1
        if i + half_region_size < len(aligns):
            if (
                aligns[i + half_region_size]["ref"]
                != aligns[i + half_region_size]["hyp"]
            ):
                errors_in_region += 1

        # silence
        prev_silence = (
            max_silence if i == 0 else (align["hyp_time"] - aligns[i - 1]["hyp_time"])
        )
        prev_silence = max_silence if prev_silence > max_silence else prev_silence
        post_silence = (
            max_silence
            if i == len(aligns) - 1
            else (aligns[i + 1]["hyp_time"] - align["hyp_time"])
        )
        post_silence = max_silence if post_silence > max_silence else post_silence

        # punctuation
        prev_punctuation = 0
        j = align["ref_pos"] - 1
        while j >= 0:
            current_token = chr(target_source.binary_text[j])
            if is_punctuation(current_token, eos_only=True):
                prev_punctuation = punctuation_score
                break
            elif current_token == " " or is_punctuation(current_token):
                j -= 1
            else:
                break

        post_punctuation = 0
        j = align["ref_pos"] + 1
        while j < target_source.binary_text.size:
            current_token = chr(target_source.binary_text[j])
            if is_punctuation(current_token, eos_only=True):
                post_punctuation = punctuation_score
                break
            elif current_token == " " or is_punctuation(current_token):
                j += 1
            else:
                break
        begin_scores.append(
            (
                i,
                prev_silence + prev_punctuation - errors_in_region / half_region_size,
            )
        )
        end_scores.append(
            (
                i,
                post_silence + post_punctuation - errors_in_region / half_region_size,
            )
        )

    sorted_begin_scores = sorted(begin_scores, key=lambda x: x[1], reverse=True)
    sorted_end_scores = sorted(end_scores, key=lambda x: x[1], reverse=True)

    top_ratio = 0.1
    top_begin = sorted_begin_scores[0 : int(len(begin_scores) * top_ratio)]
    top_end = sorted_end_scores[0 : int(len(end_scores) * top_ratio)]

    # (start, end, score)
    begin_list: List[Tuple[int, int, float]] = []
    end_list: List[Tuple[int, int, float]] = []

    num_of_best_position = 4
    max_errors_ratio = 0.20
    max_text_length = 1000
    init_duration_score = (
        base_score  # duration_score for segment between 5 ~ 20 seconds
    )
    max_duration = 30  # seconds
    min_duration = 2  # seconds
    expected_duration = (5, 20)  # seconds

    for item in top_begin:
        # Caution: Can only be modified with heappush and heappop, it is used as
        # the container of a heap.
        item_q = []
        ind = item[0] + 1
        while ind < len(end_scores) and end_scores[ind][0] - item[0] < max_text_length:
            point_score = begin_scores[item[0]][1] + end_scores[ind][1]
            # matching scores
            matched_score = (
                base_score
                * (cumsum_match[end_scores[ind][0]] - cumsum_match[item[0]])
                / (end_scores[ind][0] - item[0])
            )

            # errors penalties
            total_errors = cumsum_error[end_scores[ind][0]] - cumsum_error[item[0]]
            # skipping segment with too much matching errors.
            if total_errors >= (end_scores[ind][0] - item[0]) * max_errors_ratio:
                ind += 1
                continue
            error_score = base_score * (total_errors) / (end_scores[ind][0] - item[0])

            # duration scores
            duration = (
                aligns[end_scores[ind][0]]["hyp_time"] - aligns[item[0]]["hyp_time"]
            )

            duration_score = (
                float("-inf")
                if duration >= max_duration or duration <= min_duration
                else init_duration_score
            )

            duration_score = (
                duration_score
                - (duration - min_duration)
                / (expected_duration[0] - min_duration)
                * init_duration_score
                if duration < expected_duration[0] and duration > min_duration
                else duration_score
            )

            duration_score = (
                duration_score
                - (max_duration - duration)
                / (max_duration - expected_duration[1])
                * init_duration_score
                if duration > expected_duration[1] and duration < max_duration
                else duration_score
            )

            heappush(
                item_q,
                (
                    point_score + matched_score - error_score + duration_score,
                    (
                        item[0],
                        end_scores[ind][0],
                    ),
                ),
            )
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
        while ind >= 0 and item[0] - begin_scores[ind][0] < max_text_length:
            point_score = begin_scores[ind][1] + end_scores[item[0]][1]
            # matching scores
            matched_score = (
                base_score
                * (cumsum_match[item[0]] - cumsum_match[begin_scores[ind][0]])
                / (item[0] - begin_scores[ind][0])
            )

            # errors penalties
            total_errors = cumsum_error[item[0]] - cumsum_error[begin_scores[ind][0]]
            # skipping segment with too much matching errors.
            if total_errors >= (item[0] - begin_scores[ind][0]) * max_errors_ratio:
                ind -= 1
                continue
            error_score = base_score * (total_errors) / (item[0] - begin_scores[ind][0])

            # duration scores
            duration = (
                aligns[item[0]]["hyp_time"] - aligns[begin_scores[ind][0]]["hyp_time"]
            )

            duration_score = (
                float("-inf")
                if duration >= max_duration or duration <= min_duration
                else init_duration_score
            )

            duration_score = (
                duration_score
                - (duration - min_duration)
                / (expected_duration[0] - min_duration)
                * init_duration_score
                if duration < expected_duration[0] and duration > min_duration
                else duration_score
            )

            duration_score = (
                duration_score
                - (max_duration - duration)
                / (max_duration - expected_duration[1])
                * init_duration_score
                if duration > expected_duration[1] and duration < max_duration
                else duration_score
            )

            heappush(
                item_q,
                (
                    point_score + matched_score - error_score + duration_score,
                    (
                        begin_scores[ind][0],
                        item[0],
                    ),
                ),
            )
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
    query_source: Union[Transcript, TextSource], target_source: TextSource, alignment
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Split a long aligned query into smaller segments.

    We will create scores for each position in the alignment, corresponding to how good
    a position it is to begin or end a split. We can then create a rule to assign scores
    to potential segments. The scores would consist of the begin-scores, plus the end-scores,
    plus some kind of scores of given segment (duration, matching errors .etc).

    Args:
      query_source:
        An instance of Transcript or TextSource containing the query.
      target_source:
        An instance of TextSource containing the matched reference.
      alignment:
        The alignment, an item in the list returned by `get_alignments`.
    """
    (query_start, target_start), aligns = alignment

    # candidates : (start, end, score), start and end are indexes in aligns
    candidates = _get_segment_candidates(
        target_source=target_source, alignment=alignment
    )

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
    min_text_length = 30  # around 5 words
    preceding_context_length = 1000

    for seg in segments:
        begin_pos = aligns[seg[0]]["ref_pos"]
        end_pos = aligns[seg[1]]["ref_pos"] + 1

        preceding_index = seg[0] if seg[0] == 0 else seg[0] - 1

        # start = (aligns[preceding_index]["hyp_time"] + aligns[seg[0]]["hyp_time"]) / 2

        start = aligns[seg[0]]["hyp_time"]

        following_index = seg[1] if seg[1] == len(aligns) - 1 else seg[1] + 1
        duration = (
            aligns[following_index]["hyp_time"] + aligns[seg[1]]["hyp_time"]
        ) / 2 - start

        # duration = aligns[seg[1]]["hyp_time"] - start

        hyp_begin_pos = aligns[seg[0]]["hyp_pos"]
        hyp_end_pos = aligns[seg[1]]["hyp_pos"] + 1
        hyp = "".join(
            [chr(i) for i in query_source.binary_text[hyp_begin_pos:hyp_end_pos]]
        )

        # skipping segment that has too short text
        if len(hyp) < min_text_length:
            continue

        # output one more token for reference to include the possible punctuation.
        # end_pos plus 1 here is safe, it is probably a space or punctuation.
        ref = "".join(
            [chr(i) for i in target_source.binary_text[begin_pos : end_pos + 1]]
        )

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
                "duration": math.floor(1000 * duration) / 1000,
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
       close_matches: an np.ndarray of shape (tot_query_symbols, num_close_matches)
          as returned by find_close_matches(), indicating two close matches within
          the reference text for each symbol in the query documents.
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
