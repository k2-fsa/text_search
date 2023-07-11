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
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from heapq import heappush, heappop
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import regex

from _textsearch import (
    find_close_matches,
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
    Break long query into small segments at the matched points with the
    `segment_length` constraint.

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
    candidate_ranges: List[
        Tuple[int, int]
    ] = []  # start, end indexes in matched_points
    for i in range(1, len(matched_points)):
        # The query and reference distance between current matched_point and
        # preceding matched_point.
        gap = (
            matched_points[i][0] - matched_points[i - 1][0],
            matched_points[i][1] - matched_points[i - 1][1],
        )
        if gap[0] < gap[1]:
            # +1 for smoothing (avoid div zero error)
            ratio = (
                matched_points[i][1] - matched_points[prev_break_point][1] + 1
            ) / (matched_points[i][0] - matched_points[prev_break_point][0] + 1)
            # break on current matched_point, we will finally choose the longest
            # segment.
            half = reference_length_difference / 2
            if ratio < 1 - half or ratio > 1 + half:
                candidate_ranges.append((prev_break_point, i))
                prev_break_point = i

    candidate_ranges.append((prev_break_point, len(matched_points)))

    # Find the range containing the longest query sequence
    # max_item contains the matched points we choose :
    # matched_points[max_item[0], max_item[1]]
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
        query_start,
        query_end,
        target_start,
        target_end,
        segment_length,
        segments,
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
        # if the remaining part is smaller than segment_length // 4, we will
        # append it to the last segment rather than creating a new segment.
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
    target_end = (
        target_end if target_end <= next_target_base else next_target_base
    )

    if query_end - query_start < segment_length // 4:
        if segments:
            segments[-1] = (
                segments[-1][0],
                query_end,
                segments[-1][2],
                target_end,
            )
        else:
            segments.append((query_start, query_end, target_start, target_end,))
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
    sourced_text: SourcedText,
    sub_alignments: List[Dict[str, Any]],
    num_queries: int,
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
      Return a list of tuple containing
      ((query_start, target_start), [alignment item]).
      The `query_start` and `target_start` are indexes in sourced_text,
      the `alignment item` is a list containing `{"ref": ref, "hyp": hyp,
      "ref_pos": ref_pos, "hyp_pos": hyp_pos, "hyp_time": hyp_time}`, `ref` is
      the token from reference, `hyp` is the token from query, `ref_pos` is
      local index in reference document, `hyp_pos` is local index in query
      document, `hyp_time` is the timestamp of `hyp`.
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

        query_doc_id = sourced_text.doc[query_start]

        if alignments[query_doc_id] is None:
            alignments[query_doc_id] = (
                (query_start, target_start + target_align_start),
                [],
            )

        query_base = sourced_text.doc_splits[query_doc_id]
        query_source = sourced_text.sources[query_doc_id]

        # aligns : [{"ref": , "hyp": , "ref_pos": , "hyp_pos": , "hyp_time":}]
        aligns = alignments[query_doc_id][1]
        # Note: query_source could be TextSource or Transcript, only Transcript
        # has times.
        times = (
            query_source.times if isinstance(query_source, Transcript) else None
        )
        # The times is in byte level.
        time_stride = 1 if query_source.binary_text.dtype == np.uint8 else 4

        query_base_next = sourced_text.doc_splits[query_doc_id + 1]
        query_length = query_base_next - query_base
        query_index = query_start
        target_index = target_start + target_align_start

        for ali in align_str:
            query_index = (
                query_index if query_index < query_end else query_end - 1
            )
            hyp_pos = int(sourced_text.pos[query_index])

            hyp_time = (
                0 if times is None else float(times[hyp_pos * time_stride])
            )

            target_index = (
                target_index if target_index < target_end else target_end - 1
            )
            ref_pos = int(sourced_text.pos[target_index])
            if ali == "I":
                ref = ""
                hyp = chr(sourced_text.binary_text[query_index])
                query_index += 1
            elif ali == "D":
                ref = chr(sourced_text.binary_text[target_index])
                hyp = ""
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
                query_index += 1
                target_index += 1
            aligns.append(
                {
                    "ref": ref,
                    "hyp": hyp,
                    "ref_pos": ref_pos,
                    "hyp_pos": hyp_pos,
                    "hyp_time": hyp_time,
                }
            )
    return alignments


def align_queries(
    sourced_text: SourcedText,
    num_query_tokens: int,
    num_close_matches: int = 2,
    segment_length: int = 5000,
    reference_length_difference: float = 0.1,
    min_matched_query_ratio: float = 0.33,
    thread_pool: Optional[ThreadPool] = None,
) -> List[Tuple[Tuple[int, int], List[Dict[str, Any]]]]:
    """
    Get levenshtein alignment for each query document in sourced_text.

    For each query document we will locate its corresponding segment in the
    reference document and return the levenshtein alignment between the query
    and its corresponding reference segment.

    Args:
      sourced_text:
        The SourcedText containing the queries and references, the first N
        documents are queries and the others are references.
      num_query_tokens:
        The number of query tokens in sourced_text, it tells the boundary between
        queries and references sourced_text[0:num_query_tokens] are query tokens,
        sourced_text[num_query_tokens:] are reference tokens.
      num_close_matches:
        The number of close_matches for each query token.
      segment_length:
        The query length might be long, in order to accelerate the levenshtein
        algorithm we will split the long query into smaller segments, `segment_length`
        is the expected length of the smaller segments.
      reference_length_difference:
        Because of the insertion or deletion errors, the reference sequence might be
        shorter or longer than the query, so the reference segment length can be from
        `len(query) * (1 - reference_length_difference / 2)` to
        `len(query) * (1 + reference_length_difference / 2)`.
      min_matched_query_ratio:
        The minimum ratio of matched query tokens, if the ratio is less than
        `min_matched_query_ratio` the query will be dropped.
      thread_pool:
        The ThreadPool to run levenshtein.
    Returns:
      Return a list of tuple containing
      ((query_start, target_start), [alignment item]).
      The `query_start` and `target_start` are indexes in sourced_text,
      the `alignment item` is a list containing `{"ref": ref, "hyp": hyp,
      "ref_pos": ref_pos, "hyp_pos": hyp_pos, "hyp_time": hyp_time}`, `ref` is
      the token from reference, `hyp` is the token from query, `ref_pos` is
      local index in reference document, `hyp_pos` is local index in query
      document, `hyp_time` is the timestamp of `hyp`.
    """

    logging.debug(
        f"Creating suffix array on a sequence with length : "
        f"{sourced_text.binary_text.size}."
    )
    suffix_array = create_suffix_array(sourced_text.binary_text)

    logging.debug(
        f"Finding close matches for {num_query_tokens} query tokens with "
        f"num_close_matches={num_close_matches}."
    )
    close_matches = find_close_matches(
        suffix_array, num_query_tokens, num_close_matches=num_close_matches
    )

    tot_query_symbols, num_close_matches = close_matches.shape
    assert num_query_tokens == tot_query_symbols, (
        num_query_tokens,
        tot_query_symbols,
    )
    num_queries = sourced_text.doc[tot_query_symbols]

    logging.debug(f"Getting alignments for {num_queries} queries.")

    row_splits = sourced_text.doc_splits
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
        # matched_points is a list of (query index, target index), global index
        # in sourced_text
        matched_points = get_longest_increasing_pairs(seq1, seq2)

        if len(matched_points) == 0:
            continue

        # In the algorithm of `find_close_matches`,
        # `sourced_text.binary_text.size - 1` means no close_matches
        trim_pos = len(matched_points) - 1
        while matched_points[trim_pos][1] == sourced_text.binary_text.size - 1:
            trim_pos -= 1
        matched_points = matched_points[0:trim_pos]

        # The following code guarantees the matched points are in the same
        # reference document. We will choose the reference document that matches
        # the most number of query tokens.
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

        if max_num_matches < min_matched_query_ratio * query_len:
            logging.warning(
                f"Skipping query {q}, less than {min_matched_query_ratio * 100}"
                f"% of query tokens matched in close_matches."
            )
            continue

        matched_points = matched_points[max_ranges[0] : max_ranges[1]]

        # Break query into short segments, so that we only need to run levenshtein
        # algorithm on short sequence (for efficiency).
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

    def levenshtein_worker(sourced_text, segment):
        query = sourced_text.binary_text[segment[0] : segment[1]]
        target = sourced_text.binary_text[segment[2] : segment[3]]
        # Using global matching mode here, thanks to `get_longest_increasing_pairs`
        # we have very good matching points.
        alignment = levenshtein_distance(
            query=query, target=target, mode="global"
        )
        return {
            "segment": segment,
            "alignment": alignment,
            "target": target,
        }

    pool = ThreadPool() if thread_pool is None else thread_pool
    logging.debug(f"Matching with levenshtein for {len(arguments)} segments.")
    async_results = pool.starmap_async(levenshtein_worker, arguments)
    results = async_results.get()
    logging.debug("Matching with levenshtein done.")

    alignments = _combine_sub_alignments(sourced_text, results, num_queries)

    assert sourced_text.doc[num_query_tokens] == len(alignments), (
        sourced_text.doc[num_query_tokens],
        len(alignments),
    )
    return alignments


def _get_segment_candidates(
    target_source: TextSource,
    alignment,
    silence_length_to_break: float = 0.6,  # in second
    min_duration: float = 2,  # in second
    max_duration: float = 30,  # in second
    expected_duration: Tuple[float, float] = (5, 20),  # in second
    max_error_rate: float = 0.20,
    num_of_best_position: int = 4,
) -> List[Tuple[int, int, float]]:
    """
    Split the long aligned query into smaller segments.

    First, we get a list candidate breaking positions, these positions have
    long enough preceding silence (begin a split) or succeeding silence (end of
    a split), the length of silence is controlled by `silence_length_to_break`.
    If the `target_source` has punctuations, we will only start or end of a
    split at punctuation positions that indicate the end of a sentence.

    we create scores for each candidate position in the alignment, corresponding
    to how good a position it is to begin or end a split.

     - begin a split (i.e. this is first position in a segment)
        - plus score equal to num silence seconds this
          follows, up to some limit 3 seconds, i.e. this element
          of the Transcript's time minus the previous element's time; or some
          default (3.0 sec) if this is the first element of the Transcript.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if the previous non-whitespace character was a punctuation
          character.  (Lists of whitespace and punctuation characters will
          be passed in)

     - end a split (i.e. this is the last position in a segment).
        - plus score equal to number of silence seconds follows this position,
          up to some limit 3 seconds, i.e next element of the Transcript's time
          minus this element's time; or some default (3.0 sec) if this is the
          last element of the Transcript.
        - good if there are few errors around this point in the alignment, i.e.
          score corresponding to number of ins,del,mismatch within a certain
          region of this position.
        - good if the following non-whitespace character is a punctuation character.


    We then create a rule to assign scores to potential segments. This consist of
    the begin-scores, plus the end-scores, plus:
      - Some kind of penalty related to the duration of the segment, e.g.
        infinity if it's over some `max_duration` like 30 seconds or less than a
        `min_duration` like 2 seconds; else, one that encourages a duration
        between `expected_duration` like 5 to 20 seconds.
      - A bonus for the number of matches in the alignment.
      - A penalty for the number of errors in the alignment (could multiply this by
        some scale depending how much we don't want to have errors, but some errors
        are expected due to both ASR errors and normalization differences.)

    Next, we search for a good segmentation. We define the problem as getting
    the highest-scoring set of segments that do not overlap. For each
    begin_position, find the `num_of_best_position` like 4 best-scoring
    end_positions. For each end_position, find the `num_of_best_position`
    best-scoring begin_positions. Append the preceding 2 sets of segments to get
    a list of candidate segments.

    Args:
      target_source:
        A TextSource containing the matched reference.
      alignment:
        Alignment information, one item of the returned alignments from
        `align_queries`.
      silence_length_to_break:
        A threshold for deciding the possible breaking points, if a position has
        preceding or succeeding silence length greater than this value, we will
        add it as a possible breaking point.
        Caution: Only be used when there are no punctuations in target_source.
      min_duration:
        The minimum duration (in second) allowed for a segment.
      max_duration:
        The maximum duration (in second) allowed for a segment.
      expected_duration:
        The expected duration (in second) for a segment, it is a range (a tuple
        containing lower bound and upper bound).
        Note: The values must satisfy `min_duration <= expected_duration[0]` and
        `max_duration >= expected_duration[1]`.
      max_error_rate:
        The max levenshtein distance (char level) between query and target at
        the segment area, if the segments with higher error rate than this value
        will not appear in the final result list.
      num_of_best_position:
        For each candidate breaking points, there will be several possible start
        points (if the point is the end of segment) or end points (if the point
        is the start of a segment), this is the number of possible points.
        Normally this does not affect the final result too much, just leave it
        as default is OK.

    Returns:
      Returns a list of tuple, each tuple contains the start position,
      end position and score of current segment, start position and end
      position are indexes in aligns.
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
    # the total score. Any value is ok, choose 3 here mainly based on the
    # silence length.
    base_score = 3
    max_silence = base_score  # seconds
    # score for preceding (begin) and succeeding (end) punctuation
    punctuation_score = base_score

    # Use cumsum to get number of matches and errors in a range efficiently
    cumsum_match = [0] * len(aligns)
    cumsum_error = [0] * len(aligns)

    # to avoid breaking at somethings like Mr. Mrs. etc.
    period_patterns = regex.compile(
        "(?<!Mr|Mrs|Dr|Ms|Prof|Pro|Capt|Gen|Sen|Rev|Hon|St)\."
    )
    # the largest length of the patterns.
    period_pattern_length = 4

    for i, align in enumerate(aligns):
        matched = align["ref"] == align["hyp"]
        cumsum_match[i] = (
            int(matched) if i == 0 else (cumsum_match[i - 1] + int(matched))
        )
        cumsum_error[i] = (
            int(not matched)
            if i == 0
            else (cumsum_error[i - 1] + int(not matched))
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
            max_silence
            if i == 0
            else (align["hyp_time"] - aligns[i - 1]["hyp_time"])
        )
        prev_silence = (
            max_silence if prev_silence > max_silence else prev_silence
        )
        succ_silence = (
            max_silence
            if i == len(aligns) - 1
            else (aligns[i + 1]["hyp_time"] - align["hyp_time"])
        )
        succ_silence = (
            max_silence if succ_silence > max_silence else succ_silence
        )

        # punctuation
        prev_punctuation = 0
        j = align["ref_pos"] - 1
        while j >= 0:
            current_token = chr(target_source.binary_text[j])
            if is_punctuation(current_token, eos_only=True):
                tmp = "".join(
                    [
                        chr(x)
                        for x in target_source.binary_text[
                            j - period_pattern_length : j + 1
                        ]
                    ]
                )
                if period_patterns.search(tmp) is not None:
                    prev_punctuation = punctuation_score
                    break
                else:
                    j -= 1
            elif current_token == " " or is_punctuation(current_token):
                j -= 1
            else:
                break

        succ_punctuation = 0
        j = align["ref_pos"] + 1
        while j < target_source.binary_text.size:
            current_token = chr(target_source.binary_text[j])
            if is_punctuation(current_token, eos_only=True):
                tmp = "".join(
                    [
                        chr(x)
                        for x in target_source.binary_text[
                            j - period_pattern_length : j + 1
                        ]
                    ]
                )
                if period_patterns.search(tmp) is not None:
                    succ_punctuation = punctuation_score
                    break
                else:
                    j += 1
            elif current_token == " " or is_punctuation(current_token):
                j += 1
            else:
                break

        begin_score = (
            prev_silence
            + prev_punctuation
            - errors_in_region / half_region_size
        )

        end_score = (
            succ_silence
            + succ_punctuation
            - errors_in_region / half_region_size
        )

        if target_source.has_punctuation:
            if prev_punctuation > 0 or i == 0:
                begin_scores.append((i, begin_score,))
            if succ_punctuation > 0 or i == len(aligns) - 1:
                end_scores.append((i, end_score,))
        else:
            if matched and (prev_silence >= silence_length_to_break or i == 0):
                begin_scores.append((i, begin_score,))
            if matched and (
                succ_silence >= silence_length_to_break or i == len(aligns) - 1
            ):
                end_scores.append((i, end_score,))

    # (start, end, score)
    begin_list: List[Tuple[int, int, float]] = []
    end_list: List[Tuple[int, int, float]] = []

    init_duration_score = (
        base_score  # duration_score for segment between 5 ~ 20 seconds
    )

    last_ind = 0
    for i, item in enumerate(begin_scores):
        # Caution: Can only be modified with heappush and heappop, it is used as
        # the container of a heap.
        item_q = []
        ind = bisect_right(end_scores, item, lo=last_ind)
        last_ind = ind

        while True:
            if ind >= len(end_scores) or ind < 0:
                break

            duration = (
                aligns[end_scores[ind][0]]["hyp_time"]
                - aligns[item[0]]["hyp_time"]
            )
            if duration <= min_duration:
                ind += 1
                continue

            if duration > max_duration:
                break

            point_score = begin_scores[i][1] + end_scores[ind][1]

            # matching scores
            matched_score = (
                base_score
                * (cumsum_match[end_scores[ind][0]] - cumsum_match[item[0]])
                / (end_scores[ind][0] - item[0])
            )

            # errors penalties
            total_errors = (
                cumsum_error[end_scores[ind][0]] - cumsum_error[item[0]]
            )
            # skipping segment with too much matching errors.
            if total_errors >= (end_scores[ind][0] - item[0]) * max_error_rate:
                ind += 1
                continue
            error_score = (
                base_score * (total_errors) / (end_scores[ind][0] - item[0])
            )

            duration_score = init_duration_score

            duration_score = (
                duration_score
                - (duration - min_duration)
                / (expected_duration[0] - min_duration)
                * init_duration_score
                if duration < expected_duration[0]
                else duration_score
            )

            duration_score = (
                duration_score
                - (max_duration - duration)
                / (max_duration - expected_duration[1])
                * init_duration_score
                if duration > expected_duration[1]
                else duration_score
            )

            heappush(
                item_q,
                (
                    point_score + matched_score - error_score + duration_score,
                    (item[0], end_scores[ind][0],),
                ),
            )
            if len(item_q) > num_of_best_position:
                heappop(item_q)
            ind += 1
        while item_q:
            x = heappop(item_q)
            begin_list.append((x[1][0], x[1][1], x[0]))

    last_ind = 0
    for i, item in enumerate(end_scores):
        # Caution: Can only be modified with heappush and heappop, it is used as
        # the container of a heap.
        item_q = []
        ind = bisect_left(begin_scores, item, lo=last_ind)
        last_ind = ind

        while True:
            if ind < 0 or ind >= len(begin_scores):
                break

            duration = (
                aligns[item[0]]["hyp_time"]
                - aligns[begin_scores[ind][0]]["hyp_time"]
            )

            if duration <= min_duration:
                ind -= 1
                continue

            if duration >= max_duration:
                break

            point_score = begin_scores[ind][1] + end_scores[i][1]
            # matching scores
            matched_score = (
                base_score
                * (cumsum_match[item[0]] - cumsum_match[begin_scores[ind][0]])
                / (item[0] - begin_scores[ind][0])
            )

            # errors penalties
            total_errors = (
                cumsum_error[item[0]] - cumsum_error[begin_scores[ind][0]]
            )
            # skipping segment with too much matching errors.
            if (
                total_errors
                >= (item[0] - begin_scores[ind][0]) * max_error_rate
            ):
                ind -= 1
                continue
            error_score = (
                base_score * (total_errors) / (item[0] - begin_scores[ind][0])
            )

            duration_score = init_duration_score

            duration_score = (
                duration_score
                - (duration - min_duration)
                / (expected_duration[0] - min_duration)
                * init_duration_score
                if duration < expected_duration[0]
                else duration_score
            )

            duration_score = (
                duration_score
                - (max_duration - duration)
                / (max_duration - expected_duration[1])
                * init_duration_score
                if duration > expected_duration[1]
                else duration_score
            )

            heappush(
                item_q,
                (
                    point_score + matched_score - error_score + duration_score,
                    (begin_scores[ind][0], item[0],),
                ),
            )
            if len(item_q) > num_of_best_position:
                heappop(item_q)
            ind -= 1
        while item_q:
            x = heappop(item_q)
            end_list.append((x[1][0], x[1][1], x[0]))

    candidates = begin_list + end_list
    return candidates


def _split_into_segments(
    query_source: Union[Transcript, TextSource],
    target_source: TextSource,
    alignment: Tuple[Tuple[int, int], List[Dict[str, Any]]],
    preceding_context_length: int = 1000,
    timestamp_position: str = "middle",  # previous, middle, current
    silence_length_to_break: float = 0.6,  # in second
    min_duration: float = 2,  # in second
    max_duration: float = 30,  # in second
    expected_duration: Tuple[float, float] = (5, 20),  # in second
    max_error_rate: float = 0.15,
    num_of_best_position: int = 4,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Split a long aligned query into smaller segments.

    We will create scores for each position in the alignment, corresponding to
    how good a position it is to begin or end a split. We can then create a rule
    to assign scores to potential segments. The scores would consist of the
    begin-scores, plus the end-scores, plus some kind of scores of given segment
    (duration, matching errors .etc).

    Args:
      query_source:
        An instance of Transcript or TextSource containing the query.
      target_source:
        An instance of TextSource containing the matched reference.
      alignment:
        The alignment, an item in the list returned by `align_queries`.
      preceding_context_length:
        The number of characters of preceding context.
      timestamp_position:
        It indicates which token we will get `start_time` from, valid values are
        `previous`, `middle` and `current`. If it equals to `current` the
        `start_time` is the timestamp of token in `begin_pos`, if it equals to
        `previous` the `start_time` is the timestamp of token in `begin_pos - 1`
        if it equals to `middle`, the `start_time` is the averaged timestamp of
        tokens in `begin_pos` and `begin_pos - 1`.
      silence_length_to_break:
        A threshold for deciding the possible breaking points, if a position has
        preceding or succeeding silence length greater than this value, we will
        add it as a possible breaking point.
        Caution: Only be used when there are no punctuations in target_source.
      min_duration:
        The minimum duration (in second) allowed for a segment.
      max_duration:
        The maximum duration (in second) allowed for a segment.
      expected_duration:
        The expected duration (in second) for a segment, it is a range (a tuple
        containing lower bound and upper bound).
        Note: The values must satisfy `min_duration <= expected_duration[0]` and
        `max_duration >= expected_duration[1]`.
      max_error_rate:
        The max levenshtein distance (char level) between query and target at
        the segment area, if the segments with higher error rate than this value
        will not appear in the final result list.
      num_of_best_position:
        For each candidate breaking points, there will be several possible start
        points (if the point is the end of segment) or end points (if the point
        is the start of a segment), this is the number of possible points.
        Normally this does not affect the final result too much, just leave it
        as default is OK.

    Return
      Returns a list of Dict containing the details of each segment, looks like

         {
             "begin_byte": int,   # begin position in original target source
             "end_byte": int,     # end position in original target source
             "start_time": float, # start timestamp in the audio
             "duration": float,   # duration of this segment
             "hyp": str,          # text from query source
             "ref": str,          # text from target source
             "pre_ref": str,      # preceding text from target source
             "pre_hyp": str,      # preceding text from query source
             "post_ref": str,     # succeeding text from target source
             "post_hyp": str,     # succeeding text from query source
         }
    """
    (query_start, target_start), aligns = alignment

    # candidates : (start, end, score), start and end are indexes in aligns
    candidates = _get_segment_candidates(
        target_source=target_source,
        alignment=alignment,
        silence_length_to_break=silence_length_to_break,
        min_duration=min_duration,
        max_duration=max_duration,
        expected_duration=expected_duration,
        max_error_rate=max_error_rate,
        num_of_best_position=num_of_best_position,
    )

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    # Handle the overlapping
    # Caution: Don't modified selected_ranges, it will be manipulated in
    # `is_overlap` and will be always kept sorted.
    selected_ranges: List[Tuple[int, int]] = []
    segments = []
    for r in candidates:
        if not is_overlap(
            selected_ranges, query=(r[0], r[1]), overlap_ratio=0.5
        ):
            segments.append(r)

    results = []

    for seg in segments:
        begin_pos = aligns[seg[0]]["ref_pos"]
        end_pos = aligns[seg[1]]["ref_pos"] + 1

        preceding_index = seg[0] if seg[0] == 0 else seg[0] - 1
        succeeding_index = seg[1] if seg[1] == len(aligns) - 1 else seg[1] + 1

        if timestamp_position == "middle":
            start_time = (
                aligns[preceding_index]["hyp_time"] + aligns[seg[0]]["hyp_time"]
            ) / 2
            end_time = (
                aligns[succeeding_index]["hyp_time"]
                + aligns[seg[1]]["hyp_time"]
            ) / 2
        elif timestamp_position == "previous":
            start_time = aligns[preceding_index]["hyp_time"]
            end_time = aligns[seg[1]]["hyp_time"]
        else:
            assert timestamp_position == "current", (
                timestamp_position,
                "current",
            )
            start_time = aligns[seg[0]]["hyp_time"]
            end_time = aligns[succeeding_index]["hyp_time"]

        hyp_begin_pos = aligns[seg[0]]["hyp_pos"]
        hyp_end_pos = aligns[seg[1]]["hyp_pos"] + 1
        hyp = "".join(
            [
                chr(i)
                for i in query_source.binary_text[hyp_begin_pos:hyp_end_pos]
            ]
        )

        # output one more token for reference to include the possible punctuation.
        # end_pos plus 1 here is safe, it is probably a space or punctuation.
        ref = "".join(
            [chr(i) for i in target_source.binary_text[begin_pos : end_pos + 1]]
        )

        preceding_pos = (
            begin_pos - preceding_context_length
            if begin_pos - preceding_context_length >= 0
            else 0
        )
        preceding_ref = "".join(
            [chr(i) for i in target_source.binary_text[preceding_pos:begin_pos]]
        )
        succeeding_ref = "".join(
            [
                chr(i)
                for i in target_source.binary_text[
                    end_pos : end_pos + preceding_context_length
                ]
            ]
        )

        preceding_pos = (
            hyp_begin_pos - preceding_context_length
            if hyp_begin_pos - preceding_context_length > 0
            else 0
        )
        preceding_hyp = "".join(
            [
                chr(i)
                for i in query_source.binary_text[preceding_pos:hyp_begin_pos]
            ]
        )
        succeeding_hyp = "".join(
            [
                chr(i)
                for i in query_source.binary_text[
                    hyp_end_pos : hyp_begin_pos + preceding_context_length
                ]
            ]
        )

        results.append(
            {
                "begin_byte": begin_pos,
                "end_byte": end_pos,
                "start_time": start_time,
                "duration": math.floor(1000 * (end_time - start_time)) / 1000,
                "hyp": hyp,
                "ref": ref,
                "pre_ref": preceding_ref,
                "pre_hyp": preceding_hyp,
                "post_ref": succeeding_ref,
                "post_hyp": succeeding_hyp,
            }
        )
    return results


def _split_helper(
    query_source: Union[TextSource, Transcript],
    target_source: TextSource,
    cut_index: Tuple[int, int],
    alignment: Tuple[Tuple[int, int], List[Dict[str, Any]]],
    preceding_context_length: int,
    timestamp_position: str,
    silence_length_to_break: float,
    min_duration: float,
    max_duration: float,
    expected_duration: Tuple[float, float],
    max_error_rate: float,
    num_of_best_position: int,
):
    """
    A worker function for splitting aligned query.
    """
    segments = _split_into_segments(
        query_source,
        target_source,
        alignment,
        preceding_context_length=preceding_context_length,
        timestamp_position=timestamp_position,
        silence_length_to_break=silence_length_to_break,
        min_duration=min_duration,
        max_duration=max_duration,
        expected_duration=expected_duration,
        max_error_rate=max_error_rate,
        num_of_best_position=num_of_best_position,
    )
    return cut_index, segments


def split_aligned_queries(
    sourced_text: SourcedText,
    alignments: List[Tuple[Tuple[int, int], List[Dict[str, Any]]]],
    cut_indexes: List[Tuple[int, int]],
    process_pool: Optional[Pool] = None,
    preceding_context_length: int = 1000,
    timestamp_position: str = "current",  # previous, middle, current
    silence_length_to_break: float = 0.6,  # in second
    min_duration: float = 2,  # in second
    max_duration: float = 30,  # in second
    expected_duration: Tuple[float, float] = (5, 20),  # in second
    max_error_rate: float = 0.15,
    num_of_best_position: int = 4,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Split the aligned queries into smaller segments (A query might have several
    hours of audio, which is not suitable for ASR training)

    Args:
      sourced_text:
        The SourcedText containing the queries and references.
      alignments:
        The alignments returned by function align_queries. The length of it is
        equals to the number of queries.
      cut_indexes:
        A list of tuple containing the original cut index and supervision index
        of the query([(cut index, sup index)]), it satisfies
        `len(cut_indexes) == len(alignments)`
      process_pool:
        The process pool to split aligned queries. The algorithms are
        implemented in python, so we use process pool to get rid of the effect
        of GIL.
      preceding_context_length:
        The number of characters of preceding context.
      timestamp_position:
        It indicates which token we will get `start_time` from, valid values are
        `previous`, `middle` and `current`. If it equals to `current` the
        `start_time` is the timestamp of token in `begin_pos`, if it equals to
        `previous` the `start_time` is the timestamp of token in `begin_pos - 1`
        if it equals to `middle`, the `start_time` is the averaged timestamp of
        tokens in `begin_pos` and `begin_pos - 1`.
      silence_length_to_break:
        A threshold for deciding the possible breaking points, if a position has
        preceding or succeeding silence length greater than this value, we will
        add it as a possible breaking point.
        Caution: Only be used when there are no punctuations in target_source.
      min_duration:
        The minimum duration (in second) allowed for a segment.
      max_duration:
        The maximum duration (in second) allowed for a segment.
      expected_duration:
        The expected duration (in second) for a segment, it is a range (a tuple
        containing lower bound and upper bound).
        Note: The values must satisfy `min_duration <= expected_duration[0]` and
        `max_duration >= expected_duration[1]`.
      max_error_rate:
        The max levenshtein distance (char level) between query and target at
        the segment area, if the segments with higher error rate than this value
        will not appear in the final result list.
      num_of_best_position:
        For each candidate breaking points, there will be several possible start
        points (if the point is the end of segment) or end points (if the point
        is the start of a segment), this is the number of possible points.
        Normally this does not affect the final result too much, just leave it
        as default is OK.

    Return:
      Returns a list of Dict containing the details of each segment, looks like

         {
             "begin_byte": int,   # begin position in original target source
             "end_byte": int,     # end position in original target source
             "start_time": float, # start timestamp in the audio
             "duration": float,   # duration of this segment
             "hyp": str,          # text from query source
             "ref": str,          # text from target source
             "pre_ref": str,      # preceding text from target source
             "pre_hyp": str,      # preceding text from query source
             "post_ref": str,     # succeeding text from target source
             "post_hyp": str,     # succeeding text from query source
         }

    """
    arguments = []
    aligned_length = 0
    for i in range(len(alignments)):
        if alignments[i] is not None:
            (query_start, target_start), aligns = alignments[i]
            aligned_length += aligns[-1]["hyp_time"] - aligns[0]["hyp_time"]
            query_source = sourced_text.sources[sourced_text.doc[query_start]]
            target_source = sourced_text.sources[sourced_text.doc[target_start]]
            arguments.append(
                (
                    query_source,
                    target_source,
                    cut_indexes[i],
                    alignments[i],
                    preceding_context_length,
                    timestamp_position,
                    silence_length_to_break,
                    min_duration,
                    max_duration,
                    expected_duration,
                    max_error_rate,
                    num_of_best_position,
                )
            )
    logging.debug(
        f"Aligned length : {aligned_length} seconds for {len(arguments)} queries."
    )

    logging.debug(f"Splitting into segments for {len(arguments)} queries.")
    pool = Pool() if process_pool is None else process_pool
    async_results = pool.starmap_async(_split_helper, arguments)
    results = async_results.get()
    logging.debug(f"Splitting into segments done.")

    return results
