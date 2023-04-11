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

import _fasttextsearch
import numpy as np


def create_suffix_array(input: np.ndarray) -> np.ndarray:
    """
    Creates a suffix array from the input text and returns it as a NumPy array.  Read
    the usage carefully as it has some special requirements that will require careful data
    preparation.

    Args:
       input: an integer (or unsigned integer) type of np.ndarray.  Its shape
          should be (seq_len + 3,) where `seq_len` is the text sequence length INCLUDING
          EOS SYMBOL.
          The EOS (end of sequence) symbol must be the second largest element of the
          type (i.e. of the form 2^n - 2), must be located at input[seq_len - 1] and
          must appear nowhere else in `input` (you may have to map the input
          symbols somehow to achieve this).  It must be followed by 3 zeros, for reasons
          related to how the algorithm works.
    Returns:
          Returns a suffix array of type np.int64,
          of shape (seq_len,).  This will consist of some permutation of the elements
          0 .. seq_len - 1.
    """
    assert input.ndim == 1, input.ndim
    seq_len = input.size - 3
    assert seq_len >= 1, seq_len
    max_symbol = input[seq_len - 1]
    assert max_symbol > np.max(input[0: seq_len - 1])
    assert bool(input[seq_len:].any()) is False, input[seq_len:]

    # The C++ code requires the input array to be contiguous.
    input64 = np.ascontiguousarray(input, dtype=np.int64)
    return _fasttextsearch.create_suffix_array(input64)


def find_close_matches(
    suffix_array: np.ndarray, query_len: int, num_close_matches: int = 1
) -> np.ndarray:
    """
    Assuming the suffix array was created from a text where the first `query_len`
    positions represent the query text and the remaining positions represent
    the reference text, return a list indicating, for each suffix position in the query
    text, the two suffix positions in the reference text that immediately precede and
    follow it lexicographically.  (I think suffix position refers to the last character
    of a suffix).     This is easy to do from the suffix array without computing,
    for example, the LCP array; and it produces exactly 2 matches per position in the
    query text, which is also convenient.

    (Note: the query and reference texts could each represent multiple separate
    sequences, but that is handled by other code; class SourcedText keeps track of that
    information.)

    Args:
     suffix_array: A suffix array as created by create_suffix_array(), of dtype
        np.int64 and shape (seq_len,).

      query_len: A number 0 <= query_len < seq_len, indicating the length in symbols
       (likely bytes) of the query part of the text that was used to create `suffix_array`.

    Returns an np.ndarray of shape (query_len * 2,), of the same dtype as suffix_array,
      in which positions 2*i and 2*i + 1 represent the two positions in the original
      text that are within the reference portion, and which immediately follow and
      precede, in the suffix array, query position i.  This means that the
      suffixes ending at those positions are reverse-lexicographically close
      to the suffix ending at position i.  As a special case, if one of these
      returned numbers would equal the EOS position (position seq_len - 1), or
      if a query position is before any reference position in the suffix aray, we
      output seq_len - 2 instead to avoid having to handle special cases later on
      (anyway, these would not represent a close match).
    """
    assert query_len >= 0, query_len
    assert suffix_array.ndim == 1, suffix_array.ndim
    assert suffix_array.dtype == np.int64, suffix_array.dtype
    seq_len = suffix_array.size
    assert query_len < seq_len, (query_len, seq_len)

    output = np.ones((query_len, num_close_matches * 2), dtype=suffix_array.dtype)

    output *= seq_len - 2

    prev_refs = [seq_len - 2] * num_close_matches
    unfinished_q = {}

    refs_index = 0
    for i in range(seq_len - 1):
        text_pos = suffix_array[i]
        if text_pos >= query_len:
            prev_refs[refs_index % num_close_matches] = text_pos
            refs_index += 1
            for k in list(unfinished_q):
                output[k][unfinished_q[k]] = text_pos
                if unfinished_q[k] == num_close_matches * 2 - 1:
                    del unfinished_q[k]
                else:
                    unfinished_q[k] += 1
        else:
            for i in range(num_close_matches):
                output[text_pos][i] = prev_refs[i]
            unfinished_q[text_pos] = num_close_matches
    return output
