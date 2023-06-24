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

import _textsearch
import numpy as np


def _renumbering(array: np.ndarray) -> np.ndarray:
    """Renumber element in the input array such that the returned array
    contains entries ranging from 0 to M - 1, where M equals
    to number of unique entries in the input array.

    The order of entries in the output array is the same as the order
    of entries in the input array. That is, if array[i] < array[j], then
    ans[i] < ans[j].

    Args:
      array:
        A 1-D array.
    Returns:
      Return a renumbered 1-D array.
    """
    uniqued, inverse = np.unique(array, return_inverse=True)
    # Note: uniqued[inverse] == array

    indexes_sorted2unsorted = np.argsort(uniqued)
    indexes_unsorted2sorted = np.empty((uniqued.size), dtype=np.int32)
    indexes_unsorted2sorted[indexes_sorted2unsorted] = np.arange(uniqued.size)

    return indexes_unsorted2sorted[inverse]


def create_suffix_array(array: np.ndarray) -> np.ndarray:
    """Create a suffix array from a 1-D input array.

    hint:
      Please refer to https://en.wikipedia.org/wiki/Suffix_array
      for what suffix array is. Different from the above Wikipedia
      article the special sentinel letter ``$`` in `textsearch`_
      is known as EOS and it is larger than any other characters.

    Args:
      array:
        A 1-D integer (or unsigned integer) array of shape ``(seq_len - 1,)``.

        Note:
          Inside this function, we will append explicitly an EOS
          symbol that is larger than ``array.max()``.
    Returns:
      Returns a suffix array of type ``np.int32``, of shape ``(seq_len,)``.
      This will consist of some permutation of the elements
      ``0 .. seq_len - 1``.

    **Usage examples**:

        .. literalinclude:: code/suffix-array.py
    """
    assert array.ndim == 1, array.ndim

    # Renumber elements in the array so that array.max() equals
    # to the number of unique elements in the array.
    #
    # In the implementation, we allocate an array of size array.max().
    # A smaller value of array.max() leads to less memory allocation.
    array = _renumbering(array)

    max_symbol = array.max()
    assert max_symbol < np.iinfo(array.dtype).max - 1, max_symbol
    eos = max_symbol + 1
    padding = np.array([eos, 0, 0, 0], dtype=array.dtype)

    padded_array = np.concatenate([array, padding])

    # The C++ code requires the input array to be contiguous.
    array_int32 = np.ascontiguousarray(padded_array, dtype=np.int32)
    return _textsearch.create_suffix_array(array_int32)


def find_close_matches(
    suffix_array: np.ndarray, query_len: int, num_close_matches: int = 1
) -> np.ndarray:
    """Assuming the suffix array was created from a text where the first
    ``query_len`` positions represent the query text and the remaining positions
    represent the reference text, return a list indicating, for each suffix
    position in the query text, the two suffix positions in the reference text
    that immediately precede and follow it lexicographically.  (I think suffix
    position refers to the last character of a suffix).

    This is easy to do from the suffix array without computing, for example,
    the LCP array; and it produces exactly 2 matches per position in the query
    text, which is also convenient.

    (Note: the query and reference texts could each represent multiple separate
    sequences, but that is handled by other code; class SourcedText keeps track
    of that information.)

    Args:
     suffix_array:
       A suffix array as created by :func:`create_suffix_array`, of dtype
       ``np.int32`` and shape ``(seq_len,)``.

     query_len:
       A number ``0 <= query_len < seq_len``, indicating the length in symbols
       (likely bytes) of the query part of the text that was used to create
       ``suffix_array``.

    Returns:
      Return an np.ndarray of shape ``(query_len * 2,)``, of the same dtype as
      ``suffix_array``, in which positions ``2*i`` and ``2*i + 1`` represent
      the two positions in the original text that are within the reference
      portion, and which immediately precede and follow, in the suffix array,
      query position ``i``.  This means that the suffixes ending at those
      positions are reverse-lexicographically close to the suffix ending at
      position ``i``.  As a special case, if one of these returned numbers would
      equal the EOS position (position seq_len - 1), or if a query position is
      before any reference position in the suffix aray, we output
      ``seq_len - 2`` instead to avoid having to handle special cases later on
      (anyway, these would not represent a close match).

    .. hint::

        Please refer to :ref:`find_close_matches_tutorial` for usages.
    """
    assert query_len >= 0, query_len
    assert suffix_array.ndim == 1, suffix_array.ndim
    assert suffix_array.dtype == np.int32, suffix_array.dtype
    seq_len = suffix_array.size
    assert query_len < seq_len, (query_len, seq_len)

    assert num_close_matches >= 1, num_close_matches

    output = np.full(
        (query_len, num_close_matches * 2),
        fill_value=seq_len - 2,
        dtype=suffix_array.dtype,
    )

    prev_refs = [seq_len - 2] * num_close_matches

    # unfinished_q contains query positions that have not been processed
    # or have not completed
    unfinished_q = {}

    refs_index = 0
    for i in range(seq_len - 1):
        text_pos = suffix_array[i]
        if text_pos >= query_len:
            prev_refs[refs_index % num_close_matches] = text_pos
            refs_index += 1
            for k in list(unfinished_q):
                output[k, unfinished_q[k]] = text_pos
                if unfinished_q[k] == num_close_matches * 2 - 1:
                    del unfinished_q[k]
                else:
                    unfinished_q[k] += 1
        else:
            for i in range(num_close_matches):
                output[text_pos, i] = prev_refs[i]

            unfinished_q[text_pos] = num_close_matches

    return output
