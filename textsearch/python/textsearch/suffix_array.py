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
