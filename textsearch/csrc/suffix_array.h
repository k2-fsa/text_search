/**
 * Copyright      2021 - 2023     Xiaomi Corporation (authors: Daniel Povey
 *                                                             Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TEXTSEARCH_CSRC_SUFFIX_ARRAY_H_
#define TEXTSEARCH_CSRC_SUFFIX_ARRAY_H_

namespace fasttextsearch {
/*
  This function creates a suffix array; it is based on the
  code in https://algo2.iti.kit.edu/documents/jacm05-revised.pdf,
  "Linear Work Suffix Array construction" by J. Karkkainen.

  Template args: T should be a signed integer type, such as int8, int16, int32

    @param [in] text_array  Pointer to the input array of symbols,
           including the termination symbol ($) which must be larger
           than the other symbols.
           The suffixes of this array are to be sorted.  Logically this
           array has length `seq_len`, and symbols are required
           to be in the range [1..max_symbol].
           text_array[seq_len-1] == id of the termination symbol
           text_array is additionally required to be terminated by 3 zeros,
           for purposes of this algorithm, i.e.
           text_array[seq_len] == text_array[seq_len+1]
           == text_array[seq_len+2] == 0
    @param [in] seq_len  Length of the symbol sequence (`text_array`
            must be longer than this by at least 3, for termination.)
            Require seq_len >= 0
    @param [in] max_symbol  A number that must be >= the largest
             number that might be in `text_array`, including the
             termination symbol.  The work done
             is O(seq_len + max_symbol), so it is not advisable
             to let max_symbol be too large.
    @param [out] suffix_array   A pre-allocated array of length
             `seq_len`.  At exit it will contain a permutation of
             the list [ 0, 1, ... seq_len  - 1 ], interpreted
             as the start indexes of the nonempty suffixes of `text_array`,
             with the property that the sub-arrays of `text_array`
             starting at these positions are lexicographically sorted in
             descending order.
             For example, as a trivial case, if seq_len = 3
             and text_array contains [ 3, 2, 1, 10, 0, 0, 0 ], then
             `suffix_array` would contain [ 2, 1, 0, 3 ] at exit.
    Caution: this function allocates memory internally (although
    not much more than `text_array` itself).
 */
template <typename T>
void CreateSuffixArray(const T *text_array, T seq_len, T max_symbol,
                       T *suffix_array);

} // namespace fasttextsearch
#endif // TEXTSEARCH_CSRC_SUFFIX_ARRAY_H_
