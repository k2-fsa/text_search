/**
 * Copyright      2023     Xiaomi Corporation (authors: Wei Kang)
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

#ifndef TEXTSEARCH_CSRC_MATCH_H_
#define TEXTSEARCH_CSRC_MATCH_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace fasttextsearch {

namespace internal {
struct GoodMatch {
  int32_t i;      // position of "good match" in 1st sequence
  int32_t j;      // position of "good match" in 2nd sequence
  int32_t prev_n; // previous n in backtrace, index into list of GoodMatch. set
                  // in the algorithm.

  GoodMatch() = default;
  GoodMatch(int32_t i, int32_t j) : i(i), j(j) {}

  GoodMatch &operator=(const GoodMatch &src) = default;
  GoodMatch(const GoodMatch &src) = default;
  // Move constructor
  GoodMatch(GoodMatch &&src) = default;
  GoodMatch &operator=(GoodMatch &&src) = default;

  bool operator<(const GoodMatch &other) const { // sort on i, then j.
    if (i < other.i)
      return true;
    else if (i > other.i)
      return false;
    else
      return j < other.j;
    // note: could perhaps do this more efficiently using << and |, not sure if
    // it matters.
  }
};

struct GoodMatchCount {
  int32_t j;         // j in GoodMatch
  int32_t n;         // index in sorted_pair
  mutable int32_t c; // good match count

  GoodMatchCount() = default;
  GoodMatchCount(int32_t j, int32_t n, int32_t c) : j(j), n(n), c(c) {}

  GoodMatchCount &operator=(const GoodMatchCount &src) = default;
  GoodMatchCount(const GoodMatchCount &src) = default;
  // Move constructor
  GoodMatchCount(GoodMatchCount &&src) = default;
  GoodMatchCount &operator=(GoodMatchCount &&src) = default;

  bool operator<(const GoodMatchCount &other) const {
    if (j < other.j)
      return true;
    else if (j > other.j)
      return false;
    else
      return n < other.n;
  }
};
} // namespace internal

/*
 * Read https://github.com/danpovey/text_search/issues/21 for more contexts of
 * this function.
 *
 * Suppose we have two sequences [i1, i2, i3... iN] and [j1, j2, j3... jN], we
 * want to find the longest chain of pairs: (i1, j1), (i2,j2), ... (iN, jN)
 * such that i1 <= i2 <= ... <= iN, and j1 <= j2 <= ... <= jN.
 *
 * We could do this is as follows:
 * First sort the list of pairs on (i, then j).
 * Then, for each pair p == (i, j) in the sorted list, set
 * count[p] = 1 + max_q(count[q])
 * where q is taken from all previously processed pairs, but limited to those
 * where q.i <= p.i and q.j <= p.j. We can take the "max" expression to be 0
 * if there is no q satisfying these constraints.
 *
 * Then when we are done processing all pairs, the last element in the
 * longest-chain is the pair with the largest count. We can store
 * "back-pointers" for each processed pair to tell us which previous pair
 * satisfied the "max" expression.
 *
 * The above algorithm is quadratic on the face of it, but by making use of the
 * properties of std::set we can make n log(n).
 *
 * @param [in] seq1  The first sequence.
 * @param [in] seq2  The second sequence.
 * @param [in] size  The length of the sequences.
 * @param [in, out] best_trace  The container that backtrace will write to. It
 *                    will contain the longest chain of pairs: (i1, j1),
 *                    (i2,j2), ... (iN, jN) such that i1 <= i2 <= ... <= iN,
 *                    and j1 <= j2 <= ... <= jN when returned.
 */
void GetLongestIncreasingPairs(
    const int32_t *seq1, const int32_t *seq2, int32_t size,
    std::vector<std::pair<int32_t, int32_t>> *best_trace);

/*
 * The same as `GetLongestIncreasingPairs`, the difference is
 * `GetLongestIncreasingPairsSimple` was implemented in a simpler way by using
 * two loops (quadratic time complexity).
 *
 * Caution: Don't use this function, it is slow, we implement it for testing
 * purpose.
 */
void GetLongestIncreasingPairsSimple(
    const int32_t *seq1, const int32_t *seq2, int32_t size,
    std::vector<std::pair<int32_t, int32_t>> *best_trace);
} // namespace fasttextsearch

#endif // TEXTSEARCH_CSRC_MATCH_H_
