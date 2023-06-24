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

#include "textsearch/csrc/match.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <set>
#include <vector>

namespace fasttextsearch {

void GetLongestIncreasingPairs(
    const int32_t *seq1, const int32_t *seq2, int32_t size,
    std::vector<std::pair<int32_t, int32_t>> *best_trace) {

  if (size == 0)
    return;

  assert(size > 0);
  // sort this by i, then j.  prev_n values can be undefined for now.
  std::vector<internal::GoodMatch> sorted_pairs(size);
  for (int32_t i = 0; i < size; ++i) {
    sorted_pairs[i] = internal::GoodMatch(seq1[i], seq2[i]);
  }
  std::sort(sorted_pairs.begin(), sorted_pairs.end());

  // in addition to being sorted on j, the counts will always have the
  // property that the c values are strictly increasing.  We ensure this
  // by removing elements as necessary.
  std::set<internal::GoodMatchCount> cur_counts;
  cur_counts.insert(internal::GoodMatchCount(0, -1, 0));
  auto hint_iter = cur_counts.begin();

  int32_t N = sorted_pairs.size();
  for (int32_t n = 0; n < N; ++n) {
    int32_t j = sorted_pairs[n].j;
    internal::GoodMatchCount gmc(
        j, n, 0); // the 0 is a don't-care, we'll set it later.
    auto iter = cur_counts.insert(hint_iter, gmc);
    hint_iter = iter;
    auto prev_iter = iter, next_iter = iter;
    --prev_iter; // now points to the previous element, should have prev_iter->j
                 // <= j.
    // we know prev_iter->i <= sorted_pairs[n].i, because sorted_pairs are
    // sorted on i.
    int32_t c = prev_iter->c + 1;
    sorted_pairs[n].prev_n = prev_iter->n; // backtrace for newly added element.
    iter->c = c;
    // Erase all the elements gmc with gmc.j > j and gmc.count <= c:
    // for these j values, the count value is not as good as the one we just
    // added so we'll never want to backtrace to them.
    ++iter;
    while (iter != cur_counts.end() && iter->c <= c)
      ++iter;
    ++next_iter;
    cur_counts.erase(next_iter, iter);
  }
  assert(best_trace);
  best_trace->clear();
  auto iter = --cur_counts.end();
  auto gm = sorted_pairs[iter->n];
  while (true) {
    best_trace->emplace_back(std::make_pair(gm.i, gm.j));
    if (gm.prev_n == -1)
      break;
    gm = sorted_pairs[gm.prev_n];
  }
  std::reverse(best_trace->begin(), best_trace->end());
}

void GetLongestIncreasingPairsSimple(
    const int32_t *seq1, const int32_t *seq2, int32_t size,
    std::vector<std::pair<int32_t, int32_t>> *best_trace) {
  assert(size > 0);
  // sort this by i, then j.  prev_n values can be undefined for now.
  std::vector<internal::GoodMatch> sorted_pairs(size);
  for (int32_t i = 0; i < size; ++i) {
    sorted_pairs[i] = internal::GoodMatch(seq1[i], seq2[i]);
  }
  std::sort(sorted_pairs.begin(), sorted_pairs.end());

  std::vector<internal::GoodMatchCount> cur_counts;
  cur_counts.emplace_back(internal::GoodMatchCount(0, -1, 0));

  // for each pair p == (i, j) in the sorted pairs, set
  // count[p] = 1 + max_q(count[q])
  // where q is taken from all previously processed pairs, but limited to those
  // where q.i <= p.i and q.j <= p.j.
  int32_t N = sorted_pairs.size();
  for (int32_t n = 0; n < N; ++n) {
    int32_t j = sorted_pairs[n].j;
    auto max_count = internal::GoodMatchCount(0, 0, -1);
    for (const auto &it : cur_counts) {
      if (it.j <= j) {
        if (it.c >= max_count.c) {
          max_count = it;
        }
      }
    }
    sorted_pairs[n].prev_n = max_count.n;
    cur_counts.emplace_back(internal::GoodMatchCount(j, n, max_count.c + 1));
  }
  auto max_count = internal::GoodMatchCount(0, 0, -1);
  for (const auto &it : cur_counts) {
    if (it.c >= max_count.c) {
      max_count = it;
    }
  }
  auto gm = sorted_pairs[max_count.n];
  assert(best_trace);
  best_trace->clear();
  while (true) {
    best_trace->emplace_back(std::make_pair(gm.i, gm.j));
    if (gm.prev_n == -1)
      break;
    gm = sorted_pairs[gm.prev_n];
  }
  std::reverse(best_trace->begin(), best_trace->end());
}

} // namespace fasttextsearch
