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

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "textsearch/csrc/match.h"

namespace fasttextsearch {

TEST(GetLongestIncreasingPairs, TestBasic) {
  std::vector<int32_t> seq1({0, 1, 1, 2, 2, 3, 4, 5, 6});
  std::vector<int32_t> seq2({9, 7, 8, 9, 6, 7, 10, 12, 8});
  std::vector<std::pair<int32_t, int32_t>> best_trace;
  std::vector<std::pair<int32_t, int32_t>> best_trace_simple;
  GetLongestIncreasingPairs(seq1.data(), seq2.data(), seq1.size(), &best_trace);
  GetLongestIncreasingPairsSimple(seq1.data(), seq2.data(), seq1.size(),
                                  &best_trace_simple);
  std::vector<std::pair<int32_t, int32_t>> expected_trace(
      {{1, 7}, {1, 8}, {2, 9}, {4, 10}, {5, 12}});
  for (int32_t i = 0; i < best_trace.size(); ++i) {
    EXPECT_EQ(best_trace[i], expected_trace[i]);
    EXPECT_EQ(best_trace_simple[i], expected_trace[i]);
  }
}

TEST(GetLongestIncreasingPairs, TestRandom) {
  std::srand(std::time(0)); // use current time as seed for random generator
  int32_t length = std::rand() % 10000 + 10000;
  std::vector<int32_t> seq1(length);
  std::vector<int32_t> seq2(length);
  for (int32_t i = 0; i < length; ++i) {
    seq1[i] = std::rand() % 100000;
    seq2[i] = std::rand() % 100000;
  }
  std::vector<std::pair<int32_t, int32_t>> best_trace;
  std::vector<std::pair<int32_t, int32_t>> best_trace_simple;

  auto begin = std::chrono::steady_clock::now();
  GetLongestIncreasingPairs(seq1.data(), seq2.data(), length, &best_trace);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Sequence length : " << length << std::endl;
  std::cout << "Elapsed(ms)="
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                       .count() /
                   1000.0
            << std::endl;

  begin = std::chrono::steady_clock::now();
  GetLongestIncreasingPairsSimple(seq1.data(), seq2.data(), length,
                                  &best_trace_simple);
  end = std::chrono::steady_clock::now();
  std::cout << "Simple elapsed(ms)="
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                       .count() /
                   1000.0
            << std::endl;

  for (int32_t i = 0; i < best_trace.size(); ++i) {
    EXPECT_EQ(best_trace[i], best_trace_simple[i]);
  }
}

} // namespace fasttextsearch
