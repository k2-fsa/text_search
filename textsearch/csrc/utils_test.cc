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
#include <cstdlib>
#include <ctime>
#include <random>
#include <sstream>
#include <vector>

#include "textsearch/csrc/utils.h"

namespace fasttextsearch {

TEST(RowIdsToRowSplits, TestBasic) {
  std::vector<uint32_t> row_ids({0, 0, 1, 1, 1, 3, 3, 5});
  std::vector<uint32_t> expected_row_splits({0, 2, 5, 5, 7, 7, 8});
  std::vector<uint32_t> row_splits(expected_row_splits.size());
  RowIdsToRowSplits(row_ids.size(), row_ids.data(), row_splits.size() - 1,
                    row_splits.data());
  for (int32_t i = 0; i < expected_row_splits.size(); ++i) {
    EXPECT_EQ(row_splits[i], expected_row_splits[i]);
  }
}

TEST(FindCloseMatches, TestBasic) {
  std::vector<int32_t> suffix_array({6,  1,  23, 28, 5,  0,  22, 11, 10, 2,
                                     7,  24, 13, 3,  8,  25, 14, 27, 4,  9,
                                     12, 26, 20, 17, 15, 21, 18, 19, 16, 29});
  int32_t query_len = 10, num_close_matches = 2;
  std::vector<int32_t> close_matches(query_len * num_close_matches);
  FindCloseMatches(suffix_array.data(), suffix_array.size(), query_len,
                   num_close_matches, close_matches.data());

  std::vector<int32_t> expected_close_matches(
      {28, 22, 28, 23, 10, 24, 13, 25, 27, 12,
       28, 22, 28, 23, 10, 24, 13, 25, 27, 12}

  );
  for (int32_t i = 0; i < expected_close_matches.size(); ++i) {
    EXPECT_EQ(close_matches[i], expected_close_matches[i]);
  }

  num_close_matches = 4;
  close_matches.resize(query_len * num_close_matches);
  FindCloseMatches(suffix_array.data(), suffix_array.size(), query_len,
                   num_close_matches, close_matches.data());
  expected_close_matches = std::vector<int32_t>(
      {23, 28, 22, 11, 28, 28, 23, 28, 11, 10, 24, 13, 24, 13,
       25, 14, 14, 27, 12, 26, 23, 28, 22, 11, 28, 28, 23, 28,
       11, 10, 24, 13, 24, 13, 25, 14, 14, 27, 12, 26});
  for (int32_t i = 0; i < expected_close_matches.size(); ++i) {
    EXPECT_EQ(close_matches[i], expected_close_matches[i]);
  }
}

TEST(FindCloseMatches, TestRandom) {
  std::srand(std::time(0)); // use current time as seed for random generator
  int32_t seq_len = std::rand() % 1000 + 10000;
  int32_t query_len = std::rand() % 1000 + 1000;
  int32_t num_close_matches = std::rand() % 10 + 2;
  if (num_close_matches % 2)
    num_close_matches += 1;
  std::vector<int32_t> raw_data(seq_len + num_close_matches, seq_len - 2);
  for (int32_t i = num_close_matches / 2;
       i < seq_len + num_close_matches / 2 - 1; ++i) {
    raw_data[i] = i - num_close_matches / 2;
  }

  // simulate suffix_array by random shuffle.
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(raw_data.begin() + num_close_matches / 2,
               raw_data.begin() + num_close_matches / 2 + seq_len - 1, g);
  std::vector<int32_t> suffix_array(raw_data.begin() + num_close_matches / 2,
                                    raw_data.begin() + num_close_matches / 2 +
                                        seq_len);

  std::vector<int32_t> close_matches(query_len * num_close_matches);
  FindCloseMatches(suffix_array.data(), suffix_array.size(), query_len,
                   num_close_matches, close_matches.data());

  std::vector<int32_t> expected_close_matches(query_len * num_close_matches);
  for (int32_t i = 0; i < raw_data.size(); ++i) {
    if (raw_data[i] < query_len) {
      // find preceding references
      int32_t k = num_close_matches / 2 - 1;
      int32_t j = i - 1;
      while (k >= 0) {
        if (raw_data[j] >= query_len) {
          expected_close_matches[raw_data[i] * num_close_matches + k] =
              raw_data[j];
          --k;
        }
        --j;
      }
      // find following references
      k = num_close_matches / 2;
      j = i + 1;
      while (k < num_close_matches) {
        if (raw_data[j] >= query_len) {
          expected_close_matches[raw_data[i] * num_close_matches + k] =
              raw_data[j];
          ++k;
        }
        ++j;
      }
    }
  }

  for (int32_t i = 0; i < close_matches.size(); ++i) {
    EXPECT_EQ(close_matches[i], expected_close_matches[i]);
  }
}

} // namespace fasttextsearch
