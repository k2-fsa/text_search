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

#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>

#include "textsearch/csrc/levenshtein.h"

namespace fasttextsearch {

TEST(Levenshtein, TestBasic) {
  auto query = std::vector<int32_t>({1, 3, 4, 5, 6, 7, 8, 9});
  auto target = std::vector<int32_t>({1, 4, 5, 3, 7, 8, 9, 5, 3, 6, 3, 4, 5, 6,
                                      7, 8, 2, 3, 5, 1, 2, 3, 4, 5, 6, 7, 8});

  std::vector<AlignItem> alignments;
  auto result = LevenshteinDistance(query.data(), query.size(), target.data(),
                                    target.size(), &alignments);

  EXPECT_EQ(result, 2);
  EXPECT_EQ(alignments.size(), 4);
  auto expected_end = std::vector<int64_t>({6, 15, 16, 26});
  auto expected_start = std::vector<int64_t>({0, 10, 10, 19});
  auto expected_align = std::vector<std::string>(
      {"CICCSCCC", "ICCCCCCI", "ICCCCCCS", "CDCCCCCCI"});
  for (size_t i = 0; i < alignments.size(); ++i) {
    auto &align = alignments[i];
    EXPECT_EQ(align.cost, 2);
    EXPECT_EQ(align.start, expected_start[i]);
    EXPECT_EQ(align.end, expected_end[i]);
    EXPECT_EQ(align.align, expected_align[i]);
  }
}

TEST(Levenshtein, TestBasicGlobal) {
  auto query = std::vector<int32_t>({1, 3, 4, 5, 6, 7, 8, 9});
  auto target = std::vector<int32_t>({2, 1, 4, 5, 3, 7, 8, 9, 5, 3});

  std::vector<AlignItem> alignments;
  auto result =
      LevenshteinDistance(query.data(), query.size(), target.data(),
                          target.size(), &alignments, "global" /*mode*/);

  EXPECT_EQ(result, 5);
  EXPECT_EQ(alignments.size(), 1);
  auto &align = alignments[0];
  EXPECT_EQ(align.cost, 5);
  EXPECT_EQ(align.start, 0);
  EXPECT_EQ(align.end, 9);
  EXPECT_EQ(align.align, "DCICCSCCCDD");
}

TEST(Levenshtein, TestRandom) {
  std::srand(std::time(0)); // use current time as seed for random generator
  int32_t ref_len = std::rand() % 1000 + 10000;
  int32_t start = std::rand() % 1000 + 1000;
  int32_t query_len = std::rand() % 1000 + 1000;
  std::vector<int32_t> ref(ref_len);
  for (int32_t i = 0; i < ref_len; ++i)
    ref[i] = i;
  std::vector<int32_t> query;

  std::ostringstream oss;
  int32_t cost = 0;
  int32_t prev_type = 0;
  for (int32_t i = start; i < query_len + start; ++i) {
    // 0: I, 1: D, 2: R, other : E
    int32_t type = std::rand() % 25;
    // No successive errors
    if (prev_type < 3 && type < 3) {
      type = 4;
      prev_type = 4;
    } else {
      prev_type = type;
    }
    // The suffix chars should be right, R and I have same cost, but
    // different end position.
    if (i > query_len + start - 10)
      type = 4;
    if (type == 0) {
      query.push_back(ref_len);
      query.push_back(ref[i]);
      oss << "IC";
      cost += 1;
    } else if (type == 1) {
      oss << "D";
      cost += 1;
    } else if (type == 2) {
      query.push_back(ref_len);
      oss << "S";
      cost += 1;
    } else {
      query.push_back(ref[i]);
      oss << "C";
    }
  }
  auto align_str = oss.str();
  std::vector<AlignItem> alignments;
  auto result = LevenshteinDistance(query.data(), query.size(), ref.data(),
                                    ref.size(), &alignments);
  EXPECT_EQ(result, cost);
  EXPECT_EQ(alignments.size(), 1);
  auto &align = alignments[0];
  EXPECT_EQ(align.cost, cost);
  EXPECT_EQ(align.start, start);
  EXPECT_EQ(align.end, start + query_len - 1);
  EXPECT_EQ(align.align, align_str);
}

} // namespace fasttextsearch
