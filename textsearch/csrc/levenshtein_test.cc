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

#include <iostream>
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
      {"EIEEREEE", "IEEEEEEI", "IEEEEEER", "EDEEEEEEI"});
  for (size_t i = 0; i < alignments.size(); ++i) {
    auto &align = alignments[i];
    EXPECT_EQ(align.cost, 2);
    EXPECT_EQ(align.start, expected_start[i]);
    EXPECT_EQ(align.end, expected_end[i]);
    EXPECT_EQ(align.align, expected_align[i]);
  }
}

} // namespace fasttextsearch
