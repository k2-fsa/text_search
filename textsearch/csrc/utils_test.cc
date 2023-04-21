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

#include "textsearch/csrc/utils.h"

namespace fasttextsearch {

TEST(GetNew2Old, TestBasic) {
  std::vector<bool> keep({0, 1, 1, 0, 0, 1, 1, 1, 0});
  std::vector<uint32_t> expected_new2old({1, 2, 5, 6, 7});
  std::vector<uint32_t> new2old;
  GetNew2Old(keep.data(), keep.size(), &new2old);
  EXPECT_EQ(new2old.size(), expected_new2old.size());
  for (int32_t i = 0; i < new2old.size(); ++i) {
    EXPECT_EQ(new2old[i], expected_new2old[i]);
  }
}

TEST(GetNew2Old, TestRandom) {
  std::srand(std::time(0)); // use current time as seed for random generator
  int32_t old_elems_num = std::rand() % 1000 + 1000;
  std::vector<uint32_t> expected_new2old;
  std::vector<bool> keep;
  for (int32_t i = 0; i < old_elems_num; ++i) {
    bool k = std::rand() % 2;
    keep.push_back(k);
    if (k) {
      expected_new2old.push_back(i);
    }
  }
  std::vector<uint32_t> new2old;
  GetNew2Old(keep.data(), keep.size(), &new2old);
  EXPECT_EQ(new2old.size(), expected_new2old.size());
  for (int32_t i = 0; i < new2old.size(); ++i) {
    EXPECT_EQ(new2old[i], expected_new2old[i]);
  }
}

TEST(RowIdsToRowSplits, TestBasic) {}

TEST(FindCloseMatches, TestBasic) {}

TEST(FindCloseMatches, TestRandom) {}

} // namespace fasttextsearch
