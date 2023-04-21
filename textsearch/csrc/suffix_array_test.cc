/**
 * Copyright      2021-2023  Xiaomi Corporation (authors: Daniel Povey
 *                         .                              Wei Kang)
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
#include <random>
#include <vector>

#include "textsearch/csrc/suffix_array.h"

namespace fasttextsearch {

static int32_t RandInt(int32_t min, int32_t max) {
  std::random_device rd; // Only used once to initialise (seed) engine
  std::mt19937 rng(
      rd()); // Random-number engine used (Mersenne-Twister in this case)
  std::uniform_int_distribution<int32_t> uni(min, max); // Guaranteed unbiased

  return uni(rng);
}

TEST(SuffixArrayTest, TestBasic) {

  for (int32_t i = 0; i < 100; i++) {
    int32_t array_len = RandInt(1, 50), // 1 is min, due to termination symbol.
        max_symbol = RandInt(10, 500);

    if (i == 0) {
      array_len = 1;
    }

    std::vector<int32_t> array(array_len + 3);
    int32_t *array_data = array.data();
    for (int32_t i = 0; i + 1 < array_len; i++)
      array_data[i] = RandInt(1, max_symbol - 1); // termination symbol must
                                                  // be larger than all
                                                  // others, don't allow
    array_data[array_len - 1] = max_symbol;       // Termination symbol

    for (int32_t i = array_len; i < array_len + 3; i++)
      array_data[i] = 0;

    // really array_len, extra elem is to test that it doesn't write past
    // the end.
    std::vector<int32_t> suffix_array(array_len + 1);
    int32_t *suffix_array_data = suffix_array.data();
    suffix_array_data[array_len] = -10; // should not be changed.
    CreateSuffixArray<int32_t>(array_data, array_len, max_symbol,
                               suffix_array_data);
    EXPECT_EQ(suffix_array_data[array_len], -10); // should be unchanged.

    std::vector<int32_t> seen_indexes(array_len, 0);
    int32_t *seen_indexes_data = seen_indexes.data();
    for (int32_t i = 0; i < array_len; i++)
      seen_indexes_data[suffix_array_data[i]] = 1;

    for (int32_t i = 0; i < array_len; i++)
      EXPECT_EQ(seen_indexes_data[i], 1); // make sure all integers seen.

    for (int32_t i = 0; i + 1 < array_len; i++) {
      int32_t *suffix_a = array_data + suffix_array_data[i],
              *suffix_b = array_data + suffix_array_data[i + 1];
      // checking that each suffix is lexicographically less than the next one.
      // None are identical, because the terminating zero is always in different
      // positions.
      EXPECT_LE(*suffix_a, *suffix_b);
      ++suffix_a;
      ++suffix_b;

      while (true) {
        if (*suffix_a > *suffix_b)
          break;                         // correct order
        EXPECT_LE(*suffix_a, *suffix_b); // order is wrong!

        // past array end without correct comparison order.
        EXPECT_FALSE(suffix_a > array_data + array_len ||
                     suffix_b > array_data + array_len);
        ++suffix_a;
        ++suffix_b;
      }
    }

    // Test that suffix_array contains a permutation of 0...array_len-1
    std::sort(suffix_array_data, suffix_array_data + array_len);
    for (int32_t i = 0; i + 1 != array_len; ++i) {
      EXPECT_EQ(suffix_array_data[i], i);
    }
  }
}

} // namespace fasttextsearch
