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

#ifndef TEXTSEARCH_CSRC_LEVENSHTEIN_H_
#define TEXTSEARCH_CSRC_LEVENSHTEIN_H_

#include <algorithm>
#include <bitset>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fasttextsearch {

struct DynamicBacktrace {
  int64_t bitmap;
  std::shared_ptr<DynamicBacktrace> prev;

  DynamicBacktrace(int64_t bitmap, std::shared_ptr<DynamicBacktrace> prev)
      : bitmap(bitmap), prev(prev) {}

  explicit DynamicBacktrace(int64_t bitmap) : bitmap(bitmap) {}

  DynamicBacktrace &operator=(const DynamicBacktrace &src) = default;
  DynamicBacktrace(const DynamicBacktrace &src) = default;
  // Move constructor
  DynamicBacktrace(DynamicBacktrace &&src) = default;
  DynamicBacktrace &operator=(DynamicBacktrace &&src) = default;
};

struct Backtrace {
  int64_t bitmap; // bitmap records shape of most recent backtrace, where every
                  // time we consume a query symbol we do: self.bitmap |= (1 <<
                  // (num_bits++));   and every time we consume a reference
                  // symbol we do: self.bitmap  |= (0 << (num_bits++));  which
                  // can be optimized to: num_bits++;  When num_bits reaches 64
                  // we allocate a DynamicBacktrace object and reset it to 0.
  int32_t num_bits;
  std::shared_ptr<DynamicBacktrace> prev;

  Backtrace(int64_t bitmap, int32_t num_bits,
            std::shared_ptr<DynamicBacktrace> &prev)
      : bitmap(bitmap), num_bits(num_bits), prev(prev) {}

  Backtrace() : bitmap(0), num_bits(0), prev(nullptr){};

  Backtrace &operator=(const Backtrace &src) = default;
  Backtrace(const Backtrace &src) = default;
  // Move constructor
  Backtrace(Backtrace &&src) = default;
  Backtrace &operator=(Backtrace &&src) = default;

  void Update(bool update_bitmap) {
    if (update_bitmap)
      bitmap |= 1 << num_bits;

    num_bits++;

    if (num_bits == 64) {
      if (prev == nullptr) {
        prev = std::make_shared<DynamicBacktrace>(bitmap);
      } else {
        prev = std::make_shared<DynamicBacktrace>(bitmap, prev->prev);
      }
      bitmap = 0;
      num_bits = 0;
    }
  }

  std::string ToString() const {
    std::ostringstream oss;
    if (num_bits != 0) {
      auto str = std::bitset<64>(bitmap).to_string().substr(64 - num_bits);
      oss << str.substr(0, num_bits);
    }
    auto trace = prev;
    while (trace != nullptr) {
      oss << std::bitset<64>(trace->bitmap);
      trace = trace->prev;
    }
    auto str = oss.str();
    std::reverse(str.begin(), str.end());
    return str;
  }
};

struct LevenshteinElement {
  int32_t cost;
  int64_t position;
  Backtrace backtrace;

  explicit LevenshteinElement(int32_t cost) : cost(cost) {}

  LevenshteinElement(int32_t cost, Backtrace backtrace)
      : cost(cost), backtrace(backtrace) {}

  LevenshteinElement() = default;

  LevenshteinElement &operator=(const LevenshteinElement &src) = default;
  LevenshteinElement(const LevenshteinElement &src) = default;
  // Move constructor
  LevenshteinElement(LevenshteinElement &&src) = default;
  LevenshteinElement &operator=(LevenshteinElement &&src) = default;

  LevenshteinElement Delete(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    res.backtrace.Update(false);
    return res;
  }

  LevenshteinElement Insert(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    res.backtrace.Update(true);
    return res;
  }

  LevenshteinElement Replace(int32_t c) const {
    auto res = LevenshteinElement(cost + c, backtrace);
    res.backtrace.Update(false);
    res.backtrace.Update(true);
    return res;
  }

  LevenshteinElement Equal() const {
    auto res = LevenshteinElement(cost, backtrace);
    res.backtrace.Update(false);
    res.backtrace.Update(true);
    return res;
  }
};

template <typename T>
int32_t LevenshteinDistance(const T *query, size_t query_length,
                            const T *target, size_t target_length,
                            std::vector<LevenshteinElement> *alignments,
                            int32_t insert_cost = 1, int32_t delete_cost = 1,
                            int32_t replace_cost = 1) {

  LevenshteinElement best_score = LevenshteinElement(-1);

  // TODO(WeiKang) Handle edge cases
  if (query_length == 0 || target_length == 0) {
    assert(false);
    return query_length;
  }

  auto scores = std::vector<LevenshteinElement>(query_length + 1);

  scores[0] = LevenshteinElement(0);
  for (size_t i = 1; i <= query_length; i++) {
    scores[i] = scores[i - 1].Insert(insert_cost);
  }

  for (size_t j = 1; j <= target_length; j++) {
    LevenshteinElement prev_diag = scores[0], prev_diag_cache;

    // infix search
    scores[0] = LevenshteinElement(0);

    for (size_t k = 1; k <= query_length; k++) {
      prev_diag_cache = scores[k];
      if (query[k - 1] == target[j - 1]) {
        scores[k] = prev_diag.Equal(); // equal
      } else {
        if (scores[k].cost <= scores[k - 1].cost &&
            scores[k].cost <= prev_diag.cost) { // deletion
          scores[k] = scores[k].Delete(delete_cost);
        } else if (scores[k - 1].cost <= scores[k].cost &&
                   scores[k - 1].cost <= prev_diag.cost) { // insertion
          scores[k] = scores[k - 1].Insert(insert_cost);
        } else { // replacement
          scores[k] = prev_diag.Replace(replace_cost);
        }
      }
      prev_diag = prev_diag_cache;
    }

    auto score = scores[query_length];
    if (best_score.cost == -1 || score.cost <= best_score.cost) {
      if (score.cost < best_score.cost) {
        alignments->clear();
      }
      best_score = score;
      score.position = j - 1;
      alignments->push_back(score);
    }
  }
  return best_score.cost;
}
} // namespace fasttextsearch

#endif // TEXTSEARCH_CSRC_LEVENSHTEIN_H_
