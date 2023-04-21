// textsearch/csrc/utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "textsearch/csrc/utils.h"
#include <assert.h>
#include <map>
#include <vector>

namespace fasttextsearch {

void RowIdsToRowSplits(int32_t num_elems, const uint32_t *row_ids,
                       int32_t num_rows, uint32_t *row_splits) {
  if (num_elems == 0) {
    for (int32_t i = 0; i != num_rows + 1; ++i) {
      row_splits[i] = 0;
    }
    return;
  }

  int32_t cur_row = -1;
  for (int32_t i = 0; i < num_elems; ++i) {
    int32_t row = row_ids[i];
    assert(row >= cur_row);
    while (cur_row < row) {
      cur_row++;
      row_splits[cur_row] = i;
    }
  }
  // cur_row must be >= 0 here as num_elems > 0
  assert(cur_row >= 0);
  while (cur_row < num_rows) {
    row_splits[++cur_row] = num_elems;
  }
}

void FindCloseMatches(const int32_t *suffix_array, int32_t seq_len,
                      int32_t query_len, int32_t num_close_matches,
                      int32_t *close_matches) {
  assert(num_close_matches > 0 && num_close_matches % 2 == 0);

  for (int32_t i = 0; i < query_len * num_close_matches; ++i)
    close_matches[i] = seq_len - 2;

  int32_t half_num_close_matches = num_close_matches / 2;
  // Each query has multiple close_matches, unfinished_q contains those queries
  // that don't have enough close_matches, the key is query index in
  // [0, query_len), the value is the position next close_matches will write to
  // in [0, num_close_matches).
  auto unfinished_q = std::map<int32_t, int32_t>();
  // prev_refs contains the previous reference indexes of current pos, with the
  // help of `prev_ref_start_index` it is functioned as a FIFO queue.
  auto prev_refs = std::vector<int32_t>(half_num_close_matches, seq_len - 2);
  int32_t prev_ref_start_index = 0;

  int32_t refs_index = 0;
  // suffix_array[seq_len - 1] is the appended EOS, should not be included.
  for (int32_t i = 0; i < seq_len - 1; ++i) {
    int32_t text_pos = suffix_array[i];
    // When meeting a reference.
    if (text_pos >= query_len) {
      prev_refs[refs_index % half_num_close_matches] = text_pos;
      prev_ref_start_index =
          (prev_ref_start_index + 1) % half_num_close_matches;

      refs_index += 1;
      for (auto it = unfinished_q.begin(); it != unfinished_q.end();) {
        close_matches[it->first * num_close_matches + it->second] = text_pos;
        if (it->second == num_close_matches - 1) {
          it = unfinished_q.erase(it);
        } else {
          it->second += 1;
          ++it;
        }
      }
    } else { // When meeting a query
      for (int32_t j = 0; j < half_num_close_matches; ++j) {
        close_matches[text_pos * num_close_matches + j] =
            prev_refs[(j + prev_ref_start_index) % half_num_close_matches];
      }
      unfinished_q[text_pos] = half_num_close_matches;
    }
  }
}

} // namespace fasttextsearch
