// textsearch/csrc/utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "textsearch/csrc/utils.h"
#include <assert.h>

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

} // namespace fasttextsearch
