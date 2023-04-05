// textsearch/csrc/utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "textsearch/csrc/utils.h"
#include <assert.h>
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

void GetNew2Old(const bool *keep, uint32_t num_old_elems,
                std::vector<uint32_t> *new2old) {
  auto old2new = std::vector<uint32_t>(num_old_elems);

  // Exclusive sum
  uint32_t sum = 0;
  for (size_t i = 0; i != old2new.size(); ++i) {
    old2new[i] = sum;
    sum += (int)keep[i];
  }
  uint32_t num_new_elems = sum;

  assert(num_new_elems >= 0);
  assert(num_new_elems <= num_old_elems);

  *new2old = std::vector<uint32_t>(num_new_elems + 1);
  for (size_t i = 0; i != old2new.size(); ++i) {
    if (i == old2new.size() - 1 || old2new[i + 1] > old2new[i])
      (*new2old)[old2new[i]] = i;
  }
  new2old->resize(num_new_elems);
}

} // namespace fasttextsearch
