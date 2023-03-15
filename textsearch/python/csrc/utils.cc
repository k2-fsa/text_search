// textsearch/python/csrc/utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "textsearch/python/csrc/utils.h"
#include "textsearch/csrc/utils.h"

namespace fasttextsearch {

static constexpr const char *kRowIdsToRowSplitsDoc = R"doc(
Convert row ids to row splits.

Args:
  row_ids:
    A 1-D contiguous array of dtype np.uint32 containing row ids.
Returns:
  Return a 1-D contiguous tensors of dtype np.uint32 containing the
  corresponding row splits.
)doc";

static void PybindRowIdsToRowSplits(py::module &m) {
  m.def(
      "row_ids_to_row_splits",
      [](py::array_t<uint32_t> row_ids) -> py::array_t<uint32_t> {
        py::buffer_info row_ids_buf = row_ids.request();
        uint32_t *p_row_ids = static_cast<uint32_t *>(row_ids_buf.ptr);

        int32_t num_elems = row_ids_buf.size;
        int32_t num_rows = num_elems > 0 ? p_row_ids[num_elems - 1] : 0;

        auto row_splits = py::array_t<int64_t>(num_rows + 1);
        py::buffer_info row_splits_buf = row_splits.request();
        uint32_t *p_row_splits = static_cast<uint32_t *>(row_splits_buf.ptr);
        RowIdsToRowSplits(num_elems, p_row_ids, num_rows, p_row_splits);

        return row_splits;
      },
      py::arg("row_ids"), kRowIdsToRowSplitsDoc);
}

void PybindUtils(py::module &m) { PybindRowIdsToRowSplits(m); }

} // namespace fasttextsearch
