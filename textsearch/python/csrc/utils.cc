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
  row_splits:
    Pre-allocated array of dtype np.uint32. Its shape is (num_rows+1,).
    On return it will contain the computed row splits.
)doc";

static void PybindRowIdsToRowSplits(py::module &m) {
  m.def(
      "row_ids_to_row_splits",
      [](py::array_t<uint32_t> row_ids,
         py::array_t<uint32_t> *row_splits) -> void {
        py::buffer_info row_ids_buf = row_ids.request();
        const uint32_t *p_row_ids =
            static_cast<const uint32_t *>(row_ids_buf.ptr);

        py::buffer_info row_splits_buf = row_splits->request();
        uint32_t *p_row_splits = static_cast<uint32_t *>(row_splits_buf.ptr);

        int32_t num_elems = row_ids_buf.size;
        int32_t num_rows = row_splits_buf.size - 1;

        RowIdsToRowSplits(num_elems, p_row_ids, num_rows, p_row_splits);
      },
      py::arg("row_ids"), py::arg("row_splits"), kRowIdsToRowSplitsDoc);
}

void PybindUtils(py::module &m) { PybindRowIdsToRowSplits(m); }

} // namespace fasttextsearch
