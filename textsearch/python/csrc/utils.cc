// textsearch/python/csrc/utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "textsearch/python/csrc/utils.h"
#include "textsearch/csrc/utils.h"
#include <cassert>

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

static constexpr const char *kFindCloseMatchesDoc = R"doc(
Assuming the suffix array was created from a text where the first
``query_len`` positions represent the query text and the remaining positions
represent the reference text, return a list indicating, for each suffix
position in the query text, the ``num_close_matches`` suffix positions in the
reference text that immediately precede and follow it lexicographically.
(I think suffix position refers to the last character of a suffix).

(Note: the query and reference texts could each represent multiple separate
sequences, but that is handled by other code; class SourcedText keeps track
of that information.)

Args:
 suffix_array:
   A suffix array as created by :func:`create_suffix_array`, of dtype
   ``np.int32`` and shape ``(seq_len,)``.

 query_len:
   A number ``0 <= query_len < seq_len``, indicating the length in symbols
   (likely bytes) of the query part of the text that was used to create
   ``suffix_array``.
 num_close_matches:
   The number of close_matches for each query element.

Returns:
  Return an np.ndarray of shape ``(query_len, num_close_matches,)``, of the
  same dtype as ``suffix_array``, in which positions
  ``(i, 0), (i, 1),... (i, num_close_matches - 1)`` represent
  the num_close_matches positions in the original text that are within the
  reference portion, and which immediately precede and follow, in the suffix
  array, query position ``i``.  This means that the suffixes ending at those
  positions are reverse-lexicographically close to the suffix ending at
  position ``i``.  As a special case, if one of these returned numbers would
  equal the EOS position (position seq_len - 1), or if a query position is
  before any reference position in the suffix aray, we output
  ``seq_len - 2`` instead to avoid having to handle special cases later on
  (anyway, these would not represent a close match).

.. hint::

    Please refer to :ref:`find_close_matches_tutorial` for usages.
"""
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
        {
          py::gil_scoped_release release;
          RowIdsToRowSplits(num_elems, p_row_ids, num_rows, p_row_splits);
        }
      },
      py::arg("row_ids"), py::arg("row_splits"), kRowIdsToRowSplitsDoc);
}

static void PybindFindCloseMatches(py::module &m) {
  m.def(
      "find_close_matches",
      [](py::array_t<int32_t> suffix_array, int32_t query_len,
         int32_t num_close_matches) -> py::array_t<int32_t> {
        assert(num_close_matches > 0 && num_close_matches % 2 == 0);
        py::buffer_info suffix_buf = suffix_array.request();
        int32_t seq_len = suffix_buf.size;
        assert(query_len < seq_len);
        const int32_t *p_suffix = static_cast<const int32_t *>(suffix_buf.ptr);
        auto close_matches =
            py::array_t<int32_t>(query_len * num_close_matches);
        py::buffer_info matches_buf = close_matches.request();
        int32_t *p_matches = static_cast<int32_t *>(matches_buf.ptr);
        {
          py::gil_scoped_release release;
          FindCloseMatches(p_suffix, seq_len, query_len, num_close_matches,
                           p_matches);
        }
        close_matches = close_matches.reshape({query_len, num_close_matches});
        return close_matches;
      },
      py::arg("suffix_array"), py::arg("query_len"),
      py::arg("num_close_matches") = 2, kFindCloseMatchesDoc);
}

void PybindUtils(py::module &m) {
  PybindFindCloseMatches(m);
  PybindRowIdsToRowSplits(m);
}

} // namespace fasttextsearch
