// textsearch/python/csrc/suffix_array.cc
//
// Copyright (c)  2023  Xiaomi Corporation (authors: Wei Kang)

#include "textsearch/python/csrc/suffix_array.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "textsearch/csrc/suffix_array.h"
#include <iostream>
#include <limits>

namespace fasttextsearch {

static py::array_t<int64_t>
PybindSuffixArrayHelper(py::array_t<int64_t, py::array::c_style> &input) {

  py::buffer_info input_buf = input.request();

  if (input_buf.ndim != 1)
    throw std::runtime_error("Input MUST be a one dimension array");

  int64_t seq_len = input_buf.size - 3;
  int64_t *input_data = static_cast<int64_t *>(input_buf.ptr);

  auto suffix_array = py::array_t<int64_t>(seq_len);
  py::buffer_info sa_buf = suffix_array.request();
  int64_t *sa_data = static_cast<int64_t *>(sa_buf.ptr);

  int64_t max_symbol = input_data[seq_len - 1];

  CreateSuffixArray<int64_t>(input_data, seq_len, max_symbol, sa_data);
  return suffix_array;
}

void PybindSuffixArray(py::module &m) {
  m.def("create_suffix_array", &PybindSuffixArrayHelper, py::arg("input"));
}
} // namespace fasttextsearch
