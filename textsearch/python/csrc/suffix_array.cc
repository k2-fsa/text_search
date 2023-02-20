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

template <typename T>
static py::array_t<int64_t> PybindSuffixArrayHelper(py::array_t<T> &input) {

  py::buffer_info input_buf = input.request();

  if (input_buf.ndim != 1)
    throw std::runtime_error("Input MUST be a one dimension array");

  int64_t input_length = input_buf.size;
  int64_t seq_len = input_length - 3;
  T *input_data = static_cast<T *>(input_buf.ptr);

  // copy input to int64_t array
  auto input64 = py::array_t<int64_t>(input_length);
  py::buffer_info input64_buf = input64.request();
  int64_t *input64_data = static_cast<int64_t *>(input64_buf.ptr);
  for (int64_t i = 0; i < input_length; ++i)
    input64_data[i] = input_data[i];

  auto suffix_array = py::array_t<int64_t>(seq_len);
  py::buffer_info sa_buf = suffix_array.request();
  int64_t *sa_data = static_cast<int64_t *>(sa_buf.ptr);

  int64_t max_symbol = input_data[seq_len - 1];
  if (max_symbol != std::numeric_limits<T>::max() - 1)
    throw std::runtime_error(
        "max_symbol MUST be the largest element of given type");

  CreateSuffixArray<int64_t>(input64_data, seq_len, max_symbol, sa_data);
  return suffix_array;
}

void PybindSuffixArray(py::module &m) {
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int32_t>,
        py::arg("input"));
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int16_t>,
        py::arg("input"));
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int8_t>,
        py::arg("input"));
}
} // namespace fasttextsearch
