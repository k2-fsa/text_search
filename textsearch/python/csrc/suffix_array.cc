// textsearch/python/csrc/suffix_array.cc
//
// Copyright (c)  2023  Xiaomi Corporation (authors: Wei Kang)

#include "textsearch/python/csrc/suffix_array.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "textsearch/csrc/suffix_array.h"
#include <limits>

namespace fasttextsearch {

template <typename T>
py::array_t<T> PybindSuffixArrayHelper(py::array_t<T> &input) {

  py::buffer_info input_buf = input.request();

  if (input_buf.ndim != 1)
    throw std::runtime_error("Input MUST be a one dimension array");

  T input_length = input_buf.size;
  T seq_len = input_length - 3;

  auto suffix_array = py::array_t<T>(seq_len);

  py::buffer_info sa_buf = suffix_array.request();

  T *input_data = static_cast<T *>(input_buf.ptr);
  T *sa_data = static_cast<T *>(sa_buf.ptr);
  T max_symbol = input_data[seq_len - 1];
  if (max_symbol != std::numeric_limits<T>::max())
    throw std::runtime_error(
        "max_symbol MUST be the largest element of given type");

  CreateSuffixArray(input_data, seq_len, max_symbol, sa_data);
  return suffix_array;
}

void PybindSuffixArray(py::module &m) {
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int8_t>,
        py::arg("input"));
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int16_t>,
        py::arg("input"));
  m.def("create_suffix_array", &PybindSuffixArrayHelper<int32_t>,
        py::arg("input"));
}
} // namespace fasttextsearch
