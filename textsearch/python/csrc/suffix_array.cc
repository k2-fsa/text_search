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

static constexpr const char *kCreateSuffixArrayDoc = R"doc(
Creates a suffix array from the input text and returns it as a NumPy array.  Read
the usage carefully as it has some special requirements that will require careful data
preparation.

Args:
   input: an np.ndarray that must be one types uint8, uint16 or uint32.  Its shape
      should be (seq_len + 3,) where `seq_len` is the text sequence length INCLUDING
      EOS SYMBOL.
      The EOS (end of sequence) symbol must be the largest element of the
      type (i.e. of the form 2^n - 1), must be located at input[seq_len - 1] and
      must appear nowhere else in `input` (you may have to map the input
      symbols somehow to achieve this).  It must be followed by 3 zeros, for reasons
      related to how the algorithm works.
Returns:
      Returns a suffix array of type np.uint64,
      of shape (seq_len,).  This will consist of some permutation of the elements
      0 .. seq_len - 1.
)doc";

template <typename T>
static py::array_t<uint64_t> PybindSuffixArrayHelper(py::array_t<T> &input) {

  py::buffer_info input_buf = input.request();

  if (input_buf.ndim != 1)
    throw std::runtime_error("Input MUST be a one dimension array");

  uint64_t input_length = input_buf.size;
  uint64_t seq_len = input_length - 3;
  T *input_data = static_cast<T *>(input_buf.ptr);

  // copy input to uint64_t array
  auto input64 = py::array_t<uint64_t>(input_length);
  py::buffer_info input64_buf = input64.request();
  uint64_t *input64_data = static_cast<uint64_t *>(input64_buf.ptr);
  for (uint64_t i = 0; i < input_length; ++i)
    input64_data[i] = input_data[i];

  auto suffix_array = py::array_t<uint64_t>(seq_len);
  py::buffer_info sa_buf = suffix_array.request();
  uint64_t *sa_data = static_cast<uint64_t *>(sa_buf.ptr);

  uint64_t max_symbol = input_data[seq_len - 1];
  if (max_symbol != std::numeric_limits<T>::max() - 1)
    throw std::runtime_error(
        "max_symbol MUST be the largest element of given type");

  CreateSuffixArray<uint64_t>(input64_data, seq_len, max_symbol, sa_data);
  return suffix_array;
}

void PybindSuffixArray(py::module &m) {
  m.def("create_suffix_array", &PybindSuffixArrayHelper<uint32_t>,
        py::arg("input"), kCreateSuffixArrayDoc);
  m.def("create_suffix_array", &PybindSuffixArrayHelper<uint16_t>,
        py::arg("input"), kCreateSuffixArrayDoc);
  m.def("create_suffix_array", &PybindSuffixArrayHelper<uint8_t>,
        py::arg("input"), kCreateSuffixArrayDoc);
}
} // namespace fasttextsearch
