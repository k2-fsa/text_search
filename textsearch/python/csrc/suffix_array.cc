/**
 * Copyright      2023     Xiaomi Corporation (authors: Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "textsearch/python/csrc/suffix_array.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "textsearch/csrc/suffix_array.h"
#include <iostream>
#include <limits>

namespace fasttextsearch {

static py::array_t<int32_t>
PybindSuffixArrayHelper(py::array_t<int32_t> input) {
  py::buffer_info input_buf = input.request();
  int32_t seq_len = input_buf.size - 3;
  auto input_data = static_cast<const int32_t *>(input_buf.ptr);
  auto max_symbol = input_data[seq_len - 1];

  auto suffix_array = py::array_t<int32_t>(seq_len);
  py::buffer_info sa_buf = suffix_array.request();
  auto sa_data = static_cast<int32_t *>(sa_buf.ptr);
  {
    py::gil_scoped_release release;
    CreateSuffixArray<int32_t>(input_data, seq_len, max_symbol, sa_data);
  }
  return suffix_array;
}

void PybindSuffixArray(py::module &m) {
  m.def("create_suffix_array", &PybindSuffixArrayHelper, py::arg("input"));
}
} // namespace fasttextsearch
