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

#include "textsearch/python/csrc/match.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "textsearch/csrc/match.h"
#include <cassert>

namespace fasttextsearch {

static void PybindGetLongestIncreasingPairs(py::module &m) {
  m.def(
      "get_longest_increasing_pairs",
      [](py::array_t<int32_t> seq1, py::array_t<int32_t> seq2)
          -> std::vector<std::pair<int32_t, int32_t>> {
        py::buffer_info seq1_buf = seq1.request();
        assert(seq1_buf.ndim == 1);
        const int32_t *p_seq1 = static_cast<const int32_t *>(seq1_buf.ptr);

        py::buffer_info seq2_buf = seq2.request();
        assert(seq2_buf.ndim == 1);
        const int32_t *p_seq2 = static_cast<const int32_t *>(seq2_buf.ptr);

        assert(seq1_buf.size == seq2_buf.size);
        std::vector<std::pair<int32_t, int32_t>> trace;
        {
          py::gil_scoped_release release;
          GetLongestIncreasingPairs(p_seq1, p_seq2, seq1_buf.size, &trace);
        }
        return trace;
      },
      py::arg("seq1"), py::arg("seq2"));
}

void PybindMatch(py::module &m) { PybindGetLongestIncreasingPairs(m); }

} // namespace fasttextsearch
