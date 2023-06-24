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

#include "textsearch/python/csrc/levenshtein.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "textsearch/csrc/levenshtein.h"
#include <iostream>
#include <limits>

namespace fasttextsearch {

static constexpr const char *kLevenshteinDistanceDoc = R"doc(
Calculate the levenshtein distance between query and target sequence.

Args:
  query:
    The query sequence; it is a 1-D numpy ndarray, only ``np.int32`` is
    supported now.
  target:
    The target sequence, it is a 1-D numpy ndarray with same dtype as
    query sequence.
  mode:
    It can be either "global" or "infix", when it equals "global", we are doing
    the normal levenshtein distance, when it equals "infix", it means we are
    doing the infix search, gaps at query end and start are not penalized. What
    that means is that deleting elements from the start and end of target is
    "free"! For example, if we had ACT and CGACTGAC, the levenshtein distance
    would be 0, because removing CG from the start and GAC from the end
    of target is "free" and does not count into total levenshtein distance.
    Default: infix.
  insert_cost:
    The cost of insertion error, default 1.
  delete_cost:
    The cost of deletion error, default 1.
  replace_cost:
    The cost of replacement error, default 1.

Returns:
  Return a tuple which has two elements, the first element is the levenshtein
  distance, the second element is a list of tuple, each tuple in the list has
  `start` and `end` (index into the target sequence, for "global" mode, `start`
  is always 0, `end` is always `len(target) - 1`) and `alignment`.
  See the examples below for more details.

>>> from textsearch import levenshtein_distance
>>> import numpy as np
>>> query = np.array([1, 2, 3, 4], dtype=np.int32)
>>> target = np.array([1, 5, 3, 4, 6, 7, 1, 2, 4], dtype=np.int32)
>>> distance, alignments = levenshtein_distance(query, target)
>>> print(distance, alignments)
1 [(0, 3, 'CSCC'), (6, 8, 'CCIC')]

The result above indicates that there are two segments in target sequence having
the same levenshtein distance with query sequence. The levenshtein distance is 1,
the end index of first segment into target sequence is 3 ([1,5,3,4]), and the
end index of second sequence is 8 ([1,2,4]). For the align string, `I` means
insertion, `D` means deletion, `S` means substitution, `C` means correct.

>>> from textsearch import levenshtein_distance
>>> import numpy as np
>>> query = np.array([1, 2, 3, 4], dtype=np.int32)
>>> target = np.array([1, 5, 3, 4, 6, 7, 1, 2, 4], dtype=np.int32)
>>> distance, alignments = levenshtein_distance(query, target, model="global")
>>> print (distance, alignments)
6 [(0, 8, 'CSCDDDDDC')]

.. hint::

   Please refer to :func:`textsearch.get_nice_alignments` for how to visualize
   the alignments.
)doc";

template <typename T>
static std::pair<int32_t,
                 std::vector<std::tuple<int64_t, int64_t, std::string>>>
PybindLevenshteinHelper(py::array_t<T, py::array::c_style> &query,
                        py::array_t<T, py::array::c_style> &target,
                        std::string &mode, int32_t insert_cost,
                        int32_t delete_cost, int32_t replace_cost) {
  if (query.ndim() != 1)
    throw std::runtime_error("Query MUST be a one dimension array");

  if (target.ndim() != 1)
    throw std::runtime_error("Target MUST be a one dimension array");

  auto query_data = query.data();
  auto target_data = target.data();

  py::gil_scoped_release release;

  std::vector<AlignItem> alignments;
  int32_t distance = LevenshteinDistance(
      query_data, query.size(), target_data, target.size(), &alignments, mode,
      insert_cost, delete_cost, replace_cost);

  std::vector<std::tuple<int64_t, int64_t, std::string>> trace;
  trace.reserve(alignments.size());

  for (const auto &align : alignments) {
    trace.push_back(std::make_tuple(align.start, align.end, align.align));
  }

  return std::make_pair(distance, trace);
}

void PybindLevenshtein(py::module &m) {
  m.def("levenshtein_distance", &PybindLevenshteinHelper<int32_t>,
        py::arg("query"), py::arg("target"), py::arg("mode") = "infix",
        py::arg("insert_cost") = 1, py::arg("delete_cost") = 1,
        py::arg("replace_cost") = 1, kLevenshteinDistanceDoc);
}
} // namespace fasttextsearch
