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

Note:
  We are doing the infix search, gaps at query end and start are not
  penalized. What that means is that deleting elements from the start and end
  of target is "free"! For example, if we had ACT and CGACTGAC, the levenshtein
  distance would be 0, because removing CG from the start and GAC from the end
  of target is "free" and does not count into total levenshtein distance.

Args:
  query:
    The query sequence, it is a one dimension numpy ndarray, only np.int32 is
    supported now.
  target:
    The target sequence, it is a one dimension numpy ndarray with same dtype as
    query sequence.
  insert_cost:
    The cost of insertion error, default 1.
  delete_cost:
    The cost of deletion error, default 1.
  replace_cost:
    The cost of replacement error, default 1.

Returns:
  Return a tuple which has two elements, the first element is the levenshtein
  distance, the second element is a list of tuple, each tuple in the list has
  `end_position` (index into the target sequence) and `alignment`
  (`0,1` string containing the backtrace). See the examples below for more
  details.

>>> from textsearch import levenshtein_distance
>>> import numpy as np
>>> query = np.array([1, 2, 3, 4], dtype=np.int32)
>>> target = np.array([1, 5, 3, 4, 6, 7, 1, 2, 4], dtype=np.int32)
>>> distance, alignments = levenshtein_distance(query, target)
>>> print (distance, alignments)
1 [(3, '01010101'), (8, '0101101')]

The result above indicates that there are two segments in target sequence have the
same levenshtein distance with query sequence. The levenshtein distance is 1,
the end index of first segment into target sequence is 3 ([1,5,3,4]), and the
end index of second sequence is 8 ([1,2,4]). For the backtrace string, '1' means
consuming a query symbol, '0' means consuming a target symbol, we can interpret
the backtrace string like this, from the ending to the beginning, if we meet a
'1' followed by a '0', it means equal or replacement (consuming both query
target); if we meet a '1', it means insertion; if we meet a '0', it means
deletion. So the alignment of first segment is [euqal, replacement, equal, equal],
the alignment of the second segment is [equal, equal, insertion, equal]. We can
distinct equal and replacement with the help of query and target sequences.
)doc";

template <typename T>
static std::pair<int32_t, std::vector<std::pair<int64_t, std::string>>>
PybindLevenshteinHelper(py::array_t<T, py::array::c_style> &query,
                        py::array_t<T, py::array::c_style> &target,
                        int32_t insert_cost, int32_t delete_cost,
                        int32_t replace_cost) {

  py::buffer_info query_buf = query.request();
  py::buffer_info target_buf = target.request();

  if (query_buf.ndim != 1)
    throw std::runtime_error("Query MUST be a one dimension array");
  if (target_buf.ndim != 1)
    throw std::runtime_error("Target MUST be a one dimension array");

  T *query_data = static_cast<T *>(query_buf.ptr);
  T *target_data = static_cast<T *>(target_buf.ptr);

  std::vector<LevenshteinElement> alignments;

  int32_t distance = LevenshteinDistance(
      query_data, query_buf.size, target_data, target_buf.size, &alignments,
      insert_cost, delete_cost, replace_cost);

  std::vector<std::pair<int64_t, std::string>> trace;
  for (const auto align : alignments) {
    trace.push_back(std::make_pair(align.position, align.backtrace.ToString()));
  }

  return std::make_pair(distance, trace);
}

void PybindLevenshtein(py::module &m) {
  m.def("levenshtein_distance", &PybindLevenshteinHelper<int32_t>,
        py::arg("query"), py::arg("target"), py::arg("insert_cost") = 1,
        py::arg("delete_cost") = 1, py::arg("replace_cost") = 1,
        kLevenshteinDistanceDoc);
}
} // namespace fasttextsearch
