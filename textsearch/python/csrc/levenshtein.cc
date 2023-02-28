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

static std::pair<int32_t, std::vector<std::pair<int64_t, std::string>>>
PybindStringLevenshteinHelper(const std::string &query,
                              const std::string &target, int32_t insert_cost,
                              int32_t delete_cost, int32_t replace_cost) {
  std::vector<LevenshteinElement> alignments;

  int32_t distance = LevenshteinDistance(
      query.data(), query.size(), target.data(), target.size(), &alignments,
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
        py::arg("delete_cost") = 1, py::arg("replace_cost") = 1);
  m.def("levenshtein_distance", &PybindStringLevenshteinHelper,
        py::arg("query"), py::arg("target"), py::arg("insert_cost") = 1,
        py::arg("delete_cost") = 1, py::arg("replace_cost") = 1);
}
} // namespace fasttextsearch
