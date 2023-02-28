// textsearch/python/csrc/text_search.cc
//
// Copyright (c)  2023  Xiaomi Corporation (authors: Wei Kang)

#include "textsearch/python/csrc/text_search.h"

#include "textsearch/python/csrc/levenshtein.h"
#include "textsearch/python/csrc/suffix_array.h"

namespace fasttextsearch {

PYBIND11_MODULE(_fasttextsearch, m) {
  m.doc() = "Python wrapper for textsearch";

  PybindLevenshtein(m);
  PybindSuffixArray(m);
}

} // namespace fasttextsearch
