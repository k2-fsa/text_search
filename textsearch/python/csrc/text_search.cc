// textsearch/python/csrc/text_search.cc
//
// Copyright (c)  2023  Xiaomi Corporation (authors: Wei Kang)

#include "textsearch/python/csrc/text_search.h"

#include "textsearch/python/csrc/levenshtein.h"
#include "textsearch/python/csrc/match.h"
#include "textsearch/python/csrc/suffix_array.h"
#include "textsearch/python/csrc/utils.h"

namespace fasttextsearch {

PYBIND11_MODULE(_textsearch, m) {
  m.doc() = "Python wrapper for textsearch";

  PybindLevenshtein(m);
  PybindMatch(m);
  PybindSuffixArray(m);
  PybindUtils(m);
}

} // namespace fasttextsearch
