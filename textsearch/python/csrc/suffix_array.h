// textsearch/python/csrc/suffix_array.h
//
// Copyright (c)  2023  Xiaomi Corporation (authors: Wei Kang)

#ifndef TEXTSEARCH_PYTHON_CSRC_SUFFIX_ARRAY_H_
#define TEXTSEARCH_PYTHON_CSRC_SUFFIX_ARRAY_H_

#include "textsearch/python/csrc/text_search.h"

namespace fasttextsearch {

void PybindSuffixArray(py::module &m);

} // namespace fasttextsearch

#endif // TEXTSEARCH_PYTHON_CSRC_SUFFIX_ARRAY_H_
