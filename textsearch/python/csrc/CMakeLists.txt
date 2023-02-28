pybind11_add_module(_fasttextsearch
  levenshtein.cc
  suffix_array.cc
  text_search.cc
)

target_link_libraries(_fasttextsearch PRIVATE textsearch_core)

