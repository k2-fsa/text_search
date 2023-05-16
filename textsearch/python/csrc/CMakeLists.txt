pybind11_add_module(_textsearch
  levenshtein.cc
  match.cc
  suffix_array.cc
  text_search.cc
  utils.cc
)

target_link_libraries(_textsearch PRIVATE textsearch_core)

