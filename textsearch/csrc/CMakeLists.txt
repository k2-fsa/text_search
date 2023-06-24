set(textsearch_srcs
  match.cc
  suffix_array.cc
  utils.cc
)

add_library(textsearch_core ${textsearch_srcs})

function(textsearch_add_test source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} "${source}")
  target_link_libraries(${name}
    PRIVATE
      textsearch_core
      gtest
      gtest_main
  )

  add_test(NAME "Test.${name}"
    COMMAND
    $<TARGET_FILE:${name}>
  )
endfunction()

if(TS_ENABLE_TESTS)
  # please sort the source files alphabetically
  set(test_srcs
    levenshtein_test.cc
    match_test.cc
    suffix_array_test.cc
    utils_test.cc
  )

  foreach(source IN LISTS test_srcs)
    textsearch_add_test(${source})
  endforeach()
endif()

