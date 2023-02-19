set(textsearch_srcs
  suffix_array.cc
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
endfunction()

if(FTS_ENABLE_TESTS)
  # please sort the source files alphabetically
  set(test_srcs
    suffix_array_test.cc
  )

  foreach(source IN LISTS test_srcs)
    textsearch_add_test(${source})
  endforeach()
endif()

