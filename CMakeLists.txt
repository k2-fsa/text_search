cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fasttextsearch)

set(FTS_VERSION "1.0")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(FTS_RPATH_ORIGIN "$ORIGIN")
else()
  set(FTS_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${FTS_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${FTS_RPATH_ORIGIN})

set(BUILD_SHARED_LIBS ON)
if(WIN32)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FTS_ENABLE_TESTS "Whether to build tests" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)

include_directories(${CMAKE_SOURCE_DIR})

if(WIN32)
  # disable various warnings for MSVC
  # 4244: 'initializing': conversion from 'float' to 'int32_t',
  # 4267: 'argument': conversion from 'size_t' to 'uint32_t', possible loss of data
  set(disabled_warnings
      /wd4244
      /wd4267
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()
endif()

if(FTS_ENABLE_TESTS)
  include(googletest)
  enable_testing()
endif()

add_subdirectory(textsearch)
