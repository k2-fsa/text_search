cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(textsearch)

set(TS_VERSION "0.6")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(TS_RPATH_ORIGIN "$ORIGIN")
else()
  set(TS_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${TS_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${TS_RPATH_ORIGIN})

set(BUILD_SHARED_LIBS ON)
if(WIN32)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TS_ENABLE_TESTS "Whether to build tests" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++14" TS_COMPILER_SUPPORTS_CXX14)
else()
  # windows x86 or x86_64
  check_cxx_compiler_flag("/std:c++14" TS_COMPILER_SUPPORTS_CXX14)
endif()
if(NOT TS_COMPILER_SUPPORTS_CXX14)
  message(FATAL_ERROR "
    Text search requires a compiler supporting at least C++14.
    If you are using GCC, please upgrade it to at least version 7.0.
    If you are using Clang, please upgrade it to at least version 3.4.")
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

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

if(TS_ENABLE_TESTS)
  include(googletest)
  enable_testing()
endif()

add_subdirectory(textsearch)
