cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

# Project

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(
  pyci
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

execute_process(
  COMMAND sh -c "${CMAKE_CXX_COMPILER} --version | head -n 1"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PYCI_COMPILER_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PYCI_GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND date -u +%F\ %T
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE PYCI_BUILD_TIME
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Dependencies

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(BUILD_TESTING OFF CACHE BOOL "Build testing")

set(BUILD_SHARED_LIBS_ORIGINAL ${BUILD_SHARED_LIBS})
set(BUILD_TESTING_ORIGINAL ${BUILD_TESTING})

set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)

include(FetchContent)

find_package(Eigen3 REQUIRED NO_MODULE)

FetchContent_Populate(spectra
  URL https://github.com/yixuan/spectra/archive/refs/tags/v1.0.1.tar.gz
  URL_HASH MD5=4c5364e377f733c3b2574d39a718bc3b
  SOURCE_DIR spectra)

FetchContent_Populate(parallel_hashmap
  URL https://github.com/greg7mdp/parallel-hashmap/archive/refs/tags/v1.4.0.tar.gz
  URL_HASH MD5=264fc23f81d17d30a0ebf34cdec8f4a6
  SOURCE_DIR parallel_hashmap)

FetchContent_Populate(rapidhash
  URL https://github.com/Nicoshev/rapidhash/archive/refs/tags/rapidhash_v1.0.tar.gz
  URL_HASH MD5=eff40e2b9b66e4608ba37709995b5255
  SOURCE_DIR rapidhash)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

set(BUILD_SHARED_LIBS_ORIGINAL ${BUILD_SHARED_LIBS})
set(BUILD_TESTING_ORIGINAL ${BUILD_TESTING})

# Build

file(GLOB_RECURSE PYCI_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

pybind11_add_module(${PROJECT_NAME} MODULE ${PYCI_SOURCE_FILES})

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  OUTPUT_NAME _${PROJECT_NAME}
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/spectra/include
    ${PROJECT_BINARY_DIR}/parallel_hashmap
    ${PROJECT_BINARY_DIR}/rapidhash)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    _PYCI_VERSION="${PROJECT_VERSION}"
    _GIT_BRANCH="${PYCI_GIT_BRANCH}"
    _BUILD_TIME="${PYCI_BUILD_TIME}"
    _COMPILER_VERSION="${PYCI_COMPILER_VERSION}")

# Install

install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME})

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${SKBUILD_PLATLIB_DIR}/pyci/include)
