cmake_minimum_required(VERSION 3.10)
project(mbSuite)

enable_testing()

option(BUILD_GPU "Build GPU Microbenchmarks" OFF)

include_directories(code-common)

if (BUILD_GPU)
    add_subdirectory(code-gpu)
endif()

add_subdirectory(code-cpu)