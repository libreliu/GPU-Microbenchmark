cmake_minimum_required(VERSION 3.10)
project(GPU-MB LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "50;61;75")
add_subdirectory(code)