project(mbSuite-CPU LANGUAGES CXX)

add_executable(pchase-cpu pchase.cpp)
set_target_properties(pchase-cpu PROPERTIES CXX_STANDARD 17)
add_test(NAME t_pchase_cpu COMMAND pchase-cpu ${CMAKE_SOURCE_DIR}/result/pchase-cpu.json 10)