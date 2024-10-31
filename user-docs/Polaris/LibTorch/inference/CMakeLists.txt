cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
project(inference)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")
set(TORCH_LIBS ${TORCH_LIBRARIES})

add_executable(exe inference-example.cpp)
target_link_libraries(exe ${TORCH_LIBS})

set_property(TARGET exe PROPERTY CXX_STANDARD 17)
