cmake_minimum_required(VERSION 3.13)
project(NN_GPU)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O3")

add_executable(NN_GPU main.cpp

        headers/learning.hpp
        headers/ml.hpp
        headers/nn.hpp
        headers/populationtable.hpp
        headers/task.hpp
        headers/samplestable.hpp

        sources/learning.cpp
        sources/nn.cpp
        sources/populationtable.cpp
        sources/samplestable.cpp sources/task.cpp)

include_directories(headers)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(NN_GPU ${OpenCL_LIBRARIES})