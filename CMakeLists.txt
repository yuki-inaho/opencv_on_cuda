cmake_minimum_required(VERSION 3.5)
project(opencv_on_cuda)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER "/usr/bin/c++")
set(CMAKE_CXX_FLAGS "-O0 -std=c++11 -fopenmp")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_61,code=sm_61 -ccbin g++
)

cuda_add_executable(opencv_on_cuda src/myKernel.cu src/main.cpp)
#add_executable(opencv_on_cuda src/myKernel.cu src/main.cpp)
include_directories(
  include
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV opencv REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_on_cuda ${OpenCV_LIBRARIES})

