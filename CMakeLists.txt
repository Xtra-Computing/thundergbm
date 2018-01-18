cmake_minimum_required(VERSION 3.9)
project(thundergbm)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

# find pacages
find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${COMMON_INCLUDES})

add_subdirectory(src/thundergbm)
add_subdirectory(src/test)

# configuration file
set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/thundergbm/config.h.in config.h)
