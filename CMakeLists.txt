cmake_minimum_required(VERSION 3.9)
project(thundergbm)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_60,code=sm_60 --expt-extended-lambda")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
# find pacages
find_package(CUDA REQUIRED QUIET)

find_package(OpenMP REQUIRED QUIET)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_definitions("-DELPP_FEATURE_PERFORMANCE_TRACKING")
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${COMMON_INCLUDES})

add_subdirectory(src/thundergbm)
add_subdirectory(src/test)

# configuration file
set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/thundergbm/config.h.in config.h)

