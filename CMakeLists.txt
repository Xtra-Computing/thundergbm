cmake_minimum_required(VERSION 3.1)
project(thundergbm)

find_package(CUDA REQUIRED QUIET)
find_package(OpenMP REQUIRED QUIET)
find_package(Boost COMPONENTS serialization REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo -gencode arch=compute_60,code=sm_60 --expt-extended-lambda --default-stream per-thread")

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# for easylogging++ configuration
add_definitions("-DELPP_FEATURE_PERFORMANCE_TRACKING")
add_definitions("-DELPP_THREAD_SAFE")
add_definitions("-DELPP_STL_LOGGING")

# includes
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cub)
include_directories(${COMMON_INCLUDES})

add_subdirectory(src/thundergbm)

set(BUILD_TESTS OFF CACHE BOOL "Build Tests")
if (BUILD_TESTS)
    add_subdirectory(src/test)
endif ()

# configuration file
set(DATASET_DIR ${PROJECT_SOURCE_DIR}/datasets/)
configure_file(include/thundergbm/config.h.in config.h)

