//
// Created by jiashuai on 18-1-16.
//

#ifndef THUNDERGBM_COMMON_H
#define THUNDERGBM_COMMON_H


#include "thundergbm/thundergbm.h"

#define USE_CUDA
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (false)

#define NO_GPU \
LOG(FATAL)<<"Cannot use GPU when compiling without GPU"

//https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

#define HOST_DEVICE __host__ __device__
#endif //THUNDERGBM_COMMON_H
