//
// Created by shijiashuai on 10/7/18.
//

#include <thundergbm/thundergbm.h>
#include "thundergbm/syncarray.h"
#include "gtest/gtest.h"
#include "thrust/reduce.h"
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include "thundergbm/util/device_lambda.cuh"


void kernel(int count, int *ptr) {
    device_loop(count, [=] __device__(int i) { ptr[i] = 1; });
}

TEST(TestUnified, test) {
    int *ptr;
    size_t size = (1L << 30) * 4;
    size_t count = size / sizeof(int);

    using namespace thrust;
    cudaMallocManaged((void **) &ptr, size);

    memset(ptr, 0, size);
    {
        TIMED_SCOPE(timerObj, "prefetch kernel");
        cudaMemPrefetchAsync(ptr, size, 0);

        sort(cuda::par, ptr, ptr + count);
        cudaDeviceSynchronize();
    }

    cudaFree(ptr);
    ptr = nullptr;

    SyncArray<int> arr(count);
    arr.to_host();
    {
        TIMED_SCOPE(timerObj, "copy kernel");
        sort(cuda::par, arr.device_data(), arr.device_end());
    }
}
