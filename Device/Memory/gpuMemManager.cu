/*
 * gpuMemAllocator.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: define functions for memory allocation
 */

#include <helper_cuda.h>

#include "gpuMemManager.h"
#include "../../SharedUtility/CudaMacro.h"


/**
 * @brief: copy data from host to device
 */
void GPUMemManager::MemcpyHostToDeviceAsync(void *hostSrc, void *pDevDst, long long numofByte, void *pStream)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(hostSrc != NULL);
	PROCESS_ERROR(pDevDst != NULL);

	checkCudaErrors(cudaMemcpyAsync(pDevDst, hostSrc, numofByte, cudaMemcpyHostToDevice, (*(cudaStream_t*)pStream)));
}

/**
 * @brief: copy data from device to host
 */
void GPUMemManager::MemcpyDeviceToHostAsync(void *pDevSrc, void *pHostDst, long long numofByte, void *pStream)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(pDevSrc != NULL);
	PROCESS_ERROR(pHostDst != NULL);
	checkCudaErrors(cudaMemcpyAsync(pHostDst, pDevSrc, numofByte, cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));
}

/**
 * @brief: copy data from device to device
 */
void GPUMemManager::MemcpyDeviceToDeviceAsync(void *pDevSrc, void *pDevDst, long long numofByte, void *pStream)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(pDevSrc != NULL);
	PROCESS_ERROR(pDevDst != NULL);
	checkCudaErrors(cudaMemcpyAsync(pDevDst, pDevSrc, numofByte, cudaMemcpyDeviceToDevice, (*(cudaStream_t*)pStream)));
}

/**
 * @brief: set gpu memory
 */
void GPUMemManager::MemsetAsync(void *pDevSrc, int value, long long numofByte, void *pStream)
{
	checkCudaErrors(cudaMemsetAsync(pDevSrc, value, numofByte, (*(cudaStream_t*)pStream)));
}
