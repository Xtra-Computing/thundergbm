/*
 * gpuMemAllocator.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: define functions for memory allocation
 */

#include <helper_cuda.h>

#include "gpuMemManager.h"
#include "../pureHost/MyAssert.h"


/**
 * @brief: copy data from host to device
 */
void GPUMemManager::MemcpyHostToDevice(void *hostSrc, void *pDevDst, int numofByte)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(hostSrc != NULL);
	PROCESS_ERROR(pDevDst != NULL);

	checkCudaErrors(cudaMemcpy(pDevDst, hostSrc, numofByte, cudaMemcpyHostToDevice));
}

/**
 * @brief: copy data from host to device
 */
void GPUMemManager::TestMemcpyHostToDevice(void *hostSrc, void *pDevDst, int numofByte)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(hostSrc != NULL);
	PROCESS_ERROR(pDevDst != NULL);

	void *hostDst = new char[numofByte];

	checkCudaErrors(cudaMemcpy(hostDst, pDevDst, numofByte, cudaMemcpyDeviceToHost));

	for(int b = 0; b < numofByte; b++)
	{
		PROCESS_ERROR(((char*)hostDst)[b] == ((char*)hostSrc)[b]);
	}

	delete []hostDst;
}
