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
 * @brief: copy data from device to host
 */
void GPUMemManager::MemcpyDeviceToHost(void *pDevSrc, void *pHostDst, int numofByte)
{
	PROCESS_ERROR(numofByte > 0);
	PROCESS_ERROR(pDevSrc != NULL);
	PROCESS_ERROR(pHostDst != NULL);
	checkCudaErrors(cudaMemcpy(pHostDst, pDevSrc, numofByte, cudaMemcpyDeviceToHost));
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

/**
 * @brief:
 */
void GPUMemManager::TestMemcpyDeviceToHost()
{
	int numofEle = 10;
	int *hostValues = new int[numofEle];
	for(int i = 0; i < numofEle; i++)
	{
		hostValues[i] = i;
	}

	int *devValues;
	checkCudaErrors(cudaMalloc((void**)&devValues, sizeof(int) * numofEle));

	MemcpyHostToDevice(hostValues, devValues, numofEle * sizeof(int));

	int *hostValues2 = new int[numofEle];

	MemcpyDeviceToHost(devValues, hostValues2, numofEle * sizeof(int));

	for(int i = 0; i < numofEle; i++)
	{
		PROCESS_ERROR(hostValues[i] == hostValues2[i]);
	}

	delete []hostValues;
	delete []hostValues2;
}

