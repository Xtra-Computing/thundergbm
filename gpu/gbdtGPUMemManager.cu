/*
 * gbdtGPUMemManager.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>

#include "gbdtGPUMemManager.h"
#include "../pureHost/MyAssert.h"

int *GBDTGPUMemManager::pDInsId = NULL;//all the instance ids
double *GBDTGPUMemManager::pdDFeaValue = NULL;//all the feature values
int *GBDTGPUMemManager::pDNumofKeyValue = NULL;//the number of key-value pairs of each feature
long long GBDTGPUMemManager::totalNumofValues = -1;
int GBDTGPUMemManager::m_numofFea = -1;

/**
 * @brief: allocate memory for instances
 */
void GBDTGPUMemManager::allocMemForIns(int nTotalNumofValue, int numofFeature)
{
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	totalNumofValues = nTotalNumofValue;
	m_numofFea = numofFeature;
	checkCudaErrors(cudaMalloc((void**)&pDInsId, sizeof(int) * totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&pdDFeaValue, sizeof(float_point) * totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&pDNumofKeyValue, sizeof(int) * m_numofFea));
}

