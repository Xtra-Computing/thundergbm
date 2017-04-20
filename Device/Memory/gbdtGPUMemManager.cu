/*
 * gbdtGPUMemManager.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>

#include "gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"

//memory for instances (key on feature id)
int *GBDTGPUMemManager::m_pDInsId = NULL;				//all the instance ids for each key-value pair
float_point *GBDTGPUMemManager::m_pdDFeaValue = NULL; 	//all the feature values
int *GBDTGPUMemManager::m_pDNumofKeyValue = NULL;		//the number of key-value pairs of each feature
unsigned int *GBDTGPUMemManager::m_pFeaStartPos = NULL;	//start key-value position of each feature
//memory for instances (key on instance id)
int *GBDTGPUMemManager::m_pDFeaId = NULL;				//all the feature ids for every instance
float_point *GBDTGPUMemManager::m_pdDInsValue = NULL;	//all the feature values for every instance
int *GBDTGPUMemManager::m_pDNumofFea = NULL;			//the number of features for each instance
long long *GBDTGPUMemManager::m_pInsStartPos = NULL;	//the start position of each instance

//memory for prediction
int GBDTGPUMemManager::m_maxUsedFeaInTrees = -1;		//maximum number of used features in all the trees

unsigned int GBDTGPUMemManager::m_numFeaValue = 0;
int GBDTGPUMemManager::m_numofIns = -1;
int GBDTGPUMemManager::m_numofFea = -1;

//memory for splittable nodes
int GBDTGPUMemManager::m_maxNumofSplittable = -1;

/**
 * @brief: allocate memory for instances
 */
void GBDTGPUMemManager::allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature){
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	PROCESS_ERROR(numofIns > 0);
	m_numFeaValue = nTotalNumofValue;
	m_numofIns = numofIns;
	m_numofFea = numofFeature;

	//memory for instances (key on feature id)
	checkCudaErrors(cudaMalloc((void**)&m_pDInsId, sizeof(int) * m_numFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pdDFeaValue, sizeof(float_point) * m_numFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pDNumofKeyValue, sizeof(int) * m_numofFea));
	checkCudaErrors(cudaMalloc((void**)&m_pFeaStartPos, sizeof(unsigned int) * m_numofFea));
	//memory for instances (key on instance id)
	checkCudaErrors(cudaMalloc((void**)&m_pDFeaId, sizeof(int) * m_numFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pdDInsValue, sizeof(float_point) * m_numFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pDNumofFea, sizeof(int) * m_numofIns));
	checkCudaErrors(cudaMalloc((void**)&m_pInsStartPos, sizeof(long long) * m_numofIns));
}

void GBDTGPUMemManager::freeMemForIns(){
	//memory for instances (key on feature id)
	checkCudaErrors(cudaFree(m_pDInsId));
	checkCudaErrors(cudaFree(m_pdDFeaValue));
	checkCudaErrors(cudaFree(m_pDNumofKeyValue));
	checkCudaErrors(cudaFree(m_pFeaStartPos));
	//memory for instances (key on instance id)
	checkCudaErrors(cudaFree(m_pDFeaId));
	checkCudaErrors(cudaFree(m_pdDInsValue));
	checkCudaErrors(cudaFree(m_pDNumofFea));
	checkCudaErrors(cudaFree(m_pInsStartPos));
}
