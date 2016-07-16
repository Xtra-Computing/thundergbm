/*
 * findFeaMemManager.cu
 *
 *  Created on: 16 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>

#include "findFeaMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../KernelConf.h"

float_point *FFMemManager::m_pGDOnEachFeaValue_d = NULL;	//gradient of each feature list (same size of each node)
float_point *FFMemManager::m_pHessOnEachFeaValue_d = NULL;	//hessian of each feature list (same size of each node)
float_point *FFMemManager::m_pValueOnEachFeaValue_d = NULL;	//fea value of each item in a feature list
float_point *FFMemManager::m_pGainOnEachFeaValue_d = NULL;	//gain of each fea value in the feature list

int *FFMemManager::m_pFeaLenInBatch_d = NULL;	//length of each feature list
int *FFMemManager::m_pnEachFeaLen_h = NULL;	//length of each feature list in host
int *FFMemManager::m_pStartPosEachFeaInBatch_d = NULL; //start position of each feature list in the feature batch

float_point *FFMemManager::m_pGDPrefixSum_d = NULL;		//prefix sum of gradient of each feature list (same size of each node)
float_point *FFMemManager::m_pHessPrefixSum_d = NULL;	//prefix sum of hessian of each feature list (same size of each node)

float_point *FFMemManager::m_pfFeaLocalBestGain_d = NULL;	//feature best gain in block
int *FFMemManager::m_pnFeaLocalBestGainKey_d = NULL;		//feature key of best gain in block
float_point *FFMemManager::m_pfFeaGlobalBestGain_d = NULL;	//feature global best gain
float_point *FFMemManager::m_pnFeaGlobalBestGainKey_d = NULL; //feature key of global best gain

float_point *FFMemManager::m_pfBlockBestFea_d = NULL;	//block level feature with best split
int *FFMemManager::m_pnBlockBestKey_d = NULL;			//block level feature key with best split
float_point *FFMemManager::m_pfGlobalBestFea_d = NULL;	//global level feature with best split
int *FFMemManager::m_pnGlobalBestKey_d = NULL;			//global level feature key with best split

float_point *FFMemManager::m_pLastBiggerValue_d = NULL;	//unused variable

long long FFMemManager::m_totalEleInWholeBatch = -1; //a private variable

/**
 * @brief: allocate memory for finding best feature
 */
void FFMemManager::allocMemForFindFea(int numofValuesInABatch, int maxNumofValuePerFea, int maxNumofFea, int maxNumofNode)
{
	PROCESS_ERROR(numofValuesInABatch > 0);
	PROCESS_ERROR(maxNumofNode > 0);
	long long totalEleInWholeBatch = numofValuesInABatch * maxNumofNode;
	m_totalEleInWholeBatch = totalEleInWholeBatch;
	checkCudaErrors(cudaMalloc((void**)&m_pGDPrefixSum_d, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMalloc((void**)&m_pHessPrefixSum_d, sizeof(float_point) * totalEleInWholeBatch));

	checkCudaErrors(cudaMalloc((void**)&m_pGDOnEachFeaValue_d, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMalloc((void**)&m_pHessOnEachFeaValue_d, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMalloc((void**)&m_pValueOnEachFeaValue_d, sizeof(float_point) * totalEleInWholeBatch));

	checkCudaErrors(cudaMalloc((void**)&m_pStartPosEachFeaInBatch_d, sizeof(int) * maxNumofFea * maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pFeaLenInBatch_d, sizeof(int) * maxNumofFea * maxNumofNode));

	checkCudaErrors(cudaMalloc((void**)&m_pGainOnEachFeaValue_d, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMalloc((void**)&m_pLastBiggerValue_d, sizeof(float_point) * totalEleInWholeBatch));

	int blockSizeFillGD;
	dim3 dimNumofBlockToFillGD;
	KernelConf conf;
	conf.ConfKernel(maxNumofValuePerFea, blockSizeFillGD, dimNumofBlockToFillGD);
	int maxNumofBlockEachFea = dimNumofBlockToFillGD.x;

	checkCudaErrors(cudaMalloc((void**)&m_pfFeaLocalBestGain_d, sizeof(float_point) * maxNumofFea * maxNumofBlockEachFea * maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pnFeaLocalBestGainKey_d, sizeof(int) * maxNumofFea * maxNumofBlockEachFea * maxNumofNode));

	checkCudaErrors(cudaMalloc((void**)&m_pfFeaGlobalBestGain_d, sizeof(float_point) * maxNumofFea * maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pnFeaGlobalBestGainKey_d, sizeof(int) * maxNumofFea * maxNumofNode));

	int blockSizeBestFeaBestSplit;
	dim3 tempNumofBlockBestFea;
	conf.ConfKernel(maxNumofFea, blockSizeBestFeaBestSplit, tempNumofBlockBestFea);
	int nBlockBestFea = tempNumofBlockBestFea.x;

	checkCudaErrors(cudaMalloc((void**)&m_pfBlockBestFea_d, sizeof(float_point) * nBlockBestFea * maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pnBlockBestKey_d, sizeof(int) * nBlockBestFea * maxNumofNode));

	checkCudaErrors(cudaMalloc((void**)&m_pfGlobalBestFea_d, sizeof(float_point) * maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pnGlobalBestKey_d, sizeof(int) * maxNumofNode));


	checkCudaErrors(cudaMemset(m_pGDOnEachFeaValue_d, 0, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMemset(m_pHessOnEachFeaValue_d, 0, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMemset(m_pValueOnEachFeaValue_d, 0, sizeof(float_point) * totalEleInWholeBatch));

	checkCudaErrors(cudaMemset(m_pLastBiggerValue_d, 0, sizeof(float_point) * totalEleInWholeBatch));

	m_pnEachFeaLen_h = new int[maxNumofFea * maxNumofNode];
}

/**
 * @brief: reset memory
 */
void FFMemManager::resetMemForFindFea()
{
	checkCudaErrors(cudaMemset(m_pGDOnEachFeaValue_d, 0, sizeof(float_point) * m_totalEleInWholeBatch));
	checkCudaErrors(cudaMemset(m_pHessOnEachFeaValue_d, 0, sizeof(float_point) * m_totalEleInWholeBatch));
	checkCudaErrors(cudaMemset(m_pValueOnEachFeaValue_d, 0, sizeof(float_point) * m_totalEleInWholeBatch));

	checkCudaErrors(cudaMemset(m_pLastBiggerValue_d, 0, sizeof(float_point) * m_totalEleInWholeBatch));
}

/**
 * @brief: free memory
 */
void FFMemManager::freeMemForFindFea()
{
	checkCudaErrors(cudaFree(m_pGDPrefixSum_d));
	checkCudaErrors(cudaFree(m_pHessPrefixSum_d));
	checkCudaErrors(cudaFree(m_pfGlobalBestFea_d));
	checkCudaErrors(cudaFree(m_pnGlobalBestKey_d));
	checkCudaErrors(cudaFree(m_pfBlockBestFea_d));
	checkCudaErrors(cudaFree(m_pnBlockBestKey_d));
	checkCudaErrors(cudaFree(m_pGDOnEachFeaValue_d));
	checkCudaErrors(cudaFree(m_pHessOnEachFeaValue_d));
	checkCudaErrors(cudaFree(m_pValueOnEachFeaValue_d));
	checkCudaErrors(cudaFree(m_pStartPosEachFeaInBatch_d));
	checkCudaErrors(cudaFree(m_pFeaLenInBatch_d));
	checkCudaErrors(cudaFree(m_pGainOnEachFeaValue_d));
	checkCudaErrors(cudaFree(m_pfFeaLocalBestGain_d));
	checkCudaErrors(cudaFree(m_pnFeaLocalBestGainKey_d));
	checkCudaErrors(cudaFree(m_pfFeaGlobalBestGain_d));
	checkCudaErrors(cudaFree(m_pnFeaGlobalBestGainKey_d));
	delete[] m_pnEachFeaLen_h;
}
