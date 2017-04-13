/*
 * findFeaMemManager.cu
 *
 *  Created on: 16 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>
#include <iostream>

#include "findFeaMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../DeviceHost/svm-shared/MemInfo.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/HostUtility.h"

#ifdef testing
#undef testing
#endif

using std::cout;
using std::endl;

long long FFMemManager::m_totalEleInWholeBatch = -1; //a private variable
int FFMemManager::maxNumofSNodeInFF = -1;	//maximum number of splittable nodes in each round of find fea, due to the GPU memory constraint.

//for dense array
float_point *FFMemManager::pGDEachFeaValue = NULL;
float_point *FFMemManager::pHessEachFeaValue = NULL;
float_point *FFMemManager::pDenseFeaValue = NULL;	//feature values of consideration
float_point *FFMemManager::pGDPrefixSum = NULL;
float_point *FFMemManager::pHessPrefixSum = NULL;
float_point *FFMemManager::pGainEachFeaValue = NULL;
int FFMemManager::m_totalNumFeaValue = -1;
float_point *FFMemManager::pfLocalBestGain_d = NULL;
int *FFMemManager::pnLocalBestGainKey_d = NULL;
float_point *FFMemManager::pfGlobalBestGain_d = NULL;
int *FFMemManager::pnGlobalBestGainKey_d = NULL;
//corresponding to pinned memory
int *FFMemManager::m_pIndices_d = NULL;
long long *FFMemManager::m_pFeaValueStartPosEachNode_d = NULL;
long long *FFMemManager::m_pNumFeaValueEachNode_d = NULL;
long long *FFMemManager::m_pEachFeaStartPosEachNode_d = NULL;
int *FFMemManager::m_pEachFeaLenEachNode_d = NULL;

/**
 * @brief: get the maximum number of splittable nodes that can be processed in each round of findFea
 */
int FFMemManager::getMaxNumofSN(int numofValuesInABatch, int maxNumofNode)
{
	long long nFloatPoint = MemInfo::GetFreeGPUMem();

	int tempMaxNumofSN = nFloatPoint / (numofValuesInABatch * 8);//7 such batches for find fea function, using 8 to reserve extra memory for other usage.
	PROCESS_ERROR(tempMaxNumofSN > 0);
	if(tempMaxNumofSN > maxNumofNode)
		tempMaxNumofSN = maxNumofNode;

	int round = Ceil(maxNumofNode, tempMaxNumofSN);
	cout << "find fea requires " << round << " round(s) for the last level of " << maxNumofNode << " nodes" << endl;
	maxNumofSNodeInFF = Ceil(maxNumofNode, round);//take the average number of nodes

	return maxNumofSNodeInFF;
}

/**
 * @brief: allocate memory for finding best feature
 */
void FFMemManager::allocMemForFindFea(int numofValuesInABatch, int maxNumofValuePerFea, int maxNumofFea, int maxNumofSN)
{
	PROCESS_ERROR(numofValuesInABatch > 0);
	int maxNumofNode = maxNumofSNodeInFF;
	PROCESS_ERROR(maxNumofNode > 0);
	long long totalEleInWholeBatch = numofValuesInABatch * maxNumofNode;
	m_totalEleInWholeBatch = totalEleInWholeBatch;

	//for dense array
	PROCESS_ERROR(m_totalNumFeaValue > 0);
	checkCudaErrors(cudaMalloc((void**)&pGDEachFeaValue, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMalloc((void**)&pHessEachFeaValue, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMalloc((void**)&pDenseFeaValue, sizeof(float_point) * m_totalNumFeaValue));

	checkCudaErrors(cudaMalloc((void**)&pGDPrefixSum, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMalloc((void**)&pHessPrefixSum, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMalloc((void**)&pGainEachFeaValue, sizeof(float_point) * m_totalNumFeaValue));
	int blockSizeLocalBest;
	dim3 tempNumofBlockLocalBest;
	KernelConf conf;
	conf.ConfKernel(m_totalNumFeaValue, blockSizeLocalBest, tempNumofBlockLocalBest);
	int maxNumofBlockPerNode = tempNumofBlockLocalBest.x * tempNumofBlockLocalBest.y;
	checkCudaErrors(cudaMalloc((void**)&pfLocalBestGain_d, sizeof(float_point) * maxNumofBlockPerNode * maxNumofSN));
	checkCudaErrors(cudaMalloc((void**)&pnLocalBestGainKey_d, sizeof(int) * maxNumofBlockPerNode * maxNumofSN));
	checkCudaErrors(cudaMalloc((void**)&pfGlobalBestGain_d, sizeof(float_point) * maxNumofSN));
	checkCudaErrors(cudaMalloc((void**)&pnGlobalBestGainKey_d, sizeof(int) * maxNumofSN));
	//corresponding to pinned memory
	checkCudaErrors(cudaMalloc((void**)&m_pIndices_d, sizeof(int) * m_totalNumFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pNumFeaValueEachNode_d, sizeof(long long) * maxNumofSN));
	checkCudaErrors(cudaMalloc((void**)&m_pFeaValueStartPosEachNode_d, sizeof(long long) * maxNumofSN));
	checkCudaErrors(cudaMalloc((void**)&m_pEachFeaStartPosEachNode_d, sizeof(long long) * maxNumofSN * maxNumofFea));
	checkCudaErrors(cudaMalloc((void**)&m_pEachFeaLenEachNode_d, sizeof(int) * maxNumofSN * maxNumofFea));
}

/**
 * @brief: reset memory
 */
void FFMemManager::resetMemForFindFea()
{
	//for dense array
	checkCudaErrors(cudaMemset(pGDEachFeaValue, 0, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMemset(pHessEachFeaValue, 0, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMemset(pDenseFeaValue, 0, sizeof(float_point) * m_totalNumFeaValue));

	checkCudaErrors(cudaMemset(pGDPrefixSum, 0, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMemset(pHessPrefixSum, 0, sizeof(float_point) * m_totalNumFeaValue));
	checkCudaErrors(cudaMemset(pGainEachFeaValue, 0, sizeof(float_point) * m_totalNumFeaValue));
}

/**
 * @brief: free memory
 */
void FFMemManager::freeMemForFindFea()
{

}
