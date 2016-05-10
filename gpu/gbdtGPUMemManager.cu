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

//memory for instances
int *GBDTGPUMemManager::pDInsId = NULL;				//all the instance ids
float_point *GBDTGPUMemManager::pdDFeaValue = NULL; //all the feature values
int *GBDTGPUMemManager::pDNumofKeyValue = NULL;		//the number of key-value pairs of each feature
long long *GBDTGPUMemManager::pFeaStartPos = NULL;	//start key-value position of each feature
int *GBDTGPUMemManager::pInsIdToNodeId = NULL; 		//map instance id to node id
long long GBDTGPUMemManager::totalNumofValues = -1;
int GBDTGPUMemManager::m_numofIns = -1;
int GBDTGPUMemManager::m_numofFea = -1;

//memory for gradient and hessian
float_point *GBDTGPUMemManager::pGrad = NULL;
float_point *GBDTGPUMemManager::pHess = NULL;

//memory for splittable nodes
int GBDTGPUMemManager::m_maxNumofSplittable = -1;
TreeNode *GBDTGPUMemManager::pSplittableNode = NULL;
SplitPoint *GBDTGPUMemManager::pBestSplitPoint = NULL;
nodeStat *GBDTGPUMemManager::pSNodeStat = NULL;
nodeStat *GBDTGPUMemManager::pRChildStat = NULL;
nodeStat *GBDTGPUMemManager::pLChildStat = NULL;
nodeStat *GBDTGPUMemManager::pTempRChildStat = NULL;//store temporary statistics of right child
float_point *GBDTGPUMemManager::pLastValue = NULL;	//store the last processed value (for computing split point)

int *GBDTGPUMemManager::pSNIdToBuffId = NULL;	//map splittable node id to buffer position

/**
 * @brief: allocate memory for instances
 */
void GBDTGPUMemManager::allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature)
{
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	PROCESS_ERROR(numofIns > 0);
	totalNumofValues = nTotalNumofValue;
	m_numofIns = numofIns;
	m_numofFea = numofFeature;
	checkCudaErrors(cudaMalloc((void**)&pDInsId, sizeof(int) * totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&pdDFeaValue, sizeof(float_point) * totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&pDNumofKeyValue, sizeof(int) * m_numofFea));

	checkCudaErrors(cudaMalloc((void**)&pInsIdToNodeId, sizeof(int) * m_numofIns));
	checkCudaErrors(cudaMalloc((void**)&pFeaStartPos, sizeof(long long) * m_numofIns));

	//gradient and hessian
	checkCudaErrors(cudaMalloc((void**)&pGrad, sizeof(float_point) * m_numofIns));
	checkCudaErrors(cudaMalloc((void**)&pHess, sizeof(float_point) * m_numofIns));
}

/**
 * @brief: allocate memory for splittable nodes
 */
void GBDTGPUMemManager::allocMemForSplittableNode(int nMaxNumofSplittableNode)
{
	PROCESS_ERROR(nMaxNumofSplittableNode > 0);
	PROCESS_ERROR(sizeof(TreeNode) > sizeof(int) * 9);
	PROCESS_ERROR(m_maxNumofSplittable == -1);

	m_maxNumofSplittable = nMaxNumofSplittableNode;

	checkCudaErrors(cudaMalloc((void**)&pSplittableNode, sizeof(TreeNode) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&pBestSplitPoint, sizeof(SplitPoint) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&pSNodeStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&pRChildStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&pLChildStat, sizeof(nodeStat) * m_maxNumofSplittable));

	//temporary space for splittable nodes
	checkCudaErrors(cudaMalloc((void**)&pTempRChildStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&pLastValue, sizeof(float_point) * m_maxNumofSplittable));


	//map splittable node to buffer id
	checkCudaErrors(cudaMalloc((void**)&pSNIdToBuffId, sizeof(int) * m_maxNumofSplittable));
	checkCudaErrors(cudaMemset(pSNIdToBuffId, -1, sizeof(int) * m_maxNumofSplittable));
}
