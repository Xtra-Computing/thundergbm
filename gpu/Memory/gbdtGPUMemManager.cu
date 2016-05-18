/*
 * gbdtGPUMemManager.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>

#include "gbdtGPUMemManager.h"
#include "../../pureHost/MyAssert.h"

//memory for instances
int *GBDTGPUMemManager::m_pDInsId = NULL;				//all the instance ids for each key-value pair
float_point *GBDTGPUMemManager::m_pdDFeaValue = NULL; //all the feature values
int *GBDTGPUMemManager::m_pDNumofKeyValue = NULL;		//the number of key-value pairs of each feature
long long *GBDTGPUMemManager::m_pFeaStartPos = NULL;	//start key-value position of each feature
int *GBDTGPUMemManager::m_pInsIdToNodeId = NULL; 		//map instance id to node id
long long GBDTGPUMemManager::m_totalNumofValues = -1;
int GBDTGPUMemManager::m_numofIns = -1;
int GBDTGPUMemManager::m_numofFea = -1;

//memory for gradient and hessian
float_point *GBDTGPUMemManager::m_pGrad = NULL;
float_point *GBDTGPUMemManager::m_pHess = NULL;

//memory for splittable nodes
int GBDTGPUMemManager::m_maxNumofSplittable = -1;
TreeNode *GBDTGPUMemManager::m_pSplittableNode = NULL;
SplitPoint *GBDTGPUMemManager::m_pBestSplitPoint = NULL;//(require memset!) store the best split points
nodeStat *GBDTGPUMemManager::m_pSNodeStat = NULL;	//splittable node statistics
nodeStat *GBDTGPUMemManager::m_pRChildStat = NULL;
nodeStat *GBDTGPUMemManager::m_pLChildStat = NULL;
nodeStat *GBDTGPUMemManager::m_pTempRChildStat = NULL;//(require memset!) store temporary statistics of right child
float_point *GBDTGPUMemManager::m_pLastValue = NULL;//store the last processed value (for computing split point)

int *GBDTGPUMemManager::m_pSNIdToBuffId = NULL;	//(require memset!) map splittable node id to buffer position
int *GBDTGPUMemManager::m_pBuffIdVec = NULL;	//store all the buffer ids for splittable nodes
int *GBDTGPUMemManager::m_pNumofBuffId = NULL;	//the total number of buffer ids in the current round.

//host memory for GPU memory reset
SplitPoint *GBDTGPUMemManager::m_pBestPointHost = NULL;//best split points
int *GBDTGPUMemManager::m_pInsToNodeIdHost = NULL;//map instance id to snode id

/**
 * @brief: allocate memory for instances
 */
void GBDTGPUMemManager::allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature)
{
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	PROCESS_ERROR(numofIns > 0);
	m_totalNumofValues = nTotalNumofValue;
	m_numofIns = numofIns;
	m_numofFea = numofFeature;
	checkCudaErrors(cudaMalloc((void**)&m_pDInsId, sizeof(int) * m_totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&m_pdDFeaValue, sizeof(float_point) * m_totalNumofValues));
	checkCudaErrors(cudaMalloc((void**)&m_pDNumofKeyValue, sizeof(int) * m_numofFea));

	checkCudaErrors(cudaMalloc((void**)&m_pInsIdToNodeId, sizeof(int) * m_numofIns));
	checkCudaErrors(cudaMalloc((void**)&m_pFeaStartPos, sizeof(long long) * m_numofIns));

	//gradient and hessian
	checkCudaErrors(cudaMalloc((void**)&m_pGrad, sizeof(float_point) * m_numofIns));
	checkCudaErrors(cudaMalloc((void**)&m_pHess, sizeof(float_point) * m_numofIns));
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

	checkCudaErrors(cudaMalloc((void**)&m_pSplittableNode, sizeof(TreeNode) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pBestSplitPoint, sizeof(SplitPoint) * m_maxNumofSplittable));

	checkCudaErrors(cudaMalloc((void**)&m_pSNodeStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pRChildStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pLChildStat, sizeof(nodeStat) * m_maxNumofSplittable));

	//temporary space for splittable nodes
	checkCudaErrors(cudaMalloc((void**)&m_pTempRChildStat, sizeof(nodeStat) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pLastValue, sizeof(float_point) * m_maxNumofSplittable));
	checkCudaErrors(cudaMemset(m_pLastValue, 0, sizeof(float_point) * m_maxNumofSplittable));


	//map splittable node to buffer id
	checkCudaErrors(cudaMalloc((void**)&m_pSNIdToBuffId, sizeof(int) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pBuffIdVec, sizeof(int) * m_maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofBuffId, sizeof(int)));
}

/**
 * @brief: allocate some host memory for GPU memory reset
 */
void GBDTGPUMemManager::allocHostMemory()
{
	m_pBestPointHost = new SplitPoint[m_maxNumofSplittable];
	m_pInsToNodeIdHost = new int[m_numofIns];
}

/**
 * @brief: release host memory
 */
void GBDTGPUMemManager::releaseHostMemory()
{
	delete []m_pBestPointHost;
	delete []m_pInsToNodeIdHost;
}
