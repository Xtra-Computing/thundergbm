/*
 * ComputeGD.cu
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "Initiator.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../DeviceHashing.h"

__global__ void SaveToPredBuffer(const real *pfCurTreePredValue, int numPredIns, real *pfPreTreePredValue)
{

	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= numPredIns)
		return;
	pfPreTreePredValue[gTid] += pfCurTreePredValue[gTid];//accumulate the current prediction to the buffer
}

__global__ void ComputeGDKernel(int numofIns, const real *pfPredValue, const real *pfTrueValue, real *pGrad, real *pHess)
{
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= numofIns)//each thread computes one gd
		return;


	pGrad[gTid] = pfPredValue[gTid] - pfTrueValue[gTid];
	pHess[gTid] = 1;
}

__global__ void InitNodeStat(const real root_sum_gd, const real root_sum_hess,
							 nodeStat *pSNodeStat, int maxNumofSplittable,
							 int *pBuffId, int *pNumofBuffId)
{
	int nid = 0;//id of root node is always 0.
	bool bIsNew = false;
	int buffId = nid % maxNumofSplittable;

	if(buffId != 0)
		printf("buffId = %d\n", buffId);
	pSNodeStat[buffId].sum_gd = root_sum_gd;
	pSNodeStat[buffId].sum_hess = root_sum_hess;
	pBuffId[0] = buffId;//here we only initialise the root node
	pNumofBuffId[0] = 1;
}

/**
 * @brief: initialise the root node of a tree, and the current node of nodes in the tree.
 */
__global__ void InitRootNode(TreeNode *pAllTreeNode, int *pCurNumofNode, int numIns)
{
	pAllTreeNode[0].nodeId = 0;
	pAllTreeNode[0].level = 0;
	*pCurNumofNode = 1;

	pAllTreeNode[0].featureId = -1;
	pAllTreeNode[0].fSplitValue = -1;
	pAllTreeNode[0].leftChildId = -1;
	pAllTreeNode[0].rightChildId = -1;
	pAllTreeNode[0].loss = -1.0;

	pAllTreeNode[0].numIns = numIns;
	pAllTreeNode[0].m_bDefault2Right = false;
}
