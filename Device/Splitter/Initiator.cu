/*
 * ComputeGD.cu
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "Initiator.h"
#include "../ErrorChecker.h"
#include "../DeviceHashing.h"

__global__ void SaveToPredBuffer(const float_point *pfCurTreePredValue, int numPredIns, float_point *pfPreTreePredValue)
{

	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= numPredIns)
		return;
	pfPreTreePredValue[gTid] += pfCurTreePredValue[gTid];//accumulate the current prediction to the buffer
}

__global__ void ComputeGDKernel(int numofIns, const float_point *pfPredValue, const float_point *pfTrueValue, float_point *pGrad, float_point *pHess)
{
	for(int i = 0; i < numofIns; i++)
	{
		pGrad[i] = pfPredValue[i] - pfTrueValue[i];
//		if(pGrad[i] < -2016 || pGrad[i] > -1920)
//			printf("pGrad is too small: %f\n", pGrad[i]);

		pHess[i] = 1;
	}

}

__global__ void InitNodeStat(int numofIns, const float_point *pGrad, const float_point *pHess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable,
							 int *pBuffId, int *pNumofBuffId)
{
	float_point root_sum_gd = 0.0, root_sum_hess = 0.0;
	for(int i = 0; i < numofIns; i++)
	{
		root_sum_gd += pGrad[i];
		root_sum_hess += pHess[i];
	}

	int nid = 0;//id of root node is always 0.
	bool bIsNew = false;
	int buffId = AssignHashValue(pSNIdToBuffId, nid, maxNumofSplittable, bIsNew);
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
__global__ void InitRootNode(TreeNode *pAllTreeNode, int *pCurNumofNode)
{
	pAllTreeNode[0].nodeId = 0;
	pAllTreeNode[0].level = 0;
	*pCurNumofNode = 1;
}
