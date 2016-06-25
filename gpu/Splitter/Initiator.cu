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

__global__ void ComputeGDKernel(int numofIns, float_point *pfPredValue, float_point *pfTrueValue, float_point *pGrad, float_point *pHess)
{
	for(int i = 0; i < numofIns; i++)
	{
		pGrad[i] = pfPredValue[i] - pfTrueValue[i];
		pHess[i] = 1;
	}

}

__global__ void InitNodeStat(int numofIns, float_point *pGrad, float_point *pHess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable, int *pBuffId)
{
	float_point root_sum_gd = 0.0, root_sum_hess = 0.0;
	for(int i = 0; i < numofIns; i++)
	{
		root_sum_gd += pGrad[i];
		root_sum_hess += pHess[i];
	}

	int snid = 0;
	bool bIsNew = false;
	int buffId = AssignHashValue(pSNIdToBuffId, snid, maxNumofSplittable, bIsNew);
	pSNodeStat[buffId].sum_gd = root_sum_gd;
	pSNodeStat[buffId].sum_hess = root_sum_hess;
	pBuffId[0] = buffId;
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
