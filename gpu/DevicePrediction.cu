/*
 * DevicePrediction.cu
 *
 *  Created on: 23 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "../DeviceHost/DefineConst.h"
#include "ErrorChecker.h"
#include "DeviceHashing.h"
#include "../DeviceHost/TreeNode.h"

__device__ int GetNext(TreeNode *pNode, float_point feaValue)
{
    if(feaValue < pNode->fSplitValue)
    {
      return pNode->leftChildId;
    }
    else
    {
      return pNode->rightChildId;
    }
}

__global__ void PredTarget(TreeNode *pAllTreeNode, int totalNode, float_point *pDenseIns, int nNumofFea,
									   int *pnHashFeaIdToPos, float_point *pdTargetValue)
{
	int pid = 0; //node id
	TreeNode *curNode = &pAllTreeNode[pid];
	while(curNode->featureId != -1)//!curNode->isLeaf()
	{
		int fid = curNode->featureId;
		ErrorChecker(fid, __PRETTY_FUNCTION__, "fid < 0");

		int maxNumofUsedFea = nNumofFea;
		int pos = GetBufferId(pnHashFeaIdToPos, fid, maxNumofUsedFea);
//		printf("%d hash to %d: fea v=%f\n", fid, pos, pDenseIns[pos]);

		if(pos < nNumofFea)//feature value is available in the dense vector
			pid = GetNext(curNode, pDenseIns[pos]);
		else//feature value is stored in the dense vector (due to truncating)
			pid = GetNext(curNode, 0);
		curNode = &pAllTreeNode[pid];
	}

	pdTargetValue[0] = pAllTreeNode[pid].predValue;
}



