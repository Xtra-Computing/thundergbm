/*
 * DevicePredictorHelper.cu
 *
 *  Created on: 27 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DevicePredictorHelper.h"
#include "../SharedUtility/CudaMacro.h"
#include "DeviceHashing.h"

/**
 * @brief: get the id of next node
 */
__device__ int GetNext(const TreeNode *pNode, real feaValue)
{
	if((((*(int*)&feaValue)) ^ LARGE_REAL_NUM) == 0){//this is a missing value
		if(pNode->m_bDefault2Right == false)
			return pNode->leftChildId;
		else
			return pNode->rightChildId;
	}

    if(feaValue < pNode->fSplitValue)
    {
      return pNode->leftChildId;
    }
    else
    {
      return pNode->rightChildId;
    }
}

__global__ void PredMultiTarget(real *pdTargetValue, int numofDenseIns, const TreeNode *pAllTreeNode,
								const real *pDenseIns, int numofUsedFea,
								const int *pnHashFeaIdToPos, int maxDepth)
{
	int gTid = GLOBAL_TID();
	if(gTid >= numofDenseIns)
		return;
	int targetId = gTid;

	int pid = 0; //node id
	const TreeNode *curNode = pAllTreeNode + pid;
	CONCHECKER(curNode->nodeId == 0);
	int counter = 0;
	while(curNode->featureId != -1)//!curNode->isLeaf()
	{
		int fid = curNode->featureId;
		ECHECKER(fid);

		int maxNumofUsedFea = numofUsedFea;
		int pos = GetBufferId(pnHashFeaIdToPos, fid, maxNumofUsedFea);
//		printf("%d hash to %d: fea v=%f\n", fid, pos, pDenseIns[pos]);

		if(pos < numofUsedFea)//feature value is available in the dense vector
			pid = GetNext(curNode, pDenseIns[targetId * numofUsedFea + pos]);
		else//feature value is stored in the dense vector (due to truncating)
		{
			real temp;
			(*((int*)&temp)) |= LARGE_REAL_NUM;
			pid = GetNext(curNode, temp);
		}
		curNode = pAllTreeNode + pid;

		counter++;
		CONCHECKER(counter <= maxDepth);
	}
	pdTargetValue[targetId] += pAllTreeNode[pid].predValue;
}

__global__ void FillMultiDense(const real *pdSparseInsValue, const uint *pInsStartPos, const int *pnSpareInsFeaId,
							   const int *pNumofFeaValue, real *pdDenseIns, const int *pSortedUsedFea,
							   const int *pHashFeaIdToDenseInsPos, int numofUsedFea,
						  	   int startInsId, int numofInsToFill)
{
	int gTid = GLOBAL_TID();
	if(gTid >= numofInsToFill)
		return;

	int insId = startInsId + gTid;
	uint startPos = pInsStartPos[insId];
	ECHECKER(startPos);
	int numofFeaValue = pNumofFeaValue[insId];
	int denseInsStartPos = gTid * numofUsedFea;

	//for each value in the sparse instance
	int curDenseTop = 0;
	for(int i = 0; i < numofFeaValue; i++)
	{
		int feaId = pnSpareInsFeaId[startPos + i];

		CONCHECKER(curDenseTop < numofUsedFea);
		while(feaId > pSortedUsedFea[curDenseTop])//handle missing values
		{
			curDenseTop++;
			if(curDenseTop >= numofUsedFea)//all the used features have been assigned values
				return;
		}

		CONCHECKER(curDenseTop < numofUsedFea);
		if(feaId == pSortedUsedFea[curDenseTop])
		{//this is a feature needed to be stored in dense instance
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], numofUsedFea);
			pdDenseIns[denseInsStartPos + pos] = pdSparseInsValue[startPos + i];
			curDenseTop++;
			if(curDenseTop >= numofUsedFea)//all the used features have been assigned values
				return;
		}
	}
}
