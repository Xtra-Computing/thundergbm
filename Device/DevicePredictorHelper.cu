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

__device__ int GetNext(const TreeNode *pNode, real feaValue)
{
	if(feaValue > LARGE_REAL_NUM - 2){//this is a missing value
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
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofDenseIns)
		return;
	int targetId = nGlobalThreadId;

	int pid = 0; //node id
	const TreeNode *curNode = pAllTreeNode + pid;
	if(curNode->nodeId != 0)
	{
		printf("id of root node is %d should be 0\n", curNode->nodeId);
		return;
	}
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
			pid = GetNext(curNode, LARGE_REAL_NUM);
		curNode = pAllTreeNode + pid;

		counter++;
		if(counter > maxDepth)//for skipping from deadlock
		{
			printf("%s has bugs; fid=%d\n", __PRETTY_FUNCTION__, fid);
			break;
		}
	}

	pdTargetValue[targetId] += pAllTreeNode[pid].predValue;
}

__global__ void FillMultiDense(const real *pdSparseInsValue, const long long *pInsStartPos, const int *pnSpareInsFeaId,
							   const int *pNumofFeaValue, real *pdDenseIns, const int *pSortedUsedFea,
							   const int *pHashFeaIdToDenseInsPos, int numofUsedFea,
						  	   int startInsId, int numofInsToFill)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofInsToFill)
		return;
	if(nGlobalThreadId < 0)
		printf("global id is %d\n", nGlobalThreadId);

	int insId = startInsId + nGlobalThreadId;
	long long startPos = pInsStartPos[insId];
	ECHECKER(startPos);
	int numofFeaValue = pNumofFeaValue[insId];
	int denseInsStartPos = nGlobalThreadId * numofUsedFea;

	//for each value in the sparse instance
	int curDenseTop = 0;
	for(int i = 0; i < numofFeaValue; i++)
	{
		int feaId = pnSpareInsFeaId[startPos + i];

		CONCHECKER(curDenseTop < numofUsedFea);
		while(feaId > pSortedUsedFea[curDenseTop])//handle missing values
		{
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], numofUsedFea);
			pdDenseIns[denseInsStartPos + pos] = LARGE_REAL_NUM;//assign a very large number for missing values
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
