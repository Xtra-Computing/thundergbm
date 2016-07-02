/*
 * DevicePredictorHelper.cu
 *
 *  Created on: 27 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DevicePredictorHelper.h"
#include "ErrorChecker.h"
#include "DeviceHashing.h"

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
									   int *pnHashFeaIdToPos, float_point *pdTargetValue, int maxDepth)
{
	int pid = 0; //node id
	TreeNode *curNode = pAllTreeNode + pid;
	if(curNode->nodeId != 0)
	{
		printf("id of root node is %d should be 0\n", curNode->nodeId);
		return;
	}
	int counter = 0;
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
		curNode = pAllTreeNode + pid;

		counter++;
		if(counter > maxDepth)//for skipping from deadlock
		{
			printf("%s has bugs\n", __PRETTY_FUNCTION__);
			break;
		}
	}

	pdTargetValue[0] += pAllTreeNode[pid].predValue;
}

__global__ void FillDense(const float_point *pdSparseInsValue, const int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, const int *pSortedUsedFea, const int *pHashFeaIdToDenseInsPos, int totalUsedFea)
{
	//for each value in the sparse instance
	ErrorChecker(numofFeaValue - 1, __PRETTY_FUNCTION__, "numofFeaValue <= 0");
	int curDenseTop = 0;
	for(int i = 0; i < numofFeaValue; i++)
	{
		int feaId = pnSpareInsFeaId[i];

		while(feaId > pSortedUsedFea[curDenseTop])
		{
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], totalUsedFea);
			pdDenseIns[pos] = 0;
			curDenseTop++;
		}

		if(feaId == pSortedUsedFea[curDenseTop])
		{//this is a feature needed to be stored in dense instance
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], totalUsedFea);
			pdDenseIns[pos] = pdSparseInsValue[i];
			curDenseTop++;
		}
	}

}

__global__ void FillMultiDense(const float_point *pdSparseInsValue, const long long *pInsStartPos, const int *pnSpareInsFeaId,
							   const int *pNumofFeaValue, float_point *pdDenseIns, const int *pSortedUsedFea,
							   const int *pHashFeaIdToDenseInsPos, int numofUsedFea,
						  	   int startInsId, int numofInsToFill)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofInsToFill)
		return;

	int insId = startInsId + nGlobalThreadId;
	long long startPos = pInsStartPos[insId];
	int numofFeaValue = pNumofFeaValue[insId];
	int denseInsStartPos = nGlobalThreadId * numofUsedFea;

	//for each value in the sparse instance
	int curDenseTop = 0;
	for(int i = 0; i < numofFeaValue; i++)
	{
		int feaId = pnSpareInsFeaId[startPos + i];

		while(feaId > pSortedUsedFea[curDenseTop])
		{
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], numofUsedFea);
			pdDenseIns[denseInsStartPos + pos] = 0;
			curDenseTop++;
		}

		if(feaId == pSortedUsedFea[curDenseTop])
		{//this is a feature needed to be stored in dense instance
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], numofUsedFea);
			pdDenseIns[denseInsStartPos + pos] = pdSparseInsValue[startPos + i];
			curDenseTop++;
		}
	}

}

