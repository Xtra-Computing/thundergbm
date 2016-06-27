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
#include "DevicePredictor.h"
#include "../DeviceHost/SparsePred/DenseInstance.h"
#include "../DeviceHost/MyAssert.h"

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

__global__ void FillDense(float_point *pdSparseInsValue, int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, int *pSortedUsedFea, int *pHashFeaIdToDenseInsPos, int totalUsedFea)
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


/**
 * @brief: prediction function for sparse instances
 */
void DevicePredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue)
{
	DenseInsConverter denseInsConverter(vTree);
	//for each tree
	int nNumofIns = v_vInstance.size();

	for(int i = 0; i < nNumofIns; i++)
	{
		//start prediction ###############
		int nNumofTree = vTree.size();
		double fValue = 0;

		vector<double> vDense;
		if(nNumofTree > 0)
			denseInsConverter.SparseToDense(v_vInstance[i], vDense);
		//prediction using the last tree
		for(int t = 0; t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIdSparseInstance(vDense, denseInsConverter.fidToDensePos);
			fValue += vTree[t][nodeId]->predValue;
		}

		//end prediction #################

		v_fPredValue.push_back(fValue);
	}

	assert(v_fPredValue.size() == v_vInstance.size());
}


