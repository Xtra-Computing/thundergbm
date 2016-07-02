/*
 * DevicePrediction.cu
 *
 *  Created on: 23 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "Hashing.h"
#include "DevicePredictor.h"
#include "DevicePredictorHelper.h"
#include "Memory/gbdtGPUMemManager.h"
#include "Memory/dtMemManager.h"
#include "../DeviceHost/DefineConst.h"
#include "../DeviceHost/TreeNode.h"
#include "../DeviceHost/SparsePred/DenseInstance.h"
#include "../DeviceHost/MyAssert.h"


/**
 * @brief: prediction function for sparse instances
 */
void DevicePredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue)
{
	GBDTGPUMemManager manager;
	DTGPUMemManager treeManager;
	DenseInsConverter denseInsConverter(vTree);
	int numofUsedFea = denseInsConverter.usedFeaSet.size();

	//build the hash table for feature id and position id
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	GetUsedFeature(denseInsConverter.usedFeaSet, pHashUsedFea, pSortedUsedFea);

	//for each tree
	int nNumofIns = v_vInstance.size();
	int nNumofTree = treeManager.m_numofTreeLearnt;
	PROCESS_ERROR(treeManager.m_numofTree == treeManager.m_numofTreeLearnt);
	PROCESS_ERROR(nNumofTree > 0);

	//start prediction
	checkCudaErrors(cudaMemset(manager.m_pTargetValue, 0, sizeof(float_point) * nNumofIns));
	for(int i = 0; i < nNumofIns; i++)
	{
		FillDenseIns(i, numofUsedFea);

		//prediction using the last tree
		for(int t = 0; t < nNumofTree; t++)
		{
			int numofNodeOfTheTree = 0;
			TreeNode *pTree = NULL;

			int treeId = t;
			GetTreeInfo(pTree, numofNodeOfTheTree, treeId);
			PROCESS_ERROR(pTree != NULL);
			PredTarget<<<1, 1>>>(pTree, numofNodeOfTheTree, manager.m_pdDenseIns, numofUsedFea,
								 manager.m_pHashFeaIdToDenseInsPos, manager.m_pTargetValue + i, treeManager.m_maxTreeDepth);
			cudaDeviceSynchronize();
		}

		float_point fTarget = 0;
		manager.MemcpyDeviceToHost(manager.m_pTargetValue + i, &fTarget, sizeof(float_point));

		#ifdef _COMPARE_HOST
		vector<double> vDense;
		double fValue = 0;
		denseInsConverter.SparseToDense(v_vInstance[i], vDense);
		//host prediction
		for(int t = 0; t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIdSparseInstance(vDense, denseInsConverter.fidToDensePos);
			fValue += vTree[t][nodeId]->predValue;
		}

		if(fValue != fTarget)
			cout << "Target value diff " << fValue << " v.s. " << fTarget << endl;
		#endif


		v_fPredValue.push_back(fTarget);
	}

	#ifdef _COMPARE_HOST
	PROCESS_ERROR(v_fPredValue.size() == v_vInstance.size());
	#endif
}

/**
 * @brief: get the feature value.
 */
void DevicePredictor::GetUsedFeature(vector<int> &v_usedFeaSortedId, int *&pHashUsedFea, int *&pSortedUsedFea)
{
	int numofUsedFea = v_usedFeaSortedId.size();
	if(numofUsedFea == 0)
		return;

	pHashUsedFea = new int[numofUsedFea];
	memset(pHashUsedFea, -1, sizeof(int) * numofUsedFea);
	for(int uf = 0; uf < numofUsedFea; uf++)
	{
		bool bIsNewHashValue = false;
		int hashValue = Hashing::HostAssignHashValue(pHashUsedFea, v_usedFeaSortedId[uf], numofUsedFea, bIsNewHashValue);
//			cout << "hash value of " << denseInsConverter.usedFeaSet[uf] << " is " << hashValue << endl;
	}

	pSortedUsedFea = new int[numofUsedFea];
	for(int uf = 0; uf < numofUsedFea; uf++)
		pSortedUsedFea[uf] = v_usedFeaSortedId[uf];

	//copy hash map to gpu memory
	GBDTGPUMemManager manager;
	checkCudaErrors(cudaMemset(manager.m_pHashFeaIdToDenseInsPos, -1, sizeof(int) * manager.m_maxUsedFeaInTrees));
	checkCudaErrors(cudaMemset(manager.m_pSortedUsedFeaId, -1, sizeof(int) * manager.m_maxUsedFeaInTrees));

	manager.MemcpyHostToDevice(pHashUsedFea, manager.m_pHashFeaIdToDenseInsPos, sizeof(int) * numofUsedFea);
	manager.MemcpyHostToDevice(pSortedUsedFea, manager.m_pSortedUsedFeaId, sizeof(int) * numofUsedFea);

}

/**
 * @brief: get the pointer to the tree and its number of nodes
 */
void DevicePredictor::GetTreeInfo(TreeNode *&pTree, int &numofNodeOfTheTree, int treeId)
{
	if(treeId < 0)
		return;
	DTGPUMemManager treeManager;
	GBDTGPUMemManager manager;
	manager.MemcpyDeviceToHost(treeManager.m_pNumofNodeEachTree + treeId, &numofNodeOfTheTree, sizeof(int));
	int startPosOfLastTree = -1;
	manager.MemcpyDeviceToHost(treeManager.m_pStartPosOfEachTree + treeId, &startPosOfLastTree, sizeof(int));
	pTree = treeManager.m_pAllTree + startPosOfLastTree;

}

/**
 * @brief: construct a dense instance
 */
void DevicePredictor::FillDenseIns(int insId, int numofUsedFea)
{
	GBDTGPUMemManager manager;
	long long startPos = -1;
	long long *pInsStartPos = manager.m_pInsStartPos + (long long)insId;
	manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
//			cout << "start pos ins" << insId << "=" << startPos << endl;
	float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
	int *pDevFeaId = manager.m_pDFeaId + startPos;
	int numofFeaValue = -1;
	int *pNumofFea = manager.m_pDNumofFea + insId;
	manager.MemcpyDeviceToHost(pNumofFea, &numofFeaValue, sizeof(int));

	checkCudaErrors(cudaMemset(manager.m_pdDenseIns, 0, sizeof(float_point) * numofUsedFea));
	FillDense<<<1, 1>>>(pDevInsValue, pDevFeaId, numofFeaValue, manager.m_pdDenseIns,
						manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos, numofUsedFea);
}

void FillMultiDenseIns(int insStartId, int numofInsToFill, int numofUsedFea)
{
	GBDTGPUMemManager manager;
	long long startPos = -1;
	long long *pInsStartPos = manager.m_pInsStartPos + (long long)insStartId;
	manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
//			cout << "start pos ins" << insId << "=" << startPos << endl;
	float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
	int *pDevFeaId = manager.m_pDFeaId + startPos;
	int numofFeaValue = -1;
	int *pNumofFea = manager.m_pDNumofFea + insStartId;
	manager.MemcpyDeviceToHost(pNumofFea, &numofFeaValue, sizeof(int));

	checkCudaErrors(cudaMemset(manager.m_pdDenseIns, 0, sizeof(float_point) * numofUsedFea));
	FillDense<<<1, 1>>>(pDevInsValue, pDevFeaId, numofFeaValue, manager.m_pdDenseIns,
						manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos, numofUsedFea);
}

