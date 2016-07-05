/*
 * DevicePrediction.cu
 *
 *  Created on: 23 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "Hashing.h"
#include "KernelConf.h"
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

	if(manager.m_maxUsedFeaInTrees < numofUsedFea)
	{
		cout << "numofUsedFea=" << numofUsedFea << " v.s. maxUsedFeaInTrees " << manager.m_maxUsedFeaInTrees << endl;
		exit(0);
	}

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

	long long startPos = 0;
	int startInsId = 0;
	long long *pInsStartPos = manager.m_pInsStartPos + startInsId;
	manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
//			cout << "start pos ins" << insId << "=" << startPos << endl;
	float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
	int *pDevFeaId = manager.m_pDFeaId + startPos;
	int *pNumofFea = manager.m_pDNumofFea + startInsId;
	int numofInsToFill = nNumofIns;
	KernelConf conf;
	int threadPerBlock;
	dim3 dimNumofBlock;
	conf.ConfKernel(numofInsToFill, threadPerBlock, dimNumofBlock);

	FillMultiDense<<<dimNumofBlock, threadPerBlock>>>(
										  pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea, manager.m_pdDenseIns,
										  manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos,
										  numofUsedFea, startInsId, numofInsToFill);

#if testing
		if(cudaGetLastError() != cudaSuccess)
		{
			cout << "error in FillMultiDense" << endl;
			exit(0);
		}
#endif


//		FillDenseIns(i, numofUsedFea);
		//prediction using the last tree
		for(int t = 0; t < nNumofTree; t++)
		{
			int numofNodeOfTheTree = 0;
			TreeNode *pTree = NULL;

			int treeId = t;
			GetTreeInfo(pTree, numofNodeOfTheTree, treeId);
			PROCESS_ERROR(pTree != NULL);
			PredMultiTarget<<<dimNumofBlock, threadPerBlock>>>(
														manager.m_pTargetValue, numofInsToFill, pTree,
														manager.m_pdDenseIns, numofUsedFea,
														manager.m_pHashFeaIdToDenseInsPos, treeManager.m_maxTreeDepth);
			cudaDeviceSynchronize();
		}

	for(int i = 0; i < nNumofIns; i++)
	{
		float_point fTarget = 0;
		manager.MemcpyDeviceToHost(manager.m_pTargetValue + i, &fTarget, sizeof(float_point));

		v_fPredValue.push_back(fTarget);
	}
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
