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
#include "Bagging/BagManager.h"
#include "../DeviceHost/TreeNode.h"
#include "../DeviceHost/SparsePred/DenseInstance.h"
#include "../SharedUtility/DataType.h"
#include "../SharedUtility/CudaMacro.h"
#include "../SharedUtility/KernelConf.h"

/**
 * @brief: prediction function for sparse instances
 */
void DevicePredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree,
									   vector<real> &v_fPredValue, void *pStream, int bagId)
{
	BagManager bagManager;
	GBDTGPUMemManager manager;
	DenseInsConverter denseInsConverter(vTree);
	int numofUsedFea = denseInsConverter.usedFeaSet.size();

	if(manager.m_maxUsedFeaInATree < numofUsedFea)
	{
		cout << "numofUsedFea=" << numofUsedFea << " v.s. maxUsedFeaInTrees " << manager.m_maxUsedFeaInATree << endl;
		exit(0);
	}

	//build the hash table for feature id and position id
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	GetUsedFeature(denseInsConverter.usedFeaSet, pHashUsedFea, pSortedUsedFea, pStream, bagId);
	if(pHashUsedFea != NULL)
		delete []pHashUsedFea;
	if(pSortedUsedFea != NULL)
		delete []pSortedUsedFea;

	//for each tree
	int nNumofIns = manager.m_numofIns;
	int nNumofTree = manager.m_pNumofTreeLearntEachBag_h[bagId];
	PROCESS_ERROR(nNumofTree > 0);

	//start prediction
	real *pPredictedValue;
	checkCudaErrors(cudaMalloc((void**)&pPredictedValue, sizeof(real) * nNumofIns));
	manager.MemsetAsync(pPredictedValue, 0, sizeof(real) * nNumofIns, pStream);

	uint startPos = 0;
	int startInsId = 0;
	uint *pInsStartPos = manager.m_pInsStartPos + startInsId;
	manager.MemcpyDeviceToHostAsync(pInsStartPos, &startPos, sizeof(uint), pStream);
//	cout << "start pos ins" << insId << "=" << startPos << endl;
	real *pDevInsValue = manager.m_pdDInsValue + startPos;
	int *pDevFeaId = manager.m_pDFeaId + startPos;
	int *pNumofFea = manager.m_pDNumofFea + startInsId;
	int numofInsToFill = nNumofIns;

	KernelConf conf;
	int threadPerBlock;
	dim3 dimNumofBlock;
	conf.ConfKernel(numofInsToFill, threadPerBlock, dimNumofBlock);
	cudaDeviceSynchronize();

	if(numofUsedFea > 0){
		//memset dense instances
		real *pTempDense = manager.m_pdDenseInsEachBag + bagId * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns;
		checkCudaErrors(cudaMemset(pTempDense, -1, sizeof(real) * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns));
		FillMultiDense<<<dimNumofBlock, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(
											  pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea,
											  pTempDense,
											  manager.m_pSortedUsedFeaIdBag + bagId * bagManager.m_maxNumUsedFeaATree,
										  	  manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
											  numofUsedFea, startInsId, numofInsToFill);
	}
	cudaDeviceSynchronize();
	GETERROR("after FillMultiDense");

	//prediction using the last tree
	for(int t = 0; t < nNumofTree; t++){
		int numofNodeOfTheTree = 0;
		TreeNode *pTree = NULL;

		int treeId = t;
		GetTreeInfo(pTree, numofNodeOfTheTree, treeId, pStream, bagId);
		PROCESS_ERROR(pTree != NULL);
		PredMultiTarget<<<dimNumofBlock, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(
													pPredictedValue, numofInsToFill, pTree,
													manager.m_pdDenseInsEachBag + bagId * bagManager.m_numIns * bagManager.m_maxNumUsedFeaATree, numofUsedFea,
													manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree, bagManager.m_maxTreeDepth);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
	}

	real *pTempTarget = new real[nNumofIns];
	manager.MemcpyDeviceToHostAsync(pPredictedValue,
									pTempTarget, sizeof(real) * nNumofIns, pStream);

	for(int i = 0; i < nNumofIns; i++){
		v_fPredValue.push_back(pTempTarget[i]);
	}
	delete []pTempTarget;
	checkCudaErrors(cudaFree(pPredictedValue));
}

/**
 * @brief: get the feature value.
 */
void DevicePredictor::GetUsedFeature(vector<int> &v_usedFeaSortedId, int *&pHashUsedFea, int *&pSortedUsedFea, void *pStream, int bagId)
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
		//cout << "hash value of " << v_usedFeaSortedId[uf] << " is " << hashValue << "; " << pHashUsedFea[v_usedFeaSortedId[uf] % numofUsedFea] << endl;
	}

	pSortedUsedFea = new int[numofUsedFea];
	for(int uf = 0; uf < numofUsedFea; uf++)
		pSortedUsedFea[uf] = v_usedFeaSortedId[uf];

	//copy hash map to gpu memory
	GBDTGPUMemManager manager;
	BagManager bagManager;
	memset(manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree, -1,
									sizeof(int) * manager.m_maxUsedFeaInATree);
	checkCudaErrors(cudaMemsetAsync(manager.m_pSortedUsedFeaIdBag + bagId * bagManager.m_maxNumUsedFeaATree, -1,
									sizeof(int) * manager.m_maxUsedFeaInATree, (*(cudaStream_t*)pStream)));

	manager.MemcpyHostToDeviceAsync(pHashUsedFea, manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
									sizeof(int) * numofUsedFea, pStream);
	manager.MemcpyHostToDeviceAsync(pSortedUsedFea, manager.m_pSortedUsedFeaIdBag + bagId * bagManager.m_maxNumUsedFeaATree,
									sizeof(int) * numofUsedFea, pStream);
}

/**
 * @brief: get the pointer to the tree and its number of nodes
 */
void DevicePredictor::GetTreeInfo(TreeNode *&pTree, int &numofNodeOfTheTree, int treeId, void *pStream, int bagId)
{
	if(treeId < 0)
		return;
	GBDTGPUMemManager manager;
	BagManager bagManager;
	manager.MemcpyDeviceToHostAsync(manager.m_pNumofNodeEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + treeId,
								&numofNodeOfTheTree, sizeof(int), pStream);
	int startPosOfLastTree = -1;
	manager.MemcpyDeviceToHostAsync(manager.m_pStartPosOfEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + treeId,
								&startPosOfLastTree, sizeof(int), pStream);
	pTree = manager.m_pAllTreeEachBag + startPosOfLastTree;
}
