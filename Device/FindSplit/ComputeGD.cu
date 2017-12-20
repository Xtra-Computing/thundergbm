/*
 * ComputeGD.cu
 *
 *  Created on: 9 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda.h>

#include "FindFeaKernel.h"
#include "../DevicePredictor.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/Initiator.h"
#include "../DevicePredictorHelper.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/SparsePred/DenseInstance.h"
#include "../../SharedUtility/powerOfTwo.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/CudaMacro.h"

using std::cerr;
using std::endl;

__global__ void PredTargetViaTrainingResult(real *pdTargetValue, int numofIns, const TreeNode *pAllTreeNode,
											const int *pIns2Nid){
	uint gTid = GLOBAL_TID();
	if(gTid >= numofIns)
		return;
	uint nid = pIns2Nid[gTid];
	ECHECKER(pIns2Nid[gTid]);
	while(pAllTreeNode[nid].loss < -9.0)//-10.0 is the value for pruned node
		nid = pAllTreeNode[nid].parentId;
	pdTargetValue[gTid] += pAllTreeNode[nid].predValue;
}

/**
 * @brief: prediction and compute gradient descent
 */
void DeviceSplitter::ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > &vvInsSparse, void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	DevicePredictor pred;
	//get features and store the feature ids in a way that the access is efficient
	DenseInsConverter denseInsConverter(vTree);

	//hash feature id to position id
	int numofUsedFea = denseInsConverter.usedFeaSet.size();
	printf("# of used fea=%d\n", numofUsedFea);
	PROCESS_ERROR(numofUsedFea <= manager.m_maxUsedFeaInATree);
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	pred.GetUsedFeature(denseInsConverter.usedFeaSet, pHashUsedFea, pSortedUsedFea, pStream, bagId);

	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = bagManager.m_numIns;
	PROCESS_ERROR(nNumofIns > 0);

	//the last learned tree
	int numofNodeOfLastTree = 0;
	TreeNode *pLastTree = NULL;
	int numofTreeLearnt = manager.m_pNumofTreeLearntEachBag_h[bagId];
	int treeId = numofTreeLearnt - 1;
	pred.GetTreeInfo(pLastTree, numofNodeOfLastTree, treeId, pStream, bagId);

	KernelConf conf;
	//start prediction
//###################  for my future experiments
bool bOptimisePred = true;
	if(nNumofTree > 0 && numofUsedFea >0 && bOptimisePred == false)//numofUsedFea > 0 means the tree has more than one node.
	{
		uint startPos = 0;
		int startInsId = 0;
		uint *pInsStartPos = manager.m_pInsStartPos + startInsId;
		manager.MemcpyDeviceToHostAsync(pInsStartPos, &startPos, sizeof(uint), pStream);
	//			cout << "start pos ins" << insId << "=" << startPos << endl;
		real *pDevInsValue = manager.m_pdDInsValue + startPos;
		int *pDevFeaId = manager.m_pDFeaId + startPos;
		int *pNumofFea = manager.m_pDNumofFea + startInsId;
		int numofInsToFill = nNumofIns;

		//memset dense instances
		real *pTempDense = manager.m_pdDenseInsEachBag + bagId * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns;
		checkCudaErrors(cudaMemset(pTempDense, -1, sizeof(real) * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns));
		GETERROR("before FillMultiDense");
		FillMultiDense<<<numofInsToFill, 1, 0, (*(cudaStream_t*)pStream)>>>(
											  pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea,
											  pTempDense,
										  	  manager.m_pSortedUsedFeaIdBag + bagId * bagManager.m_maxNumUsedFeaATree,
										  	  manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
											  numofUsedFea, startInsId, numofInsToFill);
		GETERROR("after FillMultiDense");
	}

	//prediction using the last tree
	if(nNumofTree > 0)
	{
		assert(pLastTree != NULL);
		if(bOptimisePred == false){
			PredMultiTarget<<<nNumofIns, 1, 0, (*(cudaStream_t*)pStream)>>>(
												  bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns,
												  nNumofIns, pLastTree,
												  manager.m_pdDenseInsEachBag + bagId * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns,
												  numofUsedFea, manager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
												  bagManager.m_maxTreeDepth);
			GETERROR("after PredMultiTarget");
		}
		else{//prediction using the training results
			int threadPerBlock;
			dim3 dimGridThread;
			conf.ConfKernel(nNumofIns, threadPerBlock, dimGridThread);
			int *pTempIns2Nid = bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns;
			PredTargetViaTrainingResult<<<dimGridThread, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(
										bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns,
										nNumofIns, pLastTree, pTempIns2Nid);
		}
	}

	if(pHashUsedFea != NULL)
		delete []pHashUsedFea;
	if(pSortedUsedFea != NULL)
		delete []pSortedUsedFea;

	//compute GD
	int blockSizeCompGD;
	dim3 dimNumBlockComGD;
	conf.ConfKernel(nNumofIns, blockSizeCompGD, dimNumBlockComGD);
	ComputeGDKernel<<<dimNumBlockComGD, blockSizeCompGD, 0, (*(cudaStream_t*)pStream)>>>(
								nNumofIns, bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns,
								bagManager.m_pdTrueTargetValueEachBag + bagId * bagManager.m_numIns,
								bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns, bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns);

	//for gd and hess sum
	real *pTempGD = bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns;
	vector<real> hostGD(bagManager.m_numIns);
	cudaMemcpy(hostGD.data(), pTempGD, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost);
	vector<double> dHostGD(hostGD.begin(), hostGD.end());
	double *pdTempGD;
	cudaMalloc((void**)&pdTempGD, sizeof(double) * bagManager.m_numIns);
	cudaMemcpy(pdTempGD, dHostGD.data(), sizeof(double) * bagManager.m_numIns, cudaMemcpyHostToDevice);
	real gdSum = thrust::reduce(thrust::system::cuda::par, pdTempGD, pdTempGD + bagManager.m_numIns);
	cudaDeviceSynchronize();
	cudaFree(pdTempGD);

	real *pTempHess = bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns;
	real hessSum = thrust::reduce(thrust::system::cuda::par, pTempHess, pTempHess + bagManager.m_numIns);

	//copy splittable nodes to GPU memory
	//SNodeStat, SNIdToBuffId, pBuffIdVec need to be reset.
	manager.MemsetAsync(bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable, 0,
						sizeof(nodeStat) * bagManager.m_maxNumSplittable, pStream);
	manager.MemsetAsync(bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, -1, sizeof(int) * bagManager.m_maxNumSplittable, pStream);

	InitNodeStat<<<1, 1, 0, (*(cudaStream_t*)pStream)>>>(gdSum, hessSum,
						   bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
						   bagManager.m_maxNumSplittable,
						   bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after InitNodeStat");
}

