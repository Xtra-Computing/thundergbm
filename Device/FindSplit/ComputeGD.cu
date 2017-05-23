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

#include "FindFeaKernel.h"
#include "../DevicePredictor.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/Initiator.h"
#include "../Memory/dtMemManager.h"
#include "../DevicePredictorHelper.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/SparsePred/DenseInstance.h"
#include "../../SharedUtility/powerOfTwo.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/CudaMacro.h"

using std::cerr;
using std::endl;

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
	PROCESS_ERROR(numofUsedFea <= manager.m_maxUsedFeaInTrees);
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
	int numofTreeLearnt = bagManager.m_pNumofTreeLearntEachBag_h[bagId];
	int treeId = numofTreeLearnt - 1;
	pred.GetTreeInfo(pLastTree, numofNodeOfLastTree, treeId, pStream, bagId);

	KernelConf conf;
	//start prediction
	checkCudaErrors(cudaMemsetAsync(bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns, 0,
									sizeof(real) * nNumofIns, (*(cudaStream_t*)pStream)));
	if(nNumofTree > 0 && numofUsedFea >0)//numofUsedFea > 0 means the tree has more than one node.
	{
		long long startPos = 0;
		int startInsId = 0;
		long long *pInsStartPos = manager.m_pInsStartPos + startInsId;
		manager.MemcpyDeviceToHostAsync(pInsStartPos, &startPos, sizeof(long long), pStream);
	//			cout << "start pos ins" << insId << "=" << startPos << endl;
		real *pDevInsValue = manager.m_pdDInsValue + startPos;
		int *pDevFeaId = manager.m_pDFeaId + startPos;
		int *pNumofFea = manager.m_pDNumofFea + startInsId;
		int numofInsToFill = nNumofIns;

		dim3 dimGridThreadForEachIns;
		conf.ComputeBlock(numofInsToFill, dimGridThreadForEachIns);
		int sharedMemSizeEachIns = 1;

		FillMultiDense<<<dimGridThreadForEachIns, sharedMemSizeEachIns, 0, (*(cudaStream_t*)pStream)>>>(
											  pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea,
										  	  bagManager.m_pdDenseInsEachBag + bagId * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns,
										  	  bagManager.m_pSortedUsedFeaIdBag + bagId * bagManager.m_maxNumUsedFeaATree,
										  	  bagManager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
											  numofUsedFea, startInsId, numofInsToFill);
		GETERROR("after FillMultiDense");
	}

	//prediction using the last tree
	if(nNumofTree > 0)
	{
		assert(pLastTree != NULL);
		int numofInsToPre = nNumofIns;
		dim3 dimGridThreadForEachIns;
		conf.ComputeBlock(numofInsToPre, dimGridThreadForEachIns);
		int sharedMemSizeEachIns = 1;
		PredMultiTarget<<<dimGridThreadForEachIns, sharedMemSizeEachIns, 0, (*(cudaStream_t*)pStream)>>>(
											  bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns,
										  	  numofInsToPre, pLastTree,
										  	  bagManager.m_pdDenseInsEachBag + bagId * bagManager.m_maxNumUsedFeaATree * bagManager.m_numIns,
											  numofUsedFea, bagManager.m_pHashFeaIdToDenseInsPosBag + bagId * bagManager.m_maxNumUsedFeaATree,
										  	  bagManager.m_maxTreeDepth);
		GETERROR("after PredMultiTarget");

		//save to buffer
		int threadPerBlock;
		dim3 dimGridThread;
		conf.ConfKernel(nNumofIns, threadPerBlock, dimGridThread);
		SaveToPredBuffer<<<dimGridThread, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns,
																nNumofIns, bagManager.m_pPredBufferEachBag + bagId * bagManager.m_numIns);
														 //(manager.m_pTargetValue, nNumofIns, manager.m_pPredBuffer);
		//update the final prediction
		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pPredBufferEachBag + bagId * bagManager.m_numIns,
									 bagManager.m_pTargetValueEachBag + bagId * bagManager.m_numIns, sizeof(real) * nNumofIns, pStream);
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
	cudaFree(pdTempGD);

	real *pTempHess = bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns;
	real hessSum = thrust::reduce(thrust::system::cuda::par, pTempHess, pTempHess + bagManager.m_numIns);
	printf("sum_gd=%f, sum_hess=%f\n", gdSum, hessSum);

	//copy splittable nodes to GPU memory
	//SNodeStat, SNIdToBuffId, pBuffIdVec need to be reset.
	manager.MemsetAsync(bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable, 0,
						sizeof(nodeStat) * bagManager.m_maxNumSplittable, pStream);
	manager.MemsetAsync(bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable, -1, sizeof(int) * bagManager.m_maxNumSplittable, pStream);
	manager.MemsetAsync(bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, -1, sizeof(int) * bagManager.m_maxNumSplittable, pStream);
	manager.MemsetAsync(bagManager.m_pNumofBuffIdEachBag + bagId, 0, sizeof(int), pStream);

	InitNodeStat<<<1, 1, 0, (*(cudaStream_t*)pStream)>>>(gdSum, hessSum,
						   bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
						   bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable, manager.m_maxNumofSplittable,
						   bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
						   bagManager.m_pNumofBuffIdEachBag + bagId);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after InitNodeStat");
}

