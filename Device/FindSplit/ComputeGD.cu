/*
 * ComputeGD.cu
 *
 *  Created on: 9 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "FindFeaKernel.h"
#include "../KernelConf.h"
#include "../DevicePredictor.h"
#include "../DevicePredictorHelper.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Splitter/Initiator.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/dtMemManager.h"
#include "../../DeviceHost/SparsePred/DenseInstance.h"
#include "../prefix-sum/partialSum.h"
#include "../prefix-sum/powerOfTwo.h"
#include "../Bagging/BagManager.h"

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
	PROCESS_ERROR(numofUsedFea <= manager.m_maxUsedFeaInTrees);
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	pred.GetUsedFeature(denseInsConverter.usedFeaSet, pHashUsedFea, pSortedUsedFea, bagId);

	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = manager.m_numofIns;
	PROCESS_ERROR(nNumofIns > 0);

	//the last learned tree
	int numofNodeOfLastTree = 0;
	TreeNode *pLastTree = NULL;
	DTGPUMemManager treeManager;
	//int numofTreeLearnt = treeManager.m_numofTreeLearnt;
	int numofTreeLearnt = bagManager.m_pNumofTreeLearntEachBag_h[bagId];
	int treeId = numofTreeLearnt - 1;
	pred.GetTreeInfo(pLastTree, numofNodeOfLastTree, treeId, pStream, bagId);

	KernelConf conf;
	//start prediction
	cudaStream_t *tempStream = (cudaStream_t*)pStream;
//	checkCudaErrors(cudaMemsetAsync(manager.m_pTargetValue, 0, sizeof(float_point) * nNumofIns, *tempStream));
	checkCudaErrors(cudaMemsetAsync(bagManager.m_pTargetValueEachBag + bagId, 0, sizeof(float_point) * nNumofIns, *tempStream));
	if(nNumofTree > 0 && numofUsedFea >0)//numofUsedFea > 0 means the tree has more than one node.
	{
		long long startPos = 0;
		int startInsId = 0;
		long long *pInsStartPos = manager.m_pInsStartPos + startInsId;
		manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
	//			cout << "start pos ins" << insId << "=" << startPos << endl;
		float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
		int *pDevFeaId = manager.m_pDFeaId + startPos;
		int *pNumofFea = manager.m_pDNumofFea + startInsId;
		int numofInsToFill = nNumofIns;

		dim3 dimGridThreadForEachIns;
		conf.ComputeBlock(numofInsToFill, dimGridThreadForEachIns);
		int sharedMemSizeEachIns = 1;

		FillMultiDense<<<dimGridThreadForEachIns, sharedMemSizeEachIns>>>(
											  pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea, bagManager.m_pdDenseInsEachBag + bagId,
											  bagManager.m_pSortedUsedFeaIdBag + bagId, bagManager.m_pHashFeaIdToDenseInsPosBag + bagId,
											  //pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea, manager.m_pdDenseIns,
											  //manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos,
											  numofUsedFea, startInsId, numofInsToFill);
#if testing
			if(cudaGetLastError() != cudaSuccess)
			{
				cout << "error in FillMultiDense" << endl;
				exit(0);
			}
#endif
	}

	//prediction using the last tree
	if(nNumofTree > 0)
	{
		assert(pLastTree != NULL);
		int numofInsToPre = nNumofIns;
		dim3 dimGridThreadForEachIns;
		conf.ComputeBlock(numofInsToPre, dimGridThreadForEachIns);
		int sharedMemSizeEachIns = 1;
		PredMultiTarget<<<dimGridThreadForEachIns, sharedMemSizeEachIns>>>(
											  bagManager.m_pTargetValueEachBag + bagId, numofInsToPre, pLastTree, bagManager.m_pdDenseInsEachBag + bagId,
											  numofUsedFea, bagManager.m_pHashFeaIdToDenseInsPosBag + bagId, treeManager.m_maxTreeDepth);
											  //manager.m_pTargetValue, numofInsToPre, pLastTree, manager.m_pdDenseIns,
											  //numofUsedFea, manager.m_pHashFeaIdToDenseInsPos, treeManager.m_maxTreeDepth);
#if testing
		if(cudaGetLastError() != cudaSuccess)
		{
			cout << "error in PredTarget" << endl;
			exit(0);
		}
#endif
		//save to buffer
		int threadPerBlock;
		dim3 dimGridThread;
		conf.ConfKernel(nNumofIns, threadPerBlock, dimGridThread);
		SaveToPredBuffer<<<dimGridThread, threadPerBlock>>>(bagManager.m_pTargetValueEachBag + bagId, nNumofIns, bagManager.m_pPredBufferEachBag + bagId);
														 //(manager.m_pTargetValue, nNumofIns, manager.m_pPredBuffer);
		//update the final prediction
		manager.MemcpyDeviceToDevice(bagManager.m_pPredBufferEachBag + bagId, bagManager.m_pTargetValueEachBag + bagId, sizeof(float_point) * nNumofIns);
		//manager.MemcpyDeviceToDevice(manager.m_pPredBuffer, manager.m_pTargetValue, sizeof(float_point) * nNumofIns);
	}

	if(pHashUsedFea != NULL)
		delete []pHashUsedFea;
	if(pSortedUsedFea != NULL)
		delete []pSortedUsedFea;

	//compute GD
	int blockSizeCompGD;
	dim3 dimNumBlockComGD;
	conf.ConfKernel(nNumofIns, blockSizeCompGD, dimNumBlockComGD);
	ComputeGDKernel<<<dimNumBlockComGD, blockSizeCompGD>>>(
								//nNumofIns, manager.m_pTargetValue, manager.m_pdTrueTargetValue,
								nNumofIns, bagManager.m_pTargetValueEachBag + bagId, bagManager.m_pdTrueTargetValueEachBag + bagId,
								bagManager.m_pInsGradEachBag + bagId, bagManager.m_pInsHessEachBag + bagId);
								//manager.m_pGrad, manager.m_pHess);

	//copy splittable nodes to GPU memory
		//SNodeStat, SNIdToBuffId, pBuffIdVec need to be reset.
	//manager.Memset(manager.m_pSNodeStat, 0, sizeof(nodeStat) * manager.m_maxNumofSplittable);
	manager.Memset(bagManager.m_pSNodeStatEachBag + bagId, 0, sizeof(nodeStat) * manager.m_maxNumofSplittable);
	//manager.Memset(manager.m_pSNIdToBuffId, -1, sizeof(int) * manager.m_maxNumofSplittable);
	manager.Memset(bagManager.m_pSNIdToBuffIdEachBag + bagId, -1, sizeof(int) * manager.m_maxNumofSplittable);
	//manager.Memset(manager.m_pBuffIdVec, -1, sizeof(int) * manager.m_maxNumofSplittable);
	manager.Memset(bagManager.m_pBuffIdVecEachBag + bagId, -1, sizeof(int) * manager.m_maxNumofSplittable);
	//manager.Memset(manager.m_pNumofBuffId, 0, sizeof(int));
	manager.Memset(bagManager.m_pNumofBuffIdEachBag + bagId, 0, sizeof(int));

	//compute number of blocks
	int thdPerBlockSum;
	dim3 dimNumBlockSum;
	conf.ConfKernel(ceil(nNumofIns/2.0), thdPerBlockSum, dimNumBlockSum);//one thread adds two values
	blockSum<<<dimNumBlockSum, thdPerBlockSum, thdPerBlockSum * 2 * sizeof(float_point)>>>(
												//manager.m_pGrad, manager.m_pGDBlockSum, nNumofIns, false);
												bagManager.m_pInsGradEachBag + bagId, bagManager.m_pGDBlockSumEachBag + bagId, nNumofIns, false);
	int numBlockForBlockSum = dimNumBlockSum.x * dimNumBlockSum.y;
	int numThdFinalSum;
	dim3 dimDummy;
	conf.ConfKernel(ceil(numBlockForBlockSum/2.0), numThdFinalSum, dimDummy);//one thread adds two values
	if(numThdFinalSum >= conf.m_maxBlockSize)
		numThdFinalSum = conf.m_maxBlockSize;
	else
	{
		unsigned int tempNumThd;
		smallReductionKernelConf(tempNumThd, numBlockForBlockSum);
		numThdFinalSum = tempNumThd;
	}
#if false
	float_point *pGDBlockSum_d = new float_point[numBlockForBlockSum];
	manager.MemcpyDeviceToHost(manager.m_pGDBlockSum, pGDBlockSum_d, sizeof(float_point) * numBlockForBlockSum);
	float_point totalBlockSum = 0;
	for(int b = 0; b < numBlockForBlockSum; b++)
	{
		totalBlockSum +=pGDBlockSum_d[b];
	}
	cerr << "total gd block sum is " << totalBlockSum << endl;
#endif

	blockSum<<<1, numThdFinalSum, numThdFinalSum * 2 * sizeof(float_point)>>>(
												//manager.m_pGDBlockSum, manager.m_pGDBlockSum, numBlockForBlockSum, true);
												bagManager.m_pGDBlockSumEachBag + bagId, bagManager.m_pGDBlockSumEachBag + bagId, numBlockForBlockSum, true);


	//compute hessian block sum and final sum
	blockSum<<<dimNumBlockSum, thdPerBlockSum, thdPerBlockSum * 2 * sizeof(float_point)>>>(
												//manager.m_pHess, manager.m_pHessBlockSum, nNumofIns, false);
												bagManager.m_pInsHessEachBag + bagId, bagManager.m_pHessBlockSumEachBag + bagId, nNumofIns, false);
	blockSum<<<1, numThdFinalSum, numThdFinalSum * 2 * sizeof(float_point)>>>(
												//manager.m_pHessBlockSum, manager.m_pHessBlockSum, numBlockForBlockSum, true);
												bagManager.m_pHessBlockSumEachBag + bagId, bagManager.m_pHessBlockSumEachBag + bagId, numBlockForBlockSum, true);

#if false
	float_point *pGD_h = new float_point[nNumofIns];
	float_point *pHess_h = new float_point[nNumofIns];

	manager.MemcpyDeviceToHost(manager.m_pGrad, pGD_h, sizeof(float_point) * nNumofIns);
	manager.MemcpyDeviceToHost(manager.m_pHess, pHess_h, sizeof(float_point) * nNumofIns);

	float_point sumGD_h = 0;
	float_point sumHess_h = 0;
	for(int i = 0; i < nNumofIns; i++)
	{
		sumGD_h += pGD_h[i];
		sumHess_h += pHess_h[i];
	}

	float_point sumGD_d, sumHess_d;
	manager.MemcpyDeviceToHost(manager.m_pGDBlockSum, &sumGD_d, sizeof(float_point));
	manager.MemcpyDeviceToHost(manager.m_pHessBlockSum, &sumHess_d, sizeof(float_point));

	if(sumGD_d != sumGD_h || sumHess_d != sumHess_h)
	{
		cerr << "host and device have different results: " << sumGD_d << " v.s. " << sumGD_h << "; " << sumHess_d << " v.s. " << sumHess_h << endl;
	}

	delete []pGD_h;
	delete []pHess_h;
#endif

	InitNodeStat<<<1, 1>>>(//manager.m_pGDBlockSum, manager.m_pHessBlockSum,
						   bagManager.m_pGDBlockSumEachBag + bagId, bagManager.m_pHessBlockSumEachBag + bagId,
						   //manager.m_pSNodeStat, manager.m_pSNIdToBuffId, manager.m_maxNumofSplittable,
						   bagManager.m_pSNodeStatEachBag + bagId, bagManager.m_pSNIdToBuffIdEachBag + bagId, manager.m_maxNumofSplittable,
						   //manager.m_pBuffIdVec, manager.m_pNumofBuffId
						   bagManager.m_pBuffIdVecEachBag + bagId, bagManager.m_pNumofBuffIdEachBag + bagId);

	cudaDeviceSynchronize();
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in InitNodeStat" << endl;
		exit(0);
	}
#endif
}

