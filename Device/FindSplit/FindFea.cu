/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "FindFeaKernel.h"
#include "../KernelConf.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"

using std::cout;
using std::endl;
using std::make_pair;


/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	GBDTGPUMemManager manager;
	int numofSNode = manager.m_curNumofSplitable;
#if testing
	int tempSN = 0;
	manager.MemcpyDeviceToHost(manager.m_pNumofBuffId, &tempSN, sizeof(int));
	PROCESS_ERROR(numofSNode == tempSN);
#endif

	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	int maxNumofSplittable = manager.m_maxNumofSplittable;

	//set memory
	int numofElement = nNumofFeature * manager.m_maxNumofSplittable;
	checkCudaErrors(cudaMemset(manager.m_pTempRChildStatPerThread, 0, sizeof(nodeStat) * numofElement));
	manager.MemcpyHostToDevice(manager.m_pBestPointHostPerThread, manager.m_pBestSplitPointPerThread, sizeof(SplitPoint) * numofElement);
		//optional memory set
	checkCudaErrors(cudaMemset(manager.m_pRChildStatPerThread, 0, sizeof(nodeStat) * numofElement));
	checkCudaErrors(cudaMemset(manager.m_pLChildStatPerThread, 0, sizeof(nodeStat) * numofElement));
	checkCudaErrors(cudaMemset(manager.m_pLastValuePerThread, -1, sizeof(float_point) * numofElement));

	//######## testing code
	int smallestFeaId = 0;
	int feaBatch = nNumofFeature;
	int maxNumofValuePerFea = manager.m_numofIns;
	float_point *pGDOnEachFeaValue_d, *pHessOnEachFeaValue_d, *pValueOneEachFeaValue_d;
	checkCudaErrors(cudaMalloc((void**)&pGDOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pHessOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pValueOneEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));

	checkCudaErrors(cudaMemset(pGDOnEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMemset(pHessOnEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMemset(pValueOneEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));

	int maxNumofFeaValuePerFeaId = manager.m_numofIns;

	int blockSizeFillGD;
	dim3 dimNumofBlockToFillGD;
	KernelConf conf;
	conf.ConfKernel(maxNumofFeaValuePerFeaId, blockSizeFillGD, dimNumofBlockToFillGD);
	PROCESS_ERROR(dimNumofBlockToFillGD.y == 1 && dimNumofBlockToFillGD.z == 1);
	int numofBlockFillGD = dimNumofBlockToFillGD.x;
	dim3 dimGrid(numofBlockFillGD, feaBatch, numofSNode);
	dim3 dimBlock(blockSizeFillGD, 1, 1);
	//get gd for every splittable node
	ObtainGDEachNode<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pDInsId,
											manager.m_pdDFeaValue, manager.m_pInsIdToNodeId,
											manager.m_pGrad, manager.m_pHess,
											numofSNode, smallestFeaId, nNumofFeature, feaBatch,
											pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pValueOneEachFeaValue_d);
	cudaDeviceSynchronize();

#if testing
	float_point *pGDOnEachFeaVaue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	float_point *pHessOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGDOnEachFeaValue_d, pGDOnEachFeaVaue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(pHessOnEachFeaValue_d, pHessOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ObtainGDEachNode" << endl;
		exit(0);
	}
	delete []pGDOnEachFeaVaue_h;
	delete []pHessOnEachFeaValue_h;
#endif

	int *pStartPosEachFeaInBatch_d;
	int *pFeaLenInBatch_d;
	checkCudaErrors(cudaMalloc((void**)&pStartPosEachFeaInBatch_d, sizeof(int) * feaBatch));
	checkCudaErrors(cudaMalloc((void**)&pFeaLenInBatch_d, sizeof(int) * feaBatch));
	int blockSizePosEachFeaInBatch;
	dim3 dimNumofBlockFindPosEachFeaInBatch;
	conf.ConfKernel(feaBatch, blockSizePosEachFeaInBatch, dimNumofBlockFindPosEachFeaInBatch);
	PROCESS_ERROR(dimNumofBlockFindPosEachFeaInBatch.z == 1 && dimNumofBlockFindPosEachFeaInBatch.y == 1);
	GetInfoEachFeaInBatch<<<dimNumofBlockFindPosEachFeaInBatch, blockSizePosEachFeaInBatch>>>(
												manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, smallestFeaId, nNumofFeature,
											    feaBatch, pStartPosEachFeaInBatch_d, pFeaLenInBatch_d);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in GetInfoEachFeaInBatch" << endl;
		exit(0);
	}
#endif
	//compute prefix sum
	int *pnEachFeaLen = new int[feaBatch];
	manager.MemcpyDeviceToHost(pFeaLenInBatch_d, pnEachFeaLen, sizeof(int) * feaBatch);
	PrefixSumForEachNode(feaBatch, pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pStartPosEachFeaInBatch_d, pnEachFeaLen);
#if testing
	float_point *pGDScan_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	float_point *pHessScan_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGDOnEachFeaValue_d, pGDScan_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(pHessOnEachFeaValue_d, pHessScan_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PrefixSumForEachNode" << endl;
		exit(0);
	}
	delete[] pGDScan_h;
	delete[] pHessScan_h;
#endif

	//compute gain
	float_point *pGainOnEachFeaValue_d;
	checkCudaErrors(cudaMalloc((void**)&pGainOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	ComputeGain<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pSNodeStat, smallestFeaId, feaBatch,
									   manager.m_pBuffIdVec, numofSNode, DeviceSplitter::m_lambda, pGDOnEachFeaValue_d,
									   pHessOnEachFeaValue_d, pGainOnEachFeaValue_d);
#if testing
	float_point *pGainOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGainOnEachFeaValue_d, pGainOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ComputeGain" << endl;
		exit(0);
	}
	delete []pGainOnEachFeaValue_h;
#endif

	//find the local best split in this batch of features
	float_point *pfBestGain_d;
	int *pnBestGainKey_d;
	int nBlockEachFea = dimGrid.x;
	int nElePerBlock = dimBlock.x;
	checkCudaErrors(cudaMalloc((void**)&pnBestGainKey_d, sizeof(int) * feaBatch * nBlockEachFea));
	checkCudaErrors(cudaMalloc((void**)&pfBestGain_d, sizeof(float_point) * feaBatch * nBlockEachFea));
	PickLocalBestSplit<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, pGainOnEachFeaValue_d,
											  manager.m_pBuffIdVec, smallestFeaId, feaBatch,
											  numofSNode, maxNumofSplittable, pfBestGain_d, pnBestGainKey_d);

	checkCudaErrors(cudaFree(pGDOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pHessOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pValueOneEachFeaValue_d));
	checkCudaErrors(cudaFree(pStartPosEachFeaInBatch_d));
	checkCudaErrors(cudaFree(pFeaLenInBatch_d));
	checkCudaErrors(cudaFree(pGainOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pfBestGain_d));
	checkCudaErrors(cudaFree(pnBestGainKey_d));
	delete[] pnEachFeaLen;

	//####### end testing


	int threadPerBlock;
	dim3 dimNumofBlock;
	conf.ConfKernel(nNumofFeature, threadPerBlock, dimNumofBlock);

	clock_t begin_per_fea, begin_best;
	clock_t end_per_fea, end_best;
	cudaDeviceSynchronize();
	begin_per_fea = clock();
	FindFeaSplitValue<<<dimNumofBlock, threadPerBlock>>>(
									  manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pDInsId,
									  manager.m_pdDFeaValue, manager.m_pInsIdToNodeId,
									  manager.m_pGrad, manager.m_pHess,
									  manager.m_pTempRChildStatPerThread, manager.m_pLastValuePerThread,
									  manager.m_pSNodeStat, manager.m_pBestSplitPointPerThread,
									  manager.m_pRChildStatPerThread, manager.m_pLChildStatPerThread,
									  manager.m_pSNIdToBuffId, maxNumofSplittable, manager.m_pBuffIdVec, numofSNode,
									  DeviceSplitter::m_lambda, nNumofFeature);
	cudaDeviceSynchronize();
	end_per_fea = clock();
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in FindFeaSplitValue" << endl;
		exit(0);
	}
#endif

	float_point *pfBestGain;
	int *pnBestGainKey;
	int threadPerBlockFindBest;
	dim3 dimNumofBlockFindBest;
	conf.ConfKernel(nNumofFeature, threadPerBlockFindBest, dimNumofBlockFindBest);
	PROCESS_ERROR(dimNumofBlockFindBest.y == 1);
	dimNumofBlockFindBest.y = numofSNode;
//	cout << "numof local best block is x=" << dimNumofBlockFindBest.x << " y=" << dimNumofBlockFindBest.y << endl;
	int numofBlockLocalBest = dimNumofBlockFindBest.x * dimNumofBlockFindBest.y;
	int numofBlockPerNode = dimNumofBlockFindBest.x;
	checkCudaErrors(cudaMalloc((void**)&pfBestGain, sizeof(float_point) * maxNumofSplittable * numofBlockLocalBest));
	checkCudaErrors(cudaMalloc((void**)&pnBestGainKey, sizeof(int) * maxNumofSplittable * numofBlockLocalBest));
	PickLocalBestFea<<<dimNumofBlockFindBest, threadPerBlockFindBest>>>(
					 manager.m_pBestSplitPointPerThread, manager.m_pBuffIdVec, numofSNode, nNumofFeature,
					 maxNumofSplittable, pfBestGain, pnBestGainKey);
	cudaDeviceSynchronize();
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PickLocalBestFea" << endl;
		exit(0);
	}
#endif

	int blockSizeBestFea = numofBlockPerNode;
	if(blockSizeBestFea > conf.m_maxBlockSize)
		blockSizeBestFea = conf.m_maxBlockSize;

	PickGlobalBestFea<<<numofSNode, blockSizeBestFea>>>(manager.m_pLastValuePerThread,
					  manager.m_pBestSplitPointPerThread, manager.m_pRChildStatPerThread, manager.m_pLChildStatPerThread,
					  manager.m_pBuffIdVec, numofSNode, pfBestGain, pnBestGainKey, numofBlockPerNode);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PickGlobalBestFea" << endl;
		exit(0);
	}
#endif

	//Memory set for best split points; may not be necessary now.
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);

	manager.MemcpyDeviceToDevice(manager.m_pLastValuePerThread, manager.m_pLastValue, sizeof(float_point) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pBestSplitPointPerThread, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pRChildStatPerThread, manager.m_pRChildStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pLChildStatPerThread, manager.m_pLChildStat, sizeof(nodeStat) * maxNumofSplittable);

#if false
	//print best split points
	int *pTestBuffIdVect = new int[numofSNode];
	manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pTestBuffIdVect, sizeof(int) * numofSNode);
	SplitPoint *testBestSplitPoint = new SplitPoint[maxNumofSplittable];
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	for(int sn = 0; sn < numofSNode; sn++)
	{
		cout << "nid=" << pTestBuffIdVect[sn] << "; snid=" << sn << "; gain=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fGain << "; fid="
			 << testBestSplitPoint[pTestBuffIdVect[sn]].m_nFeatureId << "; sv=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fSplitValue << endl;
	}
	delete[] testBestSplitPoint;
	delete[] pTestBuffIdVect;
#endif
}


