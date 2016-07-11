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
	int smallestFeaId = 0;//######### need to change to handle datasets with a large number of features
	int feaBatch = nNumofFeature;//find best splits for a subset of features
	int maxNumofValuePerFea = manager.m_numofIns;//maximum number of instances that have non-zero value at the feature

	//gd, hess, and feature values on GPU memory
	float_point *pGDOnEachFeaValue_d, *pHessOnEachFeaValue_d, *pValueOneEachFeaValue_d;
	//######### can be more memory efficient by allocating only the total number of feature values in this batch
	checkCudaErrors(cudaMalloc((void**)&pGDOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pHessOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pValueOneEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));

	checkCudaErrors(cudaMemset(pGDOnEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMemset(pHessOnEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	checkCudaErrors(cudaMemset(pValueOneEachFeaValue_d, 0, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));

	//kernel configuration
	int blockSizeFillGD;
	dim3 dimNumofBlockToFillGD;
	KernelConf conf;
	conf.ConfKernel(maxNumofValuePerFea, blockSizeFillGD, dimNumofBlockToFillGD);
	PROCESS_ERROR(dimNumofBlockToFillGD.y == 1 && dimNumofBlockToFillGD.z == 1);//must be one dimensional block
	int numofBlockFillGD = dimNumofBlockToFillGD.x;
	dim3 dimGrid(numofBlockFillGD, feaBatch, numofSNode);
	dim3 dimBlock(blockSizeFillGD, 1, 1);

	//get gd for every splittable node
	ObtainGDEachNode<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pDInsId,
											manager.m_pdDFeaValue, manager.m_pInsIdToNodeId,
											manager.m_pGrad, manager.m_pHess,
											numofSNode, smallestFeaId, nNumofFeature, feaBatch,
											pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pValueOneEachFeaValue_d);
	//Note: pGDOnEachFeaValue_d has extra memory at the end. The prefix of the array is fully used.
	cudaDeviceSynchronize();

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ObtainGDEachNode" << endl;
		exit(0);
	}

	float_point *pGDOnEachFeaVaue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	float_point *pHessOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGDOnEachFeaValue_d, pGDOnEachFeaVaue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(pHessOnEachFeaValue_d, pHessOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	int *pnKeyValue = new int[nNumofFeature];
	long long *plFeaStartPos = new long long[nNumofFeature];
	int *pnInsId = new int[manager.m_totalNumofValues];
	float_point *pGrad = new float_point[manager.m_numofIns];

	manager.MemcpyDeviceToHost(manager.m_pDNumofKeyValue, pnKeyValue, sizeof(int) * nNumofFeature);
	manager.MemcpyDeviceToHost(manager.m_pFeaStartPos, plFeaStartPos, sizeof(long long) * nNumofFeature);
	manager.MemcpyDeviceToHost(manager.m_pDInsId, pnInsId, sizeof(int) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(manager.m_pGrad, pGrad, sizeof(float_point) * manager.m_numofIns);

	int e = 0;
	for(int f = 0; f < nNumofFeature; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		int init = e;
		for(int i = init; i < numofKeyValue + init; i++)
		{
			int insId = pnInsId[e];
			float_point gd = pGrad[insId];
			if(pGDOnEachFeaVaue_h[i] != gd || pHessOnEachFeaValue_h[i] != 1)
				cout << "hessian != 1: "<< pHessOnEachFeaValue_h[i] << "; gd diff: " << gd << " v.s. " << pGDOnEachFeaVaue_h[i] << endl;
			e++;
		}
	}

	delete []pnInsId;
	delete []pGrad;
#endif

	int *pStartPosEachFeaInBatch_d;//int is ok here
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
	int *pStartPosEachFeaInBatch_h = new int[feaBatch];
	int *pFeaLenInBatch_h = new int[feaBatch];

	manager.MemcpyDeviceToHost(pStartPosEachFeaInBatch_d, pStartPosEachFeaInBatch_h, sizeof(int) * feaBatch);
	manager.MemcpyDeviceToHost(pFeaLenInBatch_d, pFeaLenInBatch_h, sizeof(int) * feaBatch);

	for(int b = 0; b < feaBatch; b++)
	{
		int feaId = b + smallestFeaId;
		if(pStartPosEachFeaInBatch_h[feaId] != plFeaStartPos[feaId])
		{
			cout << "diff in start pos: " << pStartPosEachFeaInBatch_h[feaId] << " v.s. " << plFeaStartPos[feaId] << "; feaId=" << feaId << endl;
		}
		if(pFeaLenInBatch_h[feaId] != pnKeyValue[feaId])
		{
			cout << "diff in fea len: " << pFeaLenInBatch_h[feaId] << " v.s. " << pnKeyValue[feaId] << "; feaId=" << feaId << endl;
		}
	}

	delete []plFeaStartPos;
	delete []pStartPosEachFeaInBatch_h;
	delete []pFeaLenInBatch_h;
#endif

	//compute prefix sum
	int *pnEachFeaLen = new int[feaBatch];
	manager.MemcpyDeviceToHost(pFeaLenInBatch_d, pnEachFeaLen, sizeof(int) * feaBatch);
	PrefixSumForEachNode(feaBatch, pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pStartPosEachFeaInBatch_d, pnEachFeaLen);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PrefixSumForEachNode" << endl;
		exit(0);
	}

	float_point *pGDPrefixSumOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	float_point *pHessPrefixSumOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGDOnEachFeaValue_d, pGDPrefixSumOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(pHessOnEachFeaValue_d, pHessPrefixSumOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	e = 0;
	for(int f = 0; f < nNumofFeature; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		int init = e;
		float_point prefixSumGD = 0;
		float_point prefixSumHess = 0;
		for(int i = init; i < numofKeyValue + init; i++)
		{
			prefixSumGD += pGDOnEachFeaVaue_h[i];
			prefixSumHess += pHessOnEachFeaValue_h[i];
			if(prefixSumGD != pGDPrefixSumOnEachFeaValue_h[i] || prefixSumHess != pHessPrefixSumOnEachFeaValue_h[i])
				cout << "hessian or gd diff: hess "<< prefixSumHess << " v.s. " << pHessPrefixSumOnEachFeaValue_h[i]
				     << "; gd: " << prefixSumGD << " v.s. " << pGDPrefixSumOnEachFeaValue_h[i] << endl;
			e++;
		}
	}

	delete []pGDOnEachFeaVaue_h;
	delete []pHessOnEachFeaValue_h;
#endif

	//compute gain
	float_point *pGainOnEachFeaValue_d;
	checkCudaErrors(cudaMalloc((void**)&pGainOnEachFeaValue_d, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode));
	ComputeGain<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pSNodeStat, smallestFeaId, feaBatch,
									   manager.m_pBuffIdVec, numofSNode, DeviceSplitter::m_lambda, pGDOnEachFeaValue_d,
									   pHessOnEachFeaValue_d, pGainOnEachFeaValue_d);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ComputeGain" << endl;
		exit(0);
	}
	nodeStat *pSNodeStat_h = new nodeStat[maxNumofSplittable];
	float_point *pGainOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	int *pBuffIdVec_h = new int[numofSNode];
	manager.MemcpyDeviceToHost(manager.m_pSNodeStat, pSNodeStat_h, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(pGainOnEachFeaValue_d, pGainOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pBuffIdVec_h, sizeof(int) * numofSNode);

	e = 0;
	for(int f = 0; f < nNumofFeature; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		int init = e;
		float_point prefixSumGD = 0;
		float_point prefixSumHess = 0;
		for(int i = init; i < numofKeyValue + init; i++)
		{
			for(int n = 0; n < numofSNode; n++)
			{
				int hashValue = pBuffIdVec_h[n];
				float_point snGD = pSNodeStat_h[hashValue].sum_gd;
				float_point snHess = pSNodeStat_h[hashValue].sum_hess;
				float_point tempGD = pGDPrefixSumOnEachFeaValue_h[i];
				float_point tempHess = pHessPrefixSumOnEachFeaValue_h[i];
				float_point fLChildGD = snGD - tempGD;
				float_point fLChildHess = snHess - tempHess;
				float_point gainOnFeaValue = (tempGD * tempGD)/(tempHess + DeviceSplitter::m_lambda) +
											 (fLChildGD * fLChildGD)/(fLChildHess + DeviceSplitter::m_lambda) -
											 (snGD * snGD)/(snHess + DeviceSplitter::m_lambda);

				if(gainOnFeaValue != pGainOnEachFeaValue_h[i])
				{
					cout << "gain diff: "<< gainOnFeaValue << " v.s. " << pGainOnEachFeaValue_h[i] << endl;
				}
			}
			e++;
		}
	}

	float_point bestGain = 10000;
	for(int i = 0; i < feaBatch * maxNumofValuePerFea * numofSNode; i++)
	{
		if(bestGain < pGainOnEachFeaValue_h[i])
		{
			bestGain = pGainOnEachFeaValue_h[i];
		}
	}
	cout << "best gain = " << bestGain << endl;

	delete []pnKeyValue;
	delete []pGDPrefixSumOnEachFeaValue_h;
	delete []pHessPrefixSumOnEachFeaValue_h;

	delete []pSNodeStat_h;
	delete []pGainOnEachFeaValue_h;
	delete []pBuffIdVec_h;
#endif

	//find the local best split in this batch of features
	float_point *pfFeaLocalBestGain_d;
	int *pnFeaLocalBestGainKey_d;
	int nBlockEachFea = dimGrid.x;
	int nElePerBlock = dimBlock.x;
	checkCudaErrors(cudaMalloc((void**)&pfFeaLocalBestGain_d, sizeof(float_point) * feaBatch * nBlockEachFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnFeaLocalBestGainKey_d, sizeof(int) * feaBatch * nBlockEachFea * numofSNode));
	PickFeaLocalBestSplit<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, pGainOnEachFeaValue_d,
											  manager.m_pBuffIdVec, smallestFeaId, feaBatch,
											  numofSNode, maxNumofSplittable, pfFeaLocalBestGain_d, pnFeaLocalBestGainKey_d);
#if testing
	float_point *pfFeaLocalBestGain_h = new float_point[feaBatch * nBlockEachFea * numofSNode];
	int *pnFeaLocalBestGainKey_h = new int[feaBatch * nBlockEachFea * numofSNode];
	manager.MemcpyDeviceToHost(pfFeaLocalBestGain_d, pfFeaLocalBestGain_h, sizeof(float_point) * feaBatch * nBlockEachFea * numofSNode);
	manager.MemcpyDeviceToHost(pnFeaLocalBestGainKey_d, pnFeaLocalBestGainKey_h, sizeof(int) * feaBatch * nBlockEachFea * numofSNode);
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ComputeGain" << endl;
		exit(0);
	}

	float_point bestGain1 = 10000;
	int bestKey1 = -1;
	for(int i = 0; i < feaBatch * nBlockEachFea * numofSNode; i++)
	{
		if(bestGain1 > pfFeaLocalBestGain_h[i])
		{
			bestGain1 = pfFeaLocalBestGain_h[i];
			bestKey1 = pnFeaLocalBestGainKey_h[i];
		}
	}
	cout << "best gain = " << bestGain1 << "; best key = " << bestKey1 << endl;

	delete []pfFeaLocalBestGain_h;
	delete []pnFeaLocalBestGainKey_h;
#endif

	//find the best split for each feature in the batch
	float_point *pfFeaGlobalBestGain_d;
	int *pnFeaGlobalBestGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pfFeaGlobalBestGain_d, sizeof(float_point) * feaBatch * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnFeaGlobalBestGainKey_d, sizeof(int) * feaBatch * numofSNode));
	int nThreadFeaBestBlock = nBlockEachFea;
	if(nThreadFeaBestBlock > conf.m_maxBlockSize)
		nThreadFeaBestBlock = conf.m_maxBlockSize;

	dim3 dimBlockSizeFeaBest(nThreadFeaBestBlock, 1, 1);
	dim3 dimGridFeaBest(1, feaBatch, numofSNode);
	PickFeaGlobalBestSplit<<<dimGridFeaBest, dimBlockSizeFeaBest>>>(
							feaBatch, numofSNode, pfFeaLocalBestGain_d, pnFeaLocalBestGainKey_d,
							pfFeaGlobalBestGain_d, pnFeaGlobalBestGainKey_d, nBlockEachFea);
#if testing
	float_point *pfFeaGlobalBestGain_h = new float_point[feaBatch * numofSNode];
	int *pnFeaGlobalBestGainKey_h = new int[feaBatch * numofSNode];
	manager.MemcpyDeviceToHost(pfFeaGlobalBestGain_d, pfFeaGlobalBestGain_h, sizeof(float_point) * feaBatch * numofSNode);
	manager.MemcpyDeviceToHost(pnFeaGlobalBestGainKey_d, pnFeaGlobalBestGainKey_h, sizeof(int) * feaBatch * numofSNode);
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ComputeGain" << endl;
		exit(0);
	}
	delete []pfFeaGlobalBestGain_h;
	delete []pnFeaGlobalBestGainKey_h;
#endif

	checkCudaErrors(cudaFree(pGDOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pHessOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pValueOneEachFeaValue_d));
	checkCudaErrors(cudaFree(pStartPosEachFeaInBatch_d));
	checkCudaErrors(cudaFree(pFeaLenInBatch_d));
	checkCudaErrors(cudaFree(pGainOnEachFeaValue_d));
	checkCudaErrors(cudaFree(pfFeaLocalBestGain_d));
	checkCudaErrors(cudaFree(pnFeaLocalBestGainKey_d));
	checkCudaErrors(cudaFree(pfFeaGlobalBestGain_d));
	checkCudaErrors(cudaFree(pnFeaGlobalBestGainKey_d));
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

#if true
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


