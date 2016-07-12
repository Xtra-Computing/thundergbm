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
#include "../Hashing.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;


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
											manager.m_pGrad, manager.m_pHess, manager.m_pBuffIdVec, manager.m_pSNIdToBuffId,
											maxNumofSplittable, numofSNode, smallestFeaId, nNumofFeature, feaBatch,
											pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pValueOneEachFeaValue_d);
	//Note: pGDOnEachFeaValue_d has extra memory at the end. The prefix of the array is fully used.
	cudaDeviceSynchronize();

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ObtainGDEachNode" << endl;
		exit(0);
	}

	//nid, buffer id, snid relationships
	int *pBuffIdVec_h = new int[numofSNode];
	int *pSNIdToBuffId_h = new int[maxNumofSplittable];
	int *pInsToNodeId_h = new int[manager.m_numofIns];
	manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pBuffIdVec_h, sizeof(int) * numofSNode);
	manager.MemcpyDeviceToHost(manager.m_pSNIdToBuffId, pSNIdToBuffId_h, sizeof(int) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pInsIdToNodeId, pInsToNodeId_h, sizeof(int) * manager.m_numofIns);

	//gd/hess for each fea value
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

	//get current fea batch size
	long long startPosOfSmallest = plFeaStartPos[smallestFeaId];
	int largestFeaId = smallestFeaId + feaBatch - 1;
	long long startPosOfLargest = plFeaStartPos[largestFeaId];
	int batchSize = startPosOfLargest - startPosOfSmallest + pnKeyValue[largestFeaId];

	int e = 0;
	for(int n = 0; n < numofSNode; n++)//for each splittable node
	{
		int hashValue = pBuffIdVec_h[n];
		e = 0;
		for(int f = 0; f < nNumofFeature; f++)
		{
			int numofKeyValue = pnKeyValue[f];
			for(int i = 0; i < numofKeyValue; i++)
			{
				int insId = pnInsId[e];
				float_point gd = pGrad[insId];
				int nid = pInsToNodeId_h[insId];
				int hashValue_buffer = Hashing::HostGetBufferId(pSNIdToBuffId_h, nid, maxNumofSplittable);
				if(pHessOnEachFeaValue_h[e + n * batchSize] != 0 && pHessOnEachFeaValue_h[e + n * batchSize] != 1)
				{
					cerr << "hess should be 1 or 0: " << pHessOnEachFeaValue_h[e + n * batchSize] << endl;
				}
				if(hashValue == hashValue_buffer)
				{
					if(pGDOnEachFeaVaue_h[e + n * batchSize] != gd || pHessOnEachFeaValue_h[e + n * batchSize] != 1)
						cerr << "hessian != 1: "<< pHessOnEachFeaValue_h[e + n * batchSize] << "; gd diff: "
							 << gd << " v.s. " << pGDOnEachFeaVaue_h[e + n * batchSize] << endl;
				}
				else
					if(pGDOnEachFeaVaue_h[e + n * batchSize] != 0 || pHessOnEachFeaValue_h[e + n * batchSize] != 0)
						cerr << "hessian != 0: "<< pHessOnEachFeaValue_h[e + n * batchSize] << "; gd diff: "
							 << gd << " v.s. " << pGDOnEachFeaVaue_h[e + n * batchSize] << endl;
				e++;
			}
		}
	}

	delete []pGrad;
#endif

	//each splittable node has its own copy of feature start pos and fea value length info, for calling the API of prefix sum
	int *pStartPosEachFeaInBatch_d;//int is ok here
	int *pFeaLenInBatch_d;
	checkCudaErrors(cudaMalloc((void**)&pStartPosEachFeaInBatch_d, sizeof(int) * feaBatch * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pFeaLenInBatch_d, sizeof(int) * feaBatch * numofSNode));
	int blockSizePosEachFeaInBatch;
	dim3 dimNumofBlockFindPosEachFeaInBatch;
	conf.ConfKernel(feaBatch, blockSizePosEachFeaInBatch, dimNumofBlockFindPosEachFeaInBatch);
	PROCESS_ERROR(dimNumofBlockFindPosEachFeaInBatch.z == 1 && dimNumofBlockFindPosEachFeaInBatch.y == 1);
	GetInfoEachFeaInBatch<<<dimNumofBlockFindPosEachFeaInBatch, blockSizePosEachFeaInBatch>>>(
												manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, smallestFeaId, nNumofFeature,
											    feaBatch, numofSNode, pStartPosEachFeaInBatch_d, pFeaLenInBatch_d);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in GetInfoEachFeaInBatch" << endl;
		exit(0);
	}
	int *pStartPosEachFeaInBatch_h = new int[feaBatch * numofSNode];
	int *pFeaLenInBatch_h = new int[feaBatch * numofSNode];

	manager.MemcpyDeviceToHost(pStartPosEachFeaInBatch_d, pStartPosEachFeaInBatch_h, sizeof(int) * feaBatch * numofSNode);
	manager.MemcpyDeviceToHost(pFeaLenInBatch_d, pFeaLenInBatch_h, sizeof(int) * feaBatch * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{
		for(int b = 0; b < feaBatch; b++)
		{
			int feaId = b + smallestFeaId;
			if(pStartPosEachFeaInBatch_h[b + n * feaBatch] - n * batchSize != plFeaStartPos[feaId])
			{
				cerr << "diff in start pos: " << pStartPosEachFeaInBatch_h[b + n * feaBatch] << " v.s. "
					 << plFeaStartPos[feaId] << "; feaId=" << feaId << endl;
			}
			if(pFeaLenInBatch_h[b + n * feaBatch] != pnKeyValue[feaId])
			{
				cerr << "diff in fea len: " << pFeaLenInBatch_h[b + n * feaBatch] << " v.s. " << pnKeyValue[feaId] << "; feaId=" << feaId << endl;
			}
		}
	}

	delete []pStartPosEachFeaInBatch_h;
	delete []pFeaLenInBatch_h;
#endif

	//compute prefix sum
	int *pnEachFeaLen = new int[feaBatch * numofSNode];
	manager.MemcpyDeviceToHost(pFeaLenInBatch_d, pnEachFeaLen, sizeof(int) * feaBatch * numofSNode);
	PrefixSumForEachNode(feaBatch * numofSNode, pGDOnEachFeaValue_d, pHessOnEachFeaValue_d, pStartPosEachFeaInBatch_d, pnEachFeaLen);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PrefixSumForEachNode" << endl;
		exit(0);
	}

	float_point *pGDPrefixSumOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	float_point *pHessPrefixSumOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(pGDOnEachFeaValue_d, pGDPrefixSumOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);
	manager.MemcpyDeviceToHost(pHessOnEachFeaValue_d, pHessPrefixSumOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{
		e = 0;
		for(int f = 0; f < nNumofFeature; f++)
		{
			int numofKeyValue = pnKeyValue[f];
			float_point prefixSumGD = 0;
			float_point prefixSumHess = 0;
			for(int i = 0; i < numofKeyValue; i++)
			{
				prefixSumGD += pGDOnEachFeaVaue_h[e + n * batchSize];
				prefixSumHess += pHessOnEachFeaValue_h[e + n * batchSize];
				if(prefixSumGD != pGDPrefixSumOnEachFeaValue_h[e + n * batchSize] || prefixSumHess != pHessPrefixSumOnEachFeaValue_h[e + n * batchSize])
					cerr << "hessian or gd diff: hess "<< prefixSumHess << " v.s. " << pHessPrefixSumOnEachFeaValue_h[e + n * batchSize]
						 << "; gd: " << prefixSumGD << " v.s. " << pGDPrefixSumOnEachFeaValue_h[e + n * batchSize] << endl;
				e++;
			}
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
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}
	nodeStat *pSNodeStat_h = new nodeStat[maxNumofSplittable];
	float_point *pGainOnEachFeaValue_h = new float_point[feaBatch * maxNumofValuePerFea * numofSNode];
	manager.MemcpyDeviceToHost(manager.m_pSNodeStat, pSNodeStat_h, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(pGainOnEachFeaValue_d, pGainOnEachFeaValue_h, sizeof(float_point) * feaBatch * maxNumofValuePerFea * numofSNode);

	e = 0;
	for(int f = 0; f < nNumofFeature; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		int init = e;
		float_point prefixSumGD = 0;
		float_point prefixSumHess = 0;
		for(int i = init; i < numofKeyValue + init; i++)
		{
			int insId = pnInsId[i];
			int nid = pInsToNodeId_h[insId];
			for(int n = 0; n < numofSNode; n++)
			{
				int hashValue_buffer = Hashing::HostGetBufferId(pSNIdToBuffId_h, nid, maxNumofSplittable);
				int hashValue = pBuffIdVec_h[n];

				float_point snGD = pSNodeStat_h[hashValue].sum_gd;
				float_point snHess = pSNodeStat_h[hashValue].sum_hess;
				float_point tempGD = pGDPrefixSumOnEachFeaValue_h[i + n * batchSize];
				float_point tempHess = pHessPrefixSumOnEachFeaValue_h[i + n * batchSize];
				float_point fLChildGD = snGD - tempGD;
				float_point fLChildHess = snHess - tempHess;
				float_point gainOnFeaValue = (tempGD * tempGD)/(tempHess + DeviceSplitter::m_lambda) +
											 (fLChildGD * fLChildGD)/(fLChildHess + DeviceSplitter::m_lambda) -
											 (snGD * snGD)/(snHess + DeviceSplitter::m_lambda);

				if(gainOnFeaValue != pGainOnEachFeaValue_h[i + n * batchSize])
				{
					cerr << "gain diff: "<< gainOnFeaValue << " v.s. " << pGainOnEachFeaValue_h[i] << endl;
				}
			}
			e++;
		}
	}

	delete []pSNIdToBuffId_h;
	delete []pInsToNodeId_h;
	delete []pnInsId;
	delete []pGDPrefixSumOnEachFeaValue_h;
	delete []pHessPrefixSumOnEachFeaValue_h;

	delete []pSNodeStat_h;
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
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}

	float_point *pfFeaLocalBestGain_h = new float_point[feaBatch * nBlockEachFea * numofSNode];
	int *pnFeaLocalBestGainKey_h = new int[feaBatch * nBlockEachFea * numofSNode];
	manager.MemcpyDeviceToHost(pfFeaLocalBestGain_d, pfFeaLocalBestGain_h, sizeof(float_point) * feaBatch * nBlockEachFea * numofSNode);
	manager.MemcpyDeviceToHost(pnFeaLocalBestGainKey_d, pnFeaLocalBestGainKey_h, sizeof(int) * feaBatch * nBlockEachFea * numofSNode);

	float_point *pGlobalBest = new float_point[numofSNode];
	float_point *pFeaBest = new float_point[numofSNode * feaBatch];
	for(int n = 0; n < numofSNode; n++)
	{
		pGlobalBest[n] = 100000;
		for(int f = 0; f < feaBatch; f++)
		{
			float_point localBest = 100000;
			for(int bl = 0; bl < nBlockEachFea; bl++)
			{
				float_point temp = pfFeaLocalBestGain_h[n * feaBatch * nBlockEachFea + f * nBlockEachFea + bl];
				if(localBest > temp)
					localBest = temp;
			}
			cout << "local best for " << f << "th fea is " << localBest << " for node " << n << endl;
			if(pGlobalBest[n] > localBest)
				pGlobalBest[n] = localBest;
			pFeaBest[n * feaBatch + f] = localBest;
		}
	}

	//obtain the best for each node
	for(int n = 0; n < numofSNode; n++)
	{
		float_point bestGain = -1000000;
		int key = -1;
		for(int i = 0; i < batchSize; i++)
		{
			if(bestGain < pGainOnEachFeaValue_h[i + n * batchSize])
			{
				bestGain = pGainOnEachFeaValue_h[i + n * batchSize];
				key = i;
			}
		}

		//compute feature id
		int bestFeaId = -1;
		int valuePos = -1;
		for(int f = 0; f < feaBatch; f++)
		{
			int numofKeyValue = pnKeyValue[f];
			if(plFeaStartPos[f] + numofKeyValue < key)
				continue;
			else
			{
				bestFeaId = f;
				valuePos = key - plFeaStartPos[f];
				break;
			}
		}
		cout << "snid=" << n << "; best gain = " << bestGain << "; key is " << key << "; f=" << bestFeaId << "; value pos=" << valuePos << endl;
		if(-bestGain != pGlobalBest[n])
			cerr << "best gain diff: " << bestGain << " v.s. " << pGlobalBest[n] << endl;
	}

	delete []plFeaStartPos;
	delete []pGainOnEachFeaValue_h;
	delete []pnKeyValue;

	for(int n = 0; n < numofSNode; n++)
	{
		float_point bestGain1 = 10000;
		int bestKey1 = -1;
		for(int i = 0; i < nBlockEachFea; i++)
		{
			if(bestGain1 > pfFeaLocalBestGain_h[i + n * nBlockEachFea])
			{
				bestGain1 = pfFeaLocalBestGain_h[i + n * nBlockEachFea];
				bestKey1 = pnFeaLocalBestGainKey_h[i + n * nBlockEachFea];
			}
		}
//		cout << "best gain = " << bestGain1 << "; best key = " << bestKey1 << endl;
	}

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
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}

	float_point *pfFeaGlobalBestGain_h = new float_point[feaBatch * numofSNode];
	int *pnFeaGlobalBestGainKey_h = new int[feaBatch * numofSNode];
	manager.MemcpyDeviceToHost(pfFeaGlobalBestGain_d, pfFeaGlobalBestGain_h, sizeof(float_point) * feaBatch * numofSNode);
	manager.MemcpyDeviceToHost(pnFeaGlobalBestGainKey_d, pnFeaGlobalBestGainKey_h, sizeof(int) * feaBatch * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{
		for(int f = 0; f < feaBatch; f++)
		{
			if(pfFeaGlobalBestGain_h[f + n * feaBatch] != pFeaBest[f + n * feaBatch])
				cerr << "fea best diff: " << pfFeaGlobalBestGain_h[f + n * feaBatch]
				     << " v.s. " << pFeaBest[f + n * feaBatch] << "; snid = " << n << "; fid=" << f << endl;
		}
	}

	delete []pFeaBest;
	delete []pfFeaGlobalBestGain_h;
	delete []pnFeaGlobalBestGainKey_h;
#endif

	//find the best feature
	float_point *pfBlockBestFea_d;
	int *pnBlockBestKey_d;

	//kernel configuration
	int blockSizeBestFeaBestSplit;
	dim3 tempNumofBlockBestFea;
	conf.ConfKernel(feaBatch, blockSizeBestFeaBestSplit, tempNumofBlockBestFea);
	int nBlockBestFea = tempNumofBlockBestFea.x;
	PROCESS_ERROR(tempNumofBlockBestFea.y == 1);
	dim3 dimBlockSizeBestFeaBestSplit(blockSizeBestFeaBestSplit, 1, 1);
	dim3 dimGridBestFeaBestSplit(nBlockBestFea, numofSNode, 1);

	checkCudaErrors(cudaMalloc((void**)&pfBlockBestFea_d, sizeof(float_point) * nBlockBestFea * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnBlockBestKey_d, sizeof(int) * nBlockBestFea * numofSNode));

	PickBestFeaBestSplit<<<dimGridBestFeaBestSplit, dimBlockSizeBestFeaBestSplit>>>
											(feaBatch, numofSNode, pfFeaGlobalBestGain_d,
											 pnFeaGlobalBestGainKey_d, pfBlockBestFea_d, pnBlockBestKey_d);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PickBestFeaBestSplit" << endl;
		exit(0);
	}

	float_point *pfBlockBestFea_h = new float_point[nBlockBestFea * numofSNode];
	int *pnBlockBestKey_h = new int[nBlockBestFea * numofSNode];
	manager.MemcpyDeviceToHost(pfBlockBestFea_d, pfBlockBestFea_h, sizeof(float_point) * nBlockBestFea * numofSNode);
	manager.MemcpyDeviceToHost(pnBlockBestKey_d, pnBlockBestKey_h, sizeof(int) * nBlockBestFea * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{
		for(int f = 0; f < nBlockBestFea; f++)
		{
			if(pfBlockBestFea_h[n] != pGlobalBest[n] && nBlockBestFea == 1)
				cerr << "best gain diff: " << pfBlockBestFea_h[n] << " v.s. " << pGlobalBest[n] << endl;
		}
	}

	delete []pGlobalBest;
#endif

	checkCudaErrors(cudaFree(pfBlockBestFea_d));
	checkCudaErrors(cudaFree(pnBlockBestKey_d));
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


