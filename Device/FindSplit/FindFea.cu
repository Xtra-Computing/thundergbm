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
#include "../Memory/findFeaMemManager.h"
#include "../../DeviceHost/MyAssert.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;

#ifdef testing
//#undef testing
#endif

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	GBDTGPUMemManager manager;
	int numofSNode = manager.m_curNumofSplitable;

	FFMemManager ffManager;
	ffManager.resetMemForFindFea();

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
	long long totalEleInWholeBatch = manager.m_totalNumofValues * numofSNode;//######### use all the features

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
											ffManager.m_pGDOnEachFeaValue_d, ffManager.m_pHessOnEachFeaValue_d, ffManager.m_pValueOnEachFeaValue_d);
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
	float_point *pGDOnEachFeaVaue_h = new float_point[totalEleInWholeBatch];
	float_point *pHessOnEachFeaValue_h = new float_point[totalEleInWholeBatch];
	manager.MemcpyDeviceToHost(ffManager.m_pGDOnEachFeaValue_d, pGDOnEachFeaVaue_h, sizeof(float_point) * totalEleInWholeBatch);
	manager.MemcpyDeviceToHost(ffManager.m_pHessOnEachFeaValue_d, pHessOnEachFeaValue_h, sizeof(float_point) * totalEleInWholeBatch);

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
				if(nid == -1)
					continue;

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
	int blockSizePosEachFeaInBatch;
	dim3 dimNumofBlockFindPosEachFeaInBatch;
	conf.ConfKernel(feaBatch, blockSizePosEachFeaInBatch, dimNumofBlockFindPosEachFeaInBatch);
	PROCESS_ERROR(dimNumofBlockFindPosEachFeaInBatch.z == 1 && dimNumofBlockFindPosEachFeaInBatch.y == 1);
	GetInfoEachFeaInBatch<<<dimNumofBlockFindPosEachFeaInBatch, blockSizePosEachFeaInBatch>>>(
												manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, smallestFeaId, nNumofFeature,
											    feaBatch, numofSNode, ffManager.m_pStartPosEachFeaInBatch_d, ffManager.m_pFeaLenInBatch_d);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in GetInfoEachFeaInBatch" << endl;
		exit(0);
	}
	int *pStartPosEachFeaInBatch_h = new int[feaBatch * numofSNode];
	int *pFeaLenInBatch_h = new int[feaBatch * numofSNode];

	manager.MemcpyDeviceToHost(ffManager.m_pStartPosEachFeaInBatch_d, pStartPosEachFeaInBatch_h, sizeof(int) * feaBatch * numofSNode);
	manager.MemcpyDeviceToHost(ffManager.m_pFeaLenInBatch_d, pFeaLenInBatch_h, sizeof(int) * feaBatch * numofSNode);

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
	manager.MemcpyDeviceToDevice(ffManager.m_pGDOnEachFeaValue_d, ffManager.m_pGDPrefixSum_d, sizeof(float_point) * totalEleInWholeBatch);
	manager.MemcpyDeviceToDevice(ffManager.m_pHessOnEachFeaValue_d, ffManager.m_pHessPrefixSum_d, sizeof(float_point) * totalEleInWholeBatch);

	manager.MemcpyDeviceToHost(ffManager.m_pFeaLenInBatch_d, ffManager.m_pnEachFeaLen_h, sizeof(int) * feaBatch * numofSNode);
	PrefixSumForEachNode(feaBatch * numofSNode, ffManager.m_pGDPrefixSum_d, ffManager.m_pHessPrefixSum_d,
						 ffManager.m_pStartPosEachFeaInBatch_d, ffManager.m_pnEachFeaLen_h);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PrefixSumForEachNode" << endl;
		exit(0);
	}

	float_point *pGDPrefixSumOnEachFeaValue_h = new float_point[totalEleInWholeBatch];
	float_point *pHessPrefixSumOnEachFeaValue_h = new float_point[totalEleInWholeBatch];
	manager.MemcpyDeviceToHost(ffManager.m_pGDPrefixSum_d, pGDPrefixSumOnEachFeaValue_h, sizeof(float_point) * totalEleInWholeBatch);
	manager.MemcpyDeviceToHost(ffManager.m_pHessPrefixSum_d, pHessPrefixSumOnEachFeaValue_h, sizeof(float_point) * totalEleInWholeBatch);

	float_point deltaTest = 0.01;
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
				if(
				   abs(prefixSumGD - pGDPrefixSumOnEachFeaValue_h[e + n * batchSize]) > deltaTest ||
				   prefixSumHess != pHessPrefixSumOnEachFeaValue_h[e + n * batchSize])
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
	ComputeGain<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pSNodeStat, smallestFeaId, feaBatch,
									   manager.m_pBuffIdVec, numofSNode, DeviceSplitter::m_lambda, ffManager.m_pGDPrefixSum_d,
									   ffManager.m_pHessPrefixSum_d, manager.m_pdDFeaValue, ffManager.m_pGainOnEachFeaValue_d);


#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}
	nodeStat *pSNodeStat_h = new nodeStat[maxNumofSplittable];
	float_point *pGainOnEachFeaValue_h = new float_point[totalEleInWholeBatch];
	manager.MemcpyDeviceToHost(manager.m_pSNodeStat, pSNodeStat_h, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(ffManager.m_pGainOnEachFeaValue_d, pGainOnEachFeaValue_h, sizeof(float_point) * totalEleInWholeBatch);

	e = 0;
	float_point *pFeaBestSplit = new float_point[feaBatch * numofSNode];
	memset(pFeaBestSplit, 0, sizeof(float_point) * feaBatch * numofSNode);
	for(int f = smallestFeaId; f < feaBatch + smallestFeaId; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		int init = e;
		float_point prefixSumGD = 0;
		float_point prefixSumHess = 0;
		for(int i = init; i < numofKeyValue + init; i++)
		{
			int insId = pnInsId[i];
			int nid = pInsToNodeId_h[insId];
			if(nid == -1)
			{
				e++;
				continue;
			}

			for(int n = 0; n < numofSNode; n++)
			{
				int hashValue_buffer = Hashing::HostGetBufferId(pSNIdToBuffId_h, nid, maxNumofSplittable);
				if(hashValue_buffer < 0)
					cerr << "hash value in host side is negative: " << hashValue_buffer << " for key " << nid << endl;
				int hashValue = pBuffIdVec_h[n];
				if(hashValue_buffer != hashValue)
					continue;

				float_point snGD = pSNodeStat_h[hashValue].sum_gd;
				float_point snHess = pSNodeStat_h[hashValue].sum_hess;
				float_point gainOnFeaValue;
				if(i == init)
				{
					gainOnFeaValue = 0;
				}
				else
				{
					int exclusiveSumPos = i + n * batchSize - 1;//follow xgboost using exlusive sum on gd and hess
					float_point tempGD = pGDPrefixSumOnEachFeaValue_h[exclusiveSumPos];
					float_point tempHess = pHessPrefixSumOnEachFeaValue_h[exclusiveSumPos];
					float_point fLChildGD = snGD - tempGD;
					float_point fLChildHess = snHess - tempHess;
					if(fLChildHess >= DeviceSplitter::min_child_weight && tempHess >= DeviceSplitter::min_child_weight)
					{
						gainOnFeaValue = (tempGD * tempGD)/(tempHess + DeviceSplitter::m_lambda) +
									 (fLChildGD * fLChildGD)/(fLChildHess + DeviceSplitter::m_lambda) -
									 (snGD * snGD)/(snHess + DeviceSplitter::m_lambda);
					}
					else
						gainOnFeaValue = 0;
				}
				if(abs(gainOnFeaValue - pGainOnEachFeaValue_h[i + n * batchSize]) > deltaTest)
				{
					cerr << "gain diff: "<< gainOnFeaValue << " v.s. " << pGainOnEachFeaValue_h[i + n * batchSize] << endl;
				}

				if(pFeaBestSplit[f + feaBatch * n] < gainOnFeaValue)
					pFeaBestSplit[f + feaBatch * n] = gainOnFeaValue;
			}
			e++;
		}
	}

	for(int n = 0; n < numofSNode; n++)
	{
		for(int f = 0; f < feaBatch; f++)
		{
//			cout << "local best before fixing for " << f << "th here is " << pFeaBestSplit[f + feaBatch * n] << endl;
		}
	}

	delete []pFeaBestSplit;

	delete []pInsToNodeId_h;
	delete []pnInsId;
	delete []pGDPrefixSumOnEachFeaValue_h;
	delete []pHessPrefixSumOnEachFeaValue_h;

	delete []pSNIdToBuffId_h;
#endif

	//remove invalid gains. The same value can only have one gain
	float_point *pLastBiggerValue_d;
	checkCudaErrors(cudaMalloc((void**)&pLastBiggerValue_d, sizeof(float_point) * totalEleInWholeBatch));
	checkCudaErrors(cudaMemset(pLastBiggerValue_d, 0, sizeof(float_point) * totalEleInWholeBatch));
	FixedGain<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos,  smallestFeaId, feaBatch, numofSNode,
									 ffManager.m_pHessOnEachFeaValue_d, manager.m_pdDFeaValue, ffManager.m_pGainOnEachFeaValue_d, pLastBiggerValue_d);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in FixedGain" << endl;
		exit(0);
	}
#endif
	//find the local best split in this batch of features
	int nBlockEachFea = dimGrid.x;
	int nElePerBlock = dimBlock.x;
	PickFeaLocalBestSplit<<<dimGrid, dimBlock>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, ffManager.m_pGainOnEachFeaValue_d,
											  manager.m_pBuffIdVec, smallestFeaId, feaBatch,
											  numofSNode, maxNumofSplittable, ffManager.m_pfFeaLocalBestGain_d, ffManager.m_pnFeaLocalBestGainKey_d);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}

	float_point *pfFeaLocalBestGain_h = new float_point[feaBatch * nBlockEachFea * numofSNode];
	int *pnFeaLocalBestGainKey_h = new int[feaBatch * nBlockEachFea * numofSNode];
	manager.MemcpyDeviceToHost(ffManager.m_pfFeaLocalBestGain_d, pfFeaLocalBestGain_h, sizeof(float_point) * feaBatch * nBlockEachFea * numofSNode);
	manager.MemcpyDeviceToHost(ffManager.m_pnFeaLocalBestGainKey_d, pnFeaLocalBestGainKey_h, sizeof(int) * feaBatch * nBlockEachFea * numofSNode);

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
//			cout << "local best for " << f << "th fea is " << localBest << " for node " << n << endl;
			if(pGlobalBest[n] > localBest)
				pGlobalBest[n] = localBest;
			pFeaBest[n * feaBatch + f] = localBest;
		}
	}

	//obtain the best for each node
	//use the fixed gain to compute the best gain
	manager.MemcpyDeviceToHost(ffManager.m_pGainOnEachFeaValue_d, pGainOnEachFeaValue_h, sizeof(float_point) * totalEleInWholeBatch);
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
//		cout << "snid=" << n << "; best gain = " << bestGain << "; key is " << key << "; f=" << bestFeaId << "; value pos=" << valuePos << endl;
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
	int nThreadFeaBestBlock = nBlockEachFea;
	if(nThreadFeaBestBlock > conf.m_maxBlockSize)
		nThreadFeaBestBlock = conf.m_maxBlockSize;

	dim3 dimBlockSizeFeaBest(nThreadFeaBestBlock, 1, 1);
	dim3 dimGridFeaBest(1, feaBatch, numofSNode);
	PickFeaGlobalBestSplit<<<dimGridFeaBest, dimBlockSizeFeaBest>>>(
							feaBatch, numofSNode, ffManager.m_pfFeaLocalBestGain_d, ffManager.m_pnFeaLocalBestGainKey_d,
							ffManager.m_pfFeaGlobalBestGain_d, ffManager.m_pnFeaGlobalBestGainKey_d, nBlockEachFea);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in ComputeGain" << endl;
		exit(0);
	}

	float_point *pfFeaGlobalBestGain_h = new float_point[feaBatch * numofSNode];
	int *pnFeaGlobalBestGainKey_h = new int[feaBatch * numofSNode];
	manager.MemcpyDeviceToHost(ffManager.m_pfFeaGlobalBestGain_d, pfFeaGlobalBestGain_h, sizeof(float_point) * feaBatch * numofSNode);
	manager.MemcpyDeviceToHost(ffManager.m_pnFeaGlobalBestGainKey_d, pnFeaGlobalBestGainKey_h, sizeof(int) * feaBatch * numofSNode);

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
	//kernel configuration
	int blockSizeBestFeaBestSplit;
	dim3 tempNumofBlockBestFea;
	conf.ConfKernel(feaBatch, blockSizeBestFeaBestSplit, tempNumofBlockBestFea);
	int nBlockBestFea = tempNumofBlockBestFea.x;
	PROCESS_ERROR(tempNumofBlockBestFea.y == 1);
	dim3 dimBlockSizeBestFeaBestSplit(blockSizeBestFeaBestSplit, 1, 1);
	dim3 dimGridBestFeaBestSplit(nBlockBestFea, numofSNode, 1);

	PickLocalBestFeaBestSplit<<<dimGridBestFeaBestSplit, dimBlockSizeBestFeaBestSplit>>>
											(feaBatch, numofSNode, ffManager.m_pfFeaGlobalBestGain_d,
											 ffManager.m_pnFeaGlobalBestGainKey_d, ffManager.m_pfBlockBestFea_d, ffManager.m_pnBlockBestKey_d);

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PickBestFeaBestSplit" << endl;
		exit(0);
	}

	float_point *pfBlockBestFea_h = new float_point[nBlockBestFea * numofSNode];
	int *pnBlockBestKey_h = new int[nBlockBestFea * numofSNode];
	manager.MemcpyDeviceToHost(ffManager.m_pfBlockBestFea_d, pfBlockBestFea_h, sizeof(float_point) * nBlockBestFea * numofSNode);
	manager.MemcpyDeviceToHost(ffManager.m_pnBlockBestKey_d, pnBlockBestKey_h, sizeof(int) * nBlockBestFea * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{
		for(int f = 0; f < nBlockBestFea; f++)
		{
			if(pfBlockBestFea_h[n] != pGlobalBest[n] && nBlockBestFea == 1)
				cerr << "best gain diff: " << pfBlockBestFea_h[n] << " v.s. " << pGlobalBest[n] << endl;
		}
	}

	delete []pfBlockBestFea_h;
	delete []pnBlockBestKey_h;
#endif

	if(nBlockBestFea > 1)
	{
		int threadPerBlockBestFea;
		dim3 dimNumofBlockBestFea;
		conf.ConfKernel(nBlockBestFea, threadPerBlockBestFea, dimNumofBlockBestFea);
		PROCESS_ERROR(dimNumofBlockBestFea.y == 1 && dimNumofBlockBestFea.z == 1 && dimNumofBlockBestFea.x == 1);
		dimNumofBlockBestFea.x = numofSNode;
		PickGlobalBestFeaBestSplit<<<dimNumofBlockBestFea, threadPerBlockBestFea>>>(
													numofSNode, nBlockBestFea, ffManager.m_pfBlockBestFea_d,
													ffManager.m_pnBlockBestKey_d, ffManager.m_pfGlobalBestFea_d, ffManager.m_pnGlobalBestKey_d);
	}
	else
	{//local best fea is the global best fea
		manager.MemcpyDeviceToDevice(ffManager.m_pfBlockBestFea_d, ffManager.m_pfGlobalBestFea_d, sizeof(float_point) * numofSNode);
		manager.MemcpyDeviceToDevice(ffManager.m_pnBlockBestKey_d, ffManager.m_pnGlobalBestKey_d, sizeof(int) * numofSNode);
	}

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PickGlobalBestFeaBestSplit" << endl;
		exit(0);
	}

	float_point *pfGlobalBestFea_h = new float_point[numofSNode];
	int *pnGlobalBestKey_h = new int[numofSNode];
	manager.MemcpyDeviceToHost(ffManager.m_pfGlobalBestFea_d, pfGlobalBestFea_h, sizeof(float_point) * numofSNode);
	manager.MemcpyDeviceToHost(ffManager.m_pnGlobalBestKey_d, pnGlobalBestKey_h, sizeof(int) * numofSNode);

	for(int n = 0; n < numofSNode; n++)
	{

		if(pfGlobalBestFea_h[n] != pGlobalBest[n])
			cerr << "best gain diff: " << pfGlobalBestFea_h[n] << " v.s. " << pGlobalBest[n] << endl;
	}
	delete []pGlobalBest;
	delete []pnGlobalBestKey_h;
	delete []pfGlobalBestFea_h;
#endif

	//get split point info

	//Memory set for best split points; may not be necessary now.
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);

	FindSplitInfo<<<1, numofSNode>>>(manager.m_pDNumofKeyValue, manager.m_pFeaStartPos, manager.m_pdDFeaValue,
								  feaBatch, smallestFeaId,
								  ffManager.m_pfGlobalBestFea_d, ffManager.m_pnGlobalBestKey_d, manager.m_pBuffIdVec,
								  manager.m_pSNodeStat, ffManager.m_pGDPrefixSum_d, ffManager.m_pHessPrefixSum_d,
								  manager.m_pBestSplitPoint, manager.m_pRChildStat, manager.m_pLChildStat,
								  manager.m_pLastValue, ffManager.m_pGainOnEachFeaValue_d);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "error in PickGlobalBestFeaBestSplit" << endl;
		exit(0);
	}
	SplitPoint *testBestSplitPoint1 = new SplitPoint[maxNumofSplittable];
	nodeStat *testpRChildStat = new nodeStat[maxNumofSplittable];
	nodeStat *testpLChildStat = new nodeStat[maxNumofSplittable];
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint1, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pRChildStat, testpRChildStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pLChildStat, testpLChildStat, sizeof(nodeStat) * maxNumofSplittable);

	for(int n = 0; n < numofSNode; n++)
	{
		int buffId = pBuffIdVec_h[n];
		if(testBestSplitPoint1[buffId].m_fGain != 0)
		{
			if(pSNodeStat_h[buffId].sum_hess != testpRChildStat[buffId].sum_hess + testpLChildStat[buffId].sum_hess ||
			   abs(pSNodeStat_h[buffId].sum_gd - testpRChildStat[buffId].sum_gd - testpLChildStat[buffId].sum_gd) > deltaTest)
				cerr << "parent node stat != child node stats: "<< pSNodeStat_h[buffId].sum_hess
					 << " != " << testpRChildStat[buffId].sum_hess << "+" << testpLChildStat[buffId].sum_hess
					 << "; " << pSNodeStat_h[buffId].sum_gd
					 << " != " << testpRChildStat[buffId].sum_gd << "+" << testpLChildStat[buffId].sum_gd << endl;
		}
	}

	delete []pSNodeStat_h;

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
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in FindFeaSplitValue" << endl;
		exit(0);
	}


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

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PickLocalBestFea" << endl;
		exit(0);
	}


	int blockSizeBestFea = numofBlockPerNode;
	if(blockSizeBestFea > conf.m_maxBlockSize)
		blockSizeBestFea = conf.m_maxBlockSize;

	PickGlobalBestFea<<<numofSNode, blockSizeBestFea>>>(manager.m_pLastValuePerThread,
					  manager.m_pBestSplitPointPerThread, manager.m_pRChildStatPerThread, manager.m_pLChildStatPerThread,
					  manager.m_pBuffIdVec, numofSNode, pfBestGain, pnBestGainKey, numofBlockPerNode);

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PickGlobalBestFea" << endl;
		exit(0);
	}


	//Memory set for best split points; may not be necessary now.
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pLastValuePerThread, manager.m_pLastValue, sizeof(float_point) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pRChildStatPerThread, manager.m_pRChildStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pLChildStatPerThread, manager.m_pLChildStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pBestSplitPointPerThread, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);


	//print best split points
	SplitPoint *testBestSplitPoint2 = new SplitPoint[maxNumofSplittable];
	nodeStat *testpRChildStat2 = new nodeStat[maxNumofSplittable];
	nodeStat *testpLChildStat2 = new nodeStat[maxNumofSplittable];
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint2, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pRChildStat, testpRChildStat2, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pLChildStat, testpLChildStat2, sizeof(nodeStat) * maxNumofSplittable);
	for(int sn = 0; sn < numofSNode; sn++)
	{
		int buffId = pBuffIdVec_h[sn];
		if(testBestSplitPoint1[buffId].m_nFeatureId != 0 && testBestSplitPoint2[buffId].m_nFeatureId != -1)
		{
		if(testBestSplitPoint1[buffId].m_nFeatureId != testBestSplitPoint2[buffId].m_nFeatureId ||
		   abs(testBestSplitPoint1[buffId].m_fGain - testBestSplitPoint2[buffId].m_fGain) > deltaTest ||
		   abs(testpRChildStat[buffId].sum_gd - testpRChildStat2[buffId].sum_gd) > deltaTest ||
		   testpRChildStat[buffId].sum_hess != testpRChildStat2[buffId].sum_hess ||
		   abs(testpLChildStat[buffId].sum_gd - testpLChildStat2[buffId].sum_gd) > deltaTest ||
		   testpLChildStat[buffId].sum_hess != testpLChildStat2[buffId].sum_hess)
			cerr << "final result diff: " << testBestSplitPoint1[buffId].m_nFeatureId << " v.s. " << testBestSplitPoint2[buffId].m_nFeatureId
				 << "; " << testBestSplitPoint1[buffId].m_fGain << " v.s. " << testBestSplitPoint2[buffId].m_fGain
				 << "; " << testBestSplitPoint1[buffId].m_fSplitValue << " v.s. " << testBestSplitPoint2[buffId].m_fSplitValue
				 << "; r gd: " << testpRChildStat[buffId].sum_gd << " v.s. " << testpRChildStat2[buffId].sum_gd
				 << "; r hess: " << testpRChildStat[buffId].sum_hess << " v.s. " << testpRChildStat2[buffId].sum_hess
				 << "; l gd: " << testpLChildStat[buffId].sum_gd << " v.s. " << testpLChildStat2[buffId].sum_gd
				 << "; l hess: " << testpLChildStat[buffId].sum_hess << " v.s. " << testpLChildStat2[buffId].sum_hess
				 << endl;
		}


//		cout << "nid=" << pTestBuffIdVect[sn] << "; snid=" << sn << "; gain=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fGain << "; fid="
//			 << testBestSplitPoint[pTestBuffIdVect[sn]].m_nFeatureId << "; sv=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fSplitValue << endl;
	}
	delete []testpRChildStat;
	delete []testpLChildStat;
	delete []testBestSplitPoint1;
	delete []testpRChildStat2;
	delete []testpLChildStat2;
	delete []testBestSplitPoint2;
	delete []pBuffIdVec_h;
#endif
}


