/*
 * DeviceSplitterKernel.h
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTERKERNEL_H_
#define DEVICESPLITTERKERNEL_H_

#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/DefineConst.h"
#include "../../DeviceHost/NodeStat.h"

__global__ void FindFeaSplitValue(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId, const float_point *pFeaValue,
								  const int *pInsIdToNodeId, const float_point *pGD, const float_point *pHess,
								  nodeStat *pTempRChildStatPerThread, float_point *pLastValuePerThread,
								  const nodeStat *pSNodeStatPerThread, SplitPoint *pBestSplitPointPerThread,
								  nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
								  const int *pSNIdToBuffId, int maxNumofSplittable, const int *pBuffId, int numofSNode,
								  float_point lambda, int numofFea);
__global__ void PickLocalBestFea(const SplitPoint *pBestSplitPointPerThread, const int *pBuffId,
								 int numofSNode, int numofFea, int maxNumofSplittable,
								 float_point *pfBestGain, int *pnBestGainKey);
__global__ void PickGlobalBestFea(float_point *pLastValuePerThread,
							SplitPoint *pBestSplitPointPerThread, nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
							const int *pBuffId, int numofSNode, const float_point *pfBestGain, const int *pnBestGainKey, int numofBlockPerNode);

//a thread per gain computation
__global__ void ObtainGDEachNode(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId,
								 const float_point *pFeaValue, const int *pInsIdToNodeId, const float_point *pGD,
								 const float_point *pHess,  int numofSNode, int smallestFeaId, int totalNumofFea,
								 int feaBatch, float_point *pGDOnEachFeaValue, float_point *pHessOnEachFeaValue,
								 float_point *pValueOneEachFeaValue);
__global__ void GetInfoEachFeaInBatch(const int *pnNumofKeyValues, const long long *pnFeaStartPos, int smallestFeaId,
									  int totalNumofFea, int feaBatch, int *pStartPosEachFeaInBatch, int *pnEachFeaLen);
void PrefixSumForEachNode(int feaBatch, float_point *pGDOnEachFeaValue_d, float_point *pHessOnEachFeaValue_d,
						  const int *pnStartPosEachFeaInBatch, const int *pnEachFeaLen);
__global__ void ComputeGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const nodeStat *pSNodeStat, int smallestFeaId, int feaBatch,
							const int *pBuffId, int numofSNode, float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							float_point *pGainOnEachFeaValue);
__global__ void PickFeaLocalBestSplit(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const float_point *pGainOnEachFeaValue,
								   const int *pBuffId, int smallestFeaId, int feaBatch,
								   int numofSNode, int maxNumofSplittable,
								   float_point *pfBestGain, int *pnBestGainKey);
__global__ void PickFeaGlobalBestSplit(int feaBatch, int numofSNode,
								   const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   float_point *pfFeaGlobalBestGain, int *pnFeaGlobalBestGainKey,
								   int numofLocalBlockOfAllFeaInBatch);
//helper functions
__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);

#endif /* DEVICESPLITTERKERNEL_H_ */
