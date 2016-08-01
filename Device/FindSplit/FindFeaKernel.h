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

//dense array
__global__ void ComputeIndex(int *pDstIndexEachFeaValue, long long totalFeaValue);
__global__ void LoadGDHessFvalueRoot(const float_point *pInsGD, const float_point *pInsHess, int numIns,
						   const int *pInsId, const float_point *pAllFeaValue, int numFeaValue,
						   float_point *pGDEachFeaValue, float_point *pHessEachFeaValue, float_point *pDenseFeaValue);
__global__ void LoadGDHessFvalue(const float_point *pInsGD, const float_point *pInsHess, int numIns,
						   const int *pInsId, const float_point *pAllFeaValue, const int *pDstIndexEachFeaValue, int numFeaValue,
						   float_point *pGDEachFeaValue, float_point *pHessEachFeaValue, float_point *pDenseFeaValue);
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const long long *pFeaValueStartPosEachNode, int numSN,
							const int *pBuffId,	float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							const float_point *pDenseFeaValue, int numofDenseValue, float_point *pGainOnEachFeaValue);
__global__ void FirstFeaGain(const long long *pEachFeaStartPosEachNode, int numFeaStartPos, float_point *pGainOnEachFeaValue);
__global__ void PickLocalBestSplitEachNode(const long long *pnNumFeaValueEachNode, const long long *pFeaStartPosEachNode,
										   const float_point *pGainOnEachFeaValue,
								   	   	   float_point *pfLocalBestGain, int *pnLocalBestGainKey);
__global__ void PickGlobalBestSplitEachNode(const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   	   	    float_point *pfGlobalBestGain, int *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode);
__global__ void FindSplitInfo(const long long *pEachFeaStartPosEachNode, const int *pEachFeaLenEachNode,
							  const float_point *pDenseFeaValue, const float_point *pfGlobalBestGain, const int *pnGlobalBestGainKey,
							  const int *pPosToBuffId, const int numFea,
							  const nodeStat *snNodeStat, const float_point *pPrefixSumGD, const float_point *pPrefixSumHess,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat);

//early
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
								 const float_point *pHess, const int *pBuffId, const int *pSNIdToBuffId, int maxNumofSplittable,
								 int numofSNInProgress, int smallestNodeId, int smallestFeaId, int totalNumofFea,
								 int feaBatch, float_point *pGDOnEachFeaValue, float_point *pHessOnEachFeaValue,
								 float_point *pValueOneEachFeaValue);
__global__ void GetInfoEachFeaInBatch(const int *pnNumofKeyValues, const long long *pnFeaStartPos, int smallestFeaId,
									  int totalNumofFea, int feaBatch, int numofSNInProgress, int smallestNodeId,
									  int *pStartPosEachFeaInBatch, int *pnEachFeaLen);
void PrefixSumForEachNode(int feaBatch, float_point *pGDOnEachFeaValue_d, float_point *pHessOnEachFeaValue_d,
						  const long long *pnStartPosEachFeaInBatch, const int *pnEachFeaLen, int maxNumValuePerFea);
__global__ void ComputeGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const nodeStat *pSNodeStat, int smallestFeaId, int feaBatch,
							const int *pBuffId, int numofSNInProgress, int smallestNodeId, float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							const float_point *pHessOnEachFeaValue, const float_point *pFeaValue,
							float_point *pGainOnEachFeaValue);
__global__ void FixedGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos,
						  int smallestFeaId, int feaBatch, int numofSNode, int smallestNodeId,
						  const float_point *pHessOnEachFeaValue, const float_point *pFeaValue,
						  float_point *pGainOnEachFeaValue, float_point *pLastBiggerValue);
__global__ void PickFeaLocalBestSplit(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const float_point *pGainOnEachFeaValue,
								   const int *pBuffId, int smallestFeaId, int feaBatch,
								   int numofSNode, int smallestNodeId, int maxNumofSplittable,
								   float_point *pfBestGain, int *pnBestGainKey);
__global__ void PickFeaGlobalBestSplit(int feaBatch, int numofSNode,
								   const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   float_point *pfFeaGlobalBestGain, int *pnFeaGlobalBestGainKey,
								   int numofLocalBlockOfAllFeaInBatch);
__global__ void PickLocalBestFeaBestSplit(int feaBatch, int numofSNode,
								   const float_point *pfFeaGlobalBestGain, const int *pnFeaGlobalBestGainKey,
								   float_point *pfBlockBestFea, int *pnBlockBestKey);
__global__ void PickGlobalBestFeaBestSplit(int numofSNode, int nBlockPerNode,
								   const float_point *pfBlockBestFea, const int *pnBlockBestFeaKey,
								   float_point *pfGlobalBestFea, int *pnGlobalBestKey);
__global__ void FindSplitInfo(const int *pnKeyValue, const long long *plFeaStartPos, const float_point *pFeaValue,
							  int feaBatch, int smallestFeaId, int smallestNodeId,
							  const float_point *pfGlobalBestFea, const int *pnGlobalBestKey, const int *pBuffId,
							  const nodeStat *snNodeStat, const float_point *pPrefixSumGD, const float_point *pPrefixSumHess,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat,
							  float_point *pLastValue, const float_point *pGainOnEachFeaValue_d);

//helper functions
__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);
__device__ void GetBatchInfo(int feaBatch, int smallestFeaId, int feaId, const int *pnNumofKeyValues, const long long *pnFeaStartPos,
							 int &curFeaStartPosInBatch, int &nFeaValueInBatch);

#endif /* DEVICESPLITTERKERNEL_H_ */
