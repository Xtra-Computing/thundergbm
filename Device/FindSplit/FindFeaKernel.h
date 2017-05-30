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
#include "../../SharedUtility/DataType.h"
#include "../../DeviceHost/NodeStat.h"

//dense array
__global__ void LoadGDHessFvalueRoot(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue);
__global__ void LoadGDHessFvalue(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, const unsigned int *pDstIndexEachFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue);
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const unsigned int *pFeaValueStartPosEachNode, int numSN,
							const int *pBuffId,	real lambda,
							const double *pGDPrefixSumOnEachFeaValue, const real *pHessPrefixSumOnEachFeaValue,
							const real *pDenseFeaValue, int numofDenseValue, const unsigned int *pnLastFvalueOfThisFvalue,
							real *pGainOnEachFeaValue, bool *pDefault2Right);
__global__ void FirstFeaGain(const unsigned int *pEachFeaStartPosEachNode, int numFeaStartPos, real *pGainOnEachFeaValue, long long numFeaValue);
__global__ void PickLocalBestSplitEachNode(const unsigned int *pnNumFeaValueEachNode, const unsigned int *pFeaStartPosEachNode,
										   const real *pGainOnEachFeaValue,
								   	   	   real *pfLocalBestGain, int *pnLocalBestGainKey);
__global__ void PickGlobalBestSplitEachNode(const real *pfLocalBestGain, const int *pnLocalBestGainKey,
								   	   	    real *pfGlobalBestGain, int *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode);
__global__ void FindSplitInfo(const unsigned int *pEachFeaStartPosEachNode, const int *pEachFeaLenEachNode,
							  const real *pDenseFeaValue, const real *pfGlobalBestGain, const int *pnGlobalBestGainKey,
							  const int *pPosToBuffId, const int numFea,
							  const nodeStat *snNodeStat, const double *pPrefixSumGD, const real *pPrefixSumHess,
							  const bool *pDefault2Right, const unsigned int *pnLastFvalueOfThisFvalue,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat);

//early
__global__ void PickLocalBestFea(const SplitPoint *pBestSplitPointPerThread, const int *pBuffId,
								 int numofSNode, int numofFea, int maxNumofSplittable,
								 real *pfBestGain, int *pnBestGainKey);
__global__ void PickGlobalBestFea(real *pLastValuePerThread,
							SplitPoint *pBestSplitPointPerThread, nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
							const int *pBuffId, int numofSNode, const real *pfBestGain, const int *pnBestGainKey, int numofBlockPerNode);

//a thread per gain computation
__global__ void ObtainGDEachNode(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId,
								 const real *pFeaValue, const int *pInsIdToNodeId, const real *pGD,
								 const real *pHess, const int *pBuffId, const int *pSNIdToBuffId, int maxNumofSplittable,
								 int numofSNInProgress, int smallestNodeId, int smallestFeaId, int totalNumofFea,
								 int feaBatch, real *pGDOnEachFeaValue, real *pHessOnEachFeaValue,
								 real *pValueOneEachFeaValue);
__global__ void GetInfoEachFeaInBatch(const int *pnNumofKeyValues, const long long *pnFeaStartPos, int smallestFeaId,
									  int totalNumofFea, int feaBatch, int numofSNInProgress, int smallestNodeId,
									  int *pStartPosEachFeaInBatch, int *pnEachFeaLen);
void PrefixSumForEachNode(int feaBatch, real *pGDOnEachFeaValue_d, real *pHessOnEachFeaValue_d,
						  const long long *pnStartPosEachFeaInBatch, const int *pnEachFeaLen, int maxNumValuePerFea, void*pStream);
__global__ void ComputeGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const nodeStat *pSNodeStat, int smallestFeaId, int feaBatch,
							const int *pBuffId, int numofSNInProgress, int smallestNodeId, real lambda,
							const real *pGDPrefixSumOnEachFeaValue, const real *pHessPrefixSumOnEachFeaValue,
							const real *pHessOnEachFeaValue, const real *pFeaValue,
							real *pGainOnEachFeaValue);
__global__ void FixedGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos,
						  int smallestFeaId, int feaBatch, int numofSNode, int smallestNodeId,
						  const real *pHessOnEachFeaValue, const real *pFeaValue,
						  real *pGainOnEachFeaValue, real *pLastBiggerValue);

//helper functions
__device__ bool NeedUpdate(real &RChildHess, real &LChildHess);
__device__ void GetBatchInfo(int feaBatch, int smallestFeaId, int feaId, const int *pnNumofKeyValues, const long long *pnFeaStartPos,
							 int &curFeaStartPosInBatch, int &nFeaValueInBatch);

#endif /* DEVICESPLITTERKERNEL_H_ */
