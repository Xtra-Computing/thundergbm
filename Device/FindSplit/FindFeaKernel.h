/*
 * DeviceSplitterKernel.h
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTERKERNEL_H_
#define DEVICESPLITTERKERNEL_H_

#include "../../Host/UpdateOps/NodeStat.h"
#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/DefineConst.h"

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

//helper functions on device
__device__ double CalGain(const nodeStat &parent, const nodeStat &r_child,
						  const float_point &l_child_GD, const float_point &l_child_Hess, const float_point &lambda);

__device__ bool UpdateSplitPoint(SplitPoint &curBest, double fGain, double fSplitValue, int nFeatureId);

__device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat,
							 const nodeStat &TempRChildStat, const float_point &grad, const float_point &hess);
__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);
__device__ void UpdateSplitInfo(const nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
								const nodeStat &TempRChildStat, const float_point &tempGD, const float_point &temHess,
								const float_point &lambda, const float_point &sv, const int &featureId);


#endif /* DEVICESPLITTERKERNEL_H_ */
