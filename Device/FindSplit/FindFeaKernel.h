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


#endif /* DEVICESPLITTERKERNEL_H_ */
