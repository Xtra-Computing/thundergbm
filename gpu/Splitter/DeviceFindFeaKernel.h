/*
 * DeviceSplitterKernel.h
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTERKERNEL_H_
#define DEVICESPLITTERKERNEL_H_

#include "../../pureHost/UpdateOps/NodeStat.h"
#include "../../pureHost/UpdateOps/SplitPoint.h"
#include "../../pureHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/DefineConst.h"

__global__ void FindFeaSplitValue(int nNumofKeyValues, int *idStartAddress, float_point *pValueStartAddress, int *pInsIdToNodeId,
								  nodeStat *pTempRChildStat, float_point *pGD, float_point *pHess, float_point *pLastValue,
								  nodeStat *pSNodeState, SplitPoint *pBestSplitPoin, nodeStat *pRChildStat, nodeStat *pLChildStat,
								  int *pSNIdToBuffId, int maxNumofSplittable, int featureId, int *pBuffId, int numofSNode, float_point lambda);

__device__ double CalGain(const nodeStat &parent, const nodeStat &r_child, float_point &l_child_GD,
									 float_point &l_child_Hess, float_point &lambda);

__device__ bool UpdateSplitPoint(SplitPoint &curBest, double fGain, double fSplitValue, int nFeatureId);

__device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat, nodeStat &TempRChildStat,
										float_point &grad, float_point &hess);
__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);
__device__ void UpdateSplitInfo(nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
										 nodeStat &TempRChildStat, float_point &tempGD, float_point &temHess,
										 float_point &lambda, float_point &sv, int &featureId);

//has an identical verion in host
__device__ int GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);

#endif /* DEVICESPLITTERKERNEL_H_ */
