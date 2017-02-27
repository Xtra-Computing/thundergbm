/*
 * DevicePredKernel.h
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDKERNEL_H_
#define DEVICEPREDKERNEL_H_

#include "../../DeviceHost/DefineConst.h"
#include "../../DeviceHost/TreeNode.h"
#include "../../DeviceHost/NodeStat.h"

__global__ void SaveToPredBuffer(const float_point *pfCurTreePredValue, int numInsToPred, float_point *pfPreTreePredValue);

__global__ void ComputeGDKernel(int numofIns, const float_point *pfPredValue, const float_point *pfTrueValue,
								float_point *pGrad, float_point *pHess);
__global__ void InitNodeStat(const float_point *root_sum_gd, const float_point *root_sum_hess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable,
							 int *pBuffId, int *pNumofBuffId);
__global__ void InitNodeStat(int numofIns, const float_point *pGrad, const float_point *pHess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable, int *pBuffId, int *pNumofBuffId);
__global__ void InitRootNode(TreeNode *pAllTreeNode, int *pCurNumofNode, int numofIns);

#endif /* DEVICEPREDKERNEL_H_ */
