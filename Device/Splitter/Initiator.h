/*
 * DevicePredKernel.h
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDKERNEL_H_
#define DEVICEPREDKERNEL_H_

#include "../../SharedUtility/DataType.h"
#include "../../DeviceHost/TreeNode.h"
#include "../../DeviceHost/NodeStat.h"

__global__ void SaveToPredBuffer(const real *pfCurTreePredValue, int numInsToPred, real *pfPreTreePredValue);

__global__ void ComputeGDKernel(int numofIns, const real *pfPredValue, const real *pfTrueValue,
								real *pGrad, real *pHess);
__global__ void InitNodeStat(const real root_sum_gd, const real root_sum_hess,
							 nodeStat *pSNodeStat, int *pSNIdToBuffId, int maxNumofSplittable,
							 int *pBuffId, int *pNumofBuffId);
__global__ void InitRootNode(TreeNode *pAllTreeNode, int *pCurNumofNode, int numofIns);

#endif /* DEVICEPREDKERNEL_H_ */
