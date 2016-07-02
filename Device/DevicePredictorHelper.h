/*
 * DevicePredictorHelper.h
 *
 *  Created on: 27 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDICTORHELPER_H_
#define DEVICEPREDICTORHELPER_H_

#include <helper_cuda.h>
#include <cuda.h>
#include "../DeviceHost/DefineConst.h"
#include "../DeviceHost/TreeNode.h"

__global__ void PredTarget(TreeNode *pAllTreeNode, int totalNode, float_point *pDenseIns, int nNumofFea,
						  int *pnHashFeaIdToPos, float_point *pdTargetValue, int maxDepth);
__global__ void PredMultiTarget(float_point *pdTargetValue, int numofDenseIns, const TreeNode *pAllTreeNode,
								const float_point *pDenseIns, int numofFea,
								const int *pnHashFeaIdToPos, int maxDepth);

__global__ void FillDense(const float_point *pdSparseInsValue, const int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, const int *pSortedUsedFea, const int *pHashFeaIdToDenseInsPos, int totalDim);
__global__ void FillMultiDense(const float_point *pdSparseInsValue, const long long *pInsStartPos, const int *pnSpareInsFeaId,
							   const int *pNumofFeaValue, float_point *pdDenseIns, const int *pSortedUsedFea,
							   const int *pHashFeaIdToDenseInsPos, int numofUsedFea,
						  	   int startInsId, int numofInsToFill);


#endif /* DEVICEPREDICTORHELPER_H_ */
