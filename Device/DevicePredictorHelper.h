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
#include "../SharedUtility/DataType.h"
#include "../DeviceHost/TreeNode.h"

__global__ void PredTarget(TreeNode *pAllTreeNode, int totalNode, real *pDenseIns, int nNumofFea,
						  int *pnHashFeaIdToPos, real *pdTargetValue, int maxDepth);
__global__ void PredMultiTarget(real *pdTargetValue, int numofDenseIns, const TreeNode *pAllTreeNode,
								const real *pDenseIns, int numofFea,
								const int *pnHashFeaIdToPos, int maxDepth);

__global__ void FillMultiDense(const real *pdSparseInsValue, const uint *pInsStartPos, const int *pnSpareInsFeaId,
							   const int *pNumofFeaValue, real *pdDenseIns, const int *pSortedUsedFea,
							   const int *pHashFeaIdToDenseInsPos, int numofUsedFea,
						  	   int startInsId, int numofInsToFill);


#endif /* DEVICEPREDICTORHELPER_H_ */
