/*
 * DevicePrediction.h
 *
 *  Created on: 23 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDICTION_H_
#define DEVICEPREDICTION_H_

__global__ void PredTarget(TreeNode *pAllTreeNode, int totalNode, float_point *pDenseIns, int nNumofFea,
						  int *pnHashFeaIdToPos, float_point *pdTargetValue);

__global__ void FillDense(float_point *pdSparseInsValue, int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, int *pSortedUsedFea, int *pHashFeaIdToDenseInsPos, int totalDim);

#endif /* DEVICEPREDICTION_H_ */
