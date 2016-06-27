/*
 * DevicePredictorHelper.h
 *
 *  Created on: 27 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDICTORHELPER_H_
#define DEVICEPREDICTORHELPER_H_

__global__ void PredTarget(TreeNode *pAllTreeNode, int totalNode, float_point *pDenseIns, int nNumofFea,
						  int *pnHashFeaIdToPos, float_point *pdTargetValue);

__global__ void FillDense(float_point *pdSparseInsValue, int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, int *pSortedUsedFea, int *pHashFeaIdToDenseInsPos, int totalDim);



#endif /* DEVICEPREDICTORHELPER_H_ */
