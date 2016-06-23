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



#endif /* DEVICEPREDICTION_H_ */
