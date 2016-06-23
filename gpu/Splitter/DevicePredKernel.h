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

__global__ void FillDense(float_point *pdSparseInsValue, int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, int *pSortedUsedFea, int *pHashFeaIdToDenseInsPos, int totalDim);


#endif /* DEVICEPREDKERNEL_H_ */
