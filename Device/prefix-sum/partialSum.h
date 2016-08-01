/*
 * partialSum.h
 *
 *  Created on: 1 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PARTIALSUM_H_
#define PARTIALSUM_H_

#include <helper_cuda.h>
#include <cuda.h>
#include "../../DeviceHost/DefineConst.h"

__global__ void blockSum(float_point * input, float_point * output, int len, bool isFinalSum);



#endif /* PARTIALSUM_H_ */
