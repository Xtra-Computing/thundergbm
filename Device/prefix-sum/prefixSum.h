/*
 * prefixSum.h
 *
 *  Created on: 6 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PREFIXSUM_H_
#define PREFIXSUM_H_

#include "../../DeviceHost/DefineConst.h"

//cuda 7.5 does not support template well, so try macro here.
#define T float_point

__global__ void cuda_prefixsum(T *in_array, int in_array_size, T *out_array, const int *arrayStartPos, int numofBlockPerSubArray,
							   unsigned int *pnThreadLastBlock, unsigned int *pnEltsLastBlock);
__global__ void cuda_updatesum(T *array, const int *arrayStartPos, T *update_array);

//for testing
void prefixsumForDeviceArray(T *array_d, const int *pnArrayStartPos_d, const int *pnEachArrayLen_h, int numArray);
void prefixsumForHostArray(T *array_h, int *pnArrayStartPos, int size);
int TestPrefixSum(int argc, char *argv[]);

#endif /* PREFIXSUM_H_ */
