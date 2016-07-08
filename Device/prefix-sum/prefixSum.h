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

__global__ void cuda_prefixsum(T *in_array, T *out_array, int *arrayStartPos, int numofBlock,
							   unsigned int *pnThreadLastBlock, unsigned int *pnEltsLastBlock);
__global__ void cuda_updatesum(T *array, int *arrayStartPos, T *update_array);

//for testing
void prefixsumForDeviceArray(T *array_d, int *pnArrayStartPos, int *pnEachArrayLen, int numArray);
void prefixsumForHostArray(T *array_h, int *pnArrayStartPos, int size);
int TestPrefixSum(int argc, char *argv[]);

#endif /* PREFIXSUM_H_ */
