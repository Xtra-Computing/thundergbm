/*
 * prefixSum.h
 *
 *  Created on: 6 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PREFIXSUM_H_
#define PREFIXSUM_H_

#include "../DeviceHost/DefineConst.h"

//cuda 7.5 does not support template well, so try macro here.
#define T float_point

void prefixsumForDeviceArray(int blocks, int threads, T *array_d, int size);
void prefixsumForHostArray(int blocks, int threads, T *array_h, int size);
int TestPrefixSum(int argc, char *argv[]);

#endif /* PREFIXSUM_H_ */
