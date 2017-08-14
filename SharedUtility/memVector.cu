/*
 * memVector.cu
 *
 *  Created on: Aug 11, 2017
 *      Author: zeyi
 */

#include <helper_cuda.h>
#include "memVector.h"

void MemVector::reserveSpace(uint newSize, uint numByteEachValue){
	if(addr != NULL){
		checkCudaErrors(cudaFree(addr));
	}
	size = newSize;
	if(newSize * numByteEachValue > (uint)8 * 1024 * 1024 * (uint)1024)//larger than 2GB
		reservedSize = newSize;
	else
		reservedSize = newSize * 1;
	checkCudaErrors(cudaMalloc((void**)&addr, numByteEachValue * reservedSize));
}
