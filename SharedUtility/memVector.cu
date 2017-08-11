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
	reservedSize = newSize * 1.5;
	checkCudaErrors(cudaMalloc((void**)&addr, numByteEachValue * reservedSize));
}
