/*
 * setSegmentKey.cu
 *
 *  Created on: Aug 10, 2017
 *      Author: zeyi
 */

#include "setSegmentKey.h"

__global__ void SetKey(const uint segLen, uint *pnKey){
	uint segmentId = blockIdx.y;//each y corresponding to a segment
	uint innerSegId = blockIdx.x * blockDim.x + threadIdx.x;

	if(innerSegId >= segLen)
		return;

	pnKey[innerSegId + segmentId * segLen] = segmentId;
}
