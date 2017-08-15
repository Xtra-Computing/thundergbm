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

__global__ void SetKey(const uint *pSegStart, const int *pSegLen, int numSegEachBlk, uint totalNumSeg, uint *pnKey){
	uint firstSegId = blockIdx.x * numSegEachBlk;//use one x covering multiple ys, because the maximum number of x-dimension is larger.
	const int chunkSize = 64;
	__shared__ uint segmentLen[chunkSize];
	__shared__ uint segmentStartPos[chunkSize];
	uint tid = blockIdx.y * blockDim.x + threadIdx.x;//for supporting multiple blocks for one segment
	int numFullSet = (numSegEachBlk + chunkSize - 1)/chunkSize;

	for(int i = 0; i < numFullSet; i++){
		uint segId = firstSegId + i * chunkSize;
		if(threadIdx.x < chunkSize){//the first thread loads the segment length
			uint segOffset = segId + threadIdx.x;
			bool validLen = (segOffset < totalNumSeg);
			segmentLen[threadIdx.x] = (validLen ? pSegLen[segOffset] : 0);
			segmentStartPos[threadIdx.x] = (validLen ? pSegStart[segOffset] : 0);
		}
		__syncthreads();

		for(int j = 0; j < chunkSize; j++){
			if(segmentLen[j] == 0)
				continue;
			if(segId + j < totalNumSeg){
				uint pos = tid + segmentStartPos[j];
				while(pos < segmentLen[j] + segmentStartPos[j]){
					//pnKey[pos] = segId;
					asm("st.global.cg.u32 [%1], %0;" :: "r"(segId + j), "l"(pnKey + pos) : "memory");
					pos += blockDim.x;
				}
			}
			else
				return;
		}
		__syncthreads();
	}
}
