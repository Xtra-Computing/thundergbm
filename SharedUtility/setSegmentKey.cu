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
	uint blkId = blockIdx.x;//use one x covering multiple ys, because the maximum number of x-dimension is larger.
	uint segId = blkId * numSegEachBlk;
	for(int i = 0; i < numSegEachBlk; i++){
		__shared__ uint segmentLen, segmentStartPos;
		segId = blkId * numSegEachBlk + i;
		if(segId < totalNumSeg){
			if(threadIdx.x == 0){//the first thread loads the segment length
				segmentLen = pSegLen[segId];
				segmentStartPos = pSegStart[segId];
			}
			__syncthreads();

			uint tid = blockIdx.y * blockDim.x + threadIdx.x;//for supporting multiple blocks for one segment
			uint pos = tid + segmentStartPos;
			while(pos < segmentLen + segmentStartPos){
				//pnKey[pos] = segId;
				asm("st.global.cg.u32 [%1], %0;" :: "r"(segId), "l"(pnKey + pos) : "memory");
				pos += blockDim.x;
			}
		}
		else
			return;
		__syncthreads();
	}
}
