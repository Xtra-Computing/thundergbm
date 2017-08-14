/*
 * setSegmentKey.h
 *
 *  Created on: Aug 9, 2017
 *      Author: zeyi
 */

#ifndef SETSEGMENTKEY_H_
#define SETSEGMENTKEY_H_

#include <stdio.h>

template<class T>
__global__ void SetKey(const uint *pSegStart, const T *pSegLen, uint *pnKey){
	uint segmentId = blockIdx.x;//use one x covering multiple ys, because the maximum number of x-dimension is larger.
	__shared__ uint segmentLen, segmentStartPos;
	if(threadIdx.x == 0){//the first thread loads the segment length
		segmentLen = pSegLen[segmentId];
	}
	__syncthreads();
	if(segmentLen == 0)
		return;
	if(threadIdx.x == 0)
		segmentStartPos = pSegStart[segmentId];
	__syncthreads();

	uint tid0 = blockIdx.y * blockDim.x;//for supporting multiple blocks for one segment
	if(tid0 >= segmentLen)
		return;
	uint segmentThreadId = tid0 + threadIdx.x;

	uint pos = segmentThreadId + segmentStartPos;
	while(pos < segmentLen + segmentStartPos){
		pnKey[pos] = segmentId;
		//asm("st.global.cg.u32 [%1], %0;" :: "=r"(segmentId), "l"(pnKey + pos + segmentStartPos) : "memory");
		pos += blockDim.x;
	}
}

__global__ void SetKey(const uint *pSegStart, const int *pSegLen, int numSegEachBlk, uint totalNumSeg, uint *pnKey);

__global__ void SetKey(const uint segLen, uint *pnKey);


#endif /* SETSEGMENTKEY_H_ */
