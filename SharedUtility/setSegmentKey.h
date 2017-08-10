/*
 * setSegmentKey.h
 *
 *  Created on: Aug 9, 2017
 *      Author: zeyi
 */

#ifndef SETSEGMENTKEY_H_
#define SETSEGMENTKEY_H_

template<class T>
__global__ void SetKey(const uint *pSegStart, const T *pSegLen, uint *pnKey){
	uint segmentId = blockIdx.x;//use one x covering multiple ys, because the maximum number of x-dimension is larger.
	__shared__ uint segmentLen, segmentStartPos;
	if(threadIdx.x == 0){//the first thread loads the segment length
		segmentLen = pSegLen[segmentId];
		segmentStartPos = pSegStart[segmentId];
	}
	__syncthreads();

	uint tid0 = blockIdx.y * blockDim.x;//for supporting multiple blocks for one segment
	uint segmentThreadId = tid0 + threadIdx.x;
	if(tid0 >= segmentLen || segmentThreadId >= segmentLen)
		return;

	uint pos = segmentThreadId;
	while(pos < segmentLen){
		pnKey[pos + segmentStartPos] = segmentId;
		pos += blockDim.x;
	}
}

__global__ void SetKey(const uint segLen, uint *pnKey);


#endif /* SETSEGMENTKEY_H_ */
