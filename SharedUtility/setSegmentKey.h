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

//	if(segmentId > 0 && threadIdx.x == 0|| blockIdx.y * blockDim.x + threadIdx.x + pSegStart[segmentId] == 8479362)
//		printf("????????????????? length=%d, segId=%d, tid0=%d, blockDim.x=%d, startPos=%d\n", segmentLen, segmentId, tid0, blockDim.x, segmentStartPos);

	if(tid0 >= segmentLen)
		return;
	uint segmentThreadId = tid0 + threadIdx.x;

	uint pos = segmentThreadId + segmentStartPos;

	while(pos < segmentLen + segmentStartPos){
		pnKey[pos] = segmentId;
//		if(pos == 8479362)// && segmentLen == 5999)
//			printf("????????????????????????????? segId=%d, segLen=%d, key=%d\n", segmentId, segmentLen, pnKey[pos]);
		//asm("st.global.cg.u32 [%1], %0;" :: "=r"(segmentId), "l"(pnKey + pos + segmentStartPos) : "memory");
		pos += blockDim.x;
	}
}

template<class T>
__global__ void SetKey(const uint *pSegStart, const T *pSegLen, int numSegEachBlk, uint totalNumSeg, uint *pnKey){
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

__global__ void SetKey(const uint segLen, uint *pnKey);


#endif /* SETSEGMENTKEY_H_ */
