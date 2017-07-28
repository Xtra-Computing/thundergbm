/*
 * segmentedSum.h
 *
 *  Created on: Jul 28, 2017
 *      Author: zeyi
 */

#ifndef SEGMENTEDSUM_H_
#define SEGMENTEDSUM_H_

template <class T>
__global__ void segmentedSum(const T *eachValueLen, int numValueInSeg, T *eachSegLen){
	unsigned int segId = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int segStartPos = segId * numValueInSeg;
	extern __shared__ T pSegLen[];

	//load to shared memory
	int i = tid;
	T segLen = 0;
	while(i < numValueInSeg){
		segLen += eachValueLen[segStartPos + i];
		i += blockDim.x;
	}
	pSegLen[tid] = segLen;

	//reduction
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(tid < offset) {
			pSegLen[tid] += pSegLen[tid + offset];
		}
		__syncthreads();
	}
	if(tid == 0){
		eachSegLen[segId] = pSegLen[0];
	}
}

#endif /* SEGMENTEDSUM_H_ */
