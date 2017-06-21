/*
 * segmentedMax.h
 *
 *  Created on: Jun 21, 2017
 *      Author: zeyi
 */

#ifndef SEGMENTEDMAX_H_
#define SEGMENTEDMAX_H_

#include <limits>
#include <float.h>
#include <helper_cuda.h>
#include "getMin.h"
#include "DeviceUtility.h"
#include "DataType.h"
#include "CudaMacro.h"
#include "KernelConf.h"

template<class T>
__global__ void LocalReductionEachSeg(const uint *pEachSegSize, const uint *pEachSegStartPos,
										   const real *pValueAllSeg, real *pLocalMax, T *pLocalMaxKey)
{
	//best gain of each node is search by a few blocks
	//blockIdx.z corresponds to a splittable node id
	int snId = blockIdx.z;
	uint numValueThisNode = pEachSegSize[snId];//get the number of feature value of this node
	int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
	uint tid0 = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
	if(tid0 >= numValueThisNode){
		pLocalMax[blockId] = 0;
		pLocalMaxKey[blockId] = tid0;
		return;
	}

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;
	if(localTid == 0){//initialise local best value
		pLocalMax[blockId] = FLT_MAX;
		pLocalMaxKey[blockId] = -1;
	}

	uint tidForEachNode = tid0 + threadIdx.x;
	uint nPos = pEachSegStartPos[snId] + tidForEachNode;//feature value gain position


	if(tidForEachNode >= numValueThisNode){//no gain to load
		pfGain[localTid] = 0;
		pnBetterGainKey[localTid] = INT_MAX;
	}
	else{
		pfGain[localTid] = -pValueAllSeg[nPos];//change to find min of -gain
		pnBetterGainKey[localTid] = nPos;
	}
	__syncthreads();

	//find the local best split point
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pLocalMax[blockId] = pfGain[0];
		pLocalMaxKey[blockId] = pnBetterGainKey[0];
		ECHECKER(pnBetterGainKey[0]);
	}
}
template<class T>
__global__ void GlobalReductionEachSeg(const real *pLocalMax, const T *pLocalMaxKey,
								   	   real *pGlobalMax, T *pGlobalMaxKey, int numBlockPerSeg, int numSeg)
{
	//a block for finding the best gain of a node
	int blockId = blockIdx.x;
	int snId = blockId;
	CONCHECKER(blockIdx.y <= 1);
	CONCHECKER(snId < numSeg);

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ T pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = (T)-1;

	if(localTid >= numBlockPerSeg)//number of threads is larger than the number of blocks
		return;

	int curFeaLocalBestStartPos = snId * numBlockPerSeg;
	LoadToSharedMem(numBlockPerSeg, curFeaLocalBestStartPos, pLocalMax, pLocalMaxKey, pfGain, pnBetterGainKey);
	 __syncthreads();	//wait until the thread within the block

	//find the local best split point
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pGlobalMax[snId] = -pfGain[0];//make the gain back to its original sign
		pGlobalMaxKey[snId] = pnBetterGainKey[0];
	}
}

template<class T>
void SegmentedMax(int sizeofLargestSeg, int numSeg, const uint *pEachSegSize, const uint *pEachSegStartPos,
				  const real *pValueAllSeg, void *pStream, real *pEachSegMax, T *pEachSegMaxKey){
	cudaStream_t tempStream = *(cudaStream_t*)pStream;
	real *pLocalMax;
	T *pLocalMaxKey;
	//compute # of blocks for each segment
	PROCESS_ERROR(sizeofLargestSeg > 0);
	int blockSizeLocalReduction;
	dim3 dimNumofBlockLocalReduction;
	KernelConf conf;
	conf.ConfKernel(sizeofLargestSeg, blockSizeLocalReduction, dimNumofBlockLocalReduction);
	PROCESS_ERROR(dimNumofBlockLocalReduction.z == 1);
	dimNumofBlockLocalReduction.z = numSeg;	//each node per super block
	int numBlockPerSeg = dimNumofBlockLocalReduction.x * dimNumofBlockLocalReduction.y;

	checkCudaErrors(cudaMalloc((void**)&pLocalMax, sizeof(real) * numBlockPerSeg * numSeg));
	checkCudaErrors(cudaMalloc((void**)&pLocalMaxKey, sizeof(T) * numBlockPerSeg * numSeg));
	//find the block level best gain for each node
	LocalReductionEachSeg<<<dimNumofBlockLocalReduction, blockSizeLocalReduction, 0, tempStream>>>(
									pEachSegSize, pEachSegStartPos, pValueAllSeg, pLocalMax, pLocalMaxKey);
	cudaStreamSynchronize(tempStream);
	GETERROR("after PickLocalBestSplitEachNode");

	//find the global best gain for each node
	if(numBlockPerSeg > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerSeg, blockSizeBestGain, dimNumofBlockDummy);
		if(blockSizeBestGain < 64)//make sure the reduction is power of two
			blockSizeBestGain = 64;
		GlobalReductionEachSeg<<<numSeg, blockSizeBestGain, 0, tempStream>>>(
										pLocalMax, pLocalMaxKey, pEachSegMax, pEachSegMaxKey, numBlockPerSeg, numSeg);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		GETERROR("after PickGlobalBestSplitEachNode");
	}
	else{//local best fea is the global best fea
		checkCudaErrors(cudaMemcpyAsync(pLocalMax, pEachSegMax, sizeof(real) * numSeg, cudaMemcpyDeviceToDevice, tempStream));
		checkCudaErrors(cudaMemcpyAsync(pLocalMaxKey, pEachSegMaxKey, sizeof(T) * numSeg, cudaMemcpyDeviceToDevice, tempStream));
	}
	checkCudaErrors(cudaFree(pLocalMax));
	checkCudaErrors(cudaFree(pLocalMaxKey));
}

#endif /* SEGMENTEDMAX_H_ */
