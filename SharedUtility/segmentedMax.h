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

#include <unistd.h>

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
		pLocalMaxKey[blockId] = LARGE_4B_UINT;
	}
	else{
		__shared__ real pfGain[BLOCK_SIZE];
		__shared__ int pnBetterGainKey[BLOCK_SIZE];
		int tid = threadIdx.x;
		pfGain[tid] = FLT_MAX;//initialise to a large positive number
		pnBetterGainKey[tid] = -1;
		if(tid == 0){//initialise local best value
			pLocalMax[blockId] = FLT_MAX;
			pLocalMaxKey[blockId] = LARGE_4B_UINT;
		}

		uint tidForEachNode = tid0 + threadIdx.x;
		uint nPos = pEachSegStartPos[snId] + tidForEachNode;//feature value gain position

		if(tidForEachNode >= numValueThisNode){//no gain to load
			pfGain[tid] = 0;
			pnBetterGainKey[tid] = INT_MAX;
		}
		else{
			pfGain[tid] = -pValueAllSeg[nPos];//change to find min of -gain
			pnBetterGainKey[tid] = nPos;
		}
		__syncthreads();

		//find the local best split point
		GetMinValueOriginal(pfGain, pnBetterGainKey);
		__syncthreads();
		if(tid == 0)//copy the best gain to global memory
		{
			pLocalMax[blockId] = pfGain[0];
			pLocalMaxKey[blockId] = pnBetterGainKey[0];
			ECHECKER(pnBetterGainKey[0]);
		}
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
	int tid = threadIdx.x;
	pfGain[tid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[tid] = (T)-1;
	__syncthreads();

	//if(tid < numBlockPerSeg){//number of threads is smaller than the number of blocks
	int curFeaLocalBestStartPos = snId * numBlockPerSeg;
	LoadToSharedMem(numBlockPerSeg, curFeaLocalBestStartPos, pLocalMax, pLocalMaxKey, pfGain, pnBetterGainKey);

	 __syncthreads();	//wait until the thread within the block

	//find the local best split point
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(tid == 0)//copy the best gain to global memory
	{
		pGlobalMax[snId] = -pfGain[0];//make the gain back to its original sign
		pGlobalMaxKey[snId] = pnBetterGainKey[0];
	}
}

template<class T>
void SegmentedMax(int sizeofLargestSeg, int numNode, const uint *pEachSegSize, const uint *pEachSegStartPos,
				  const real *pValueAllSeg, void *pStream, real *pEachNodeMax, T *pEachSegMaxKey){
	real *pLocalMax;
	T *pLocalMaxKey;
	//compute # of blocks for each segment
	PROCESS_ERROR(sizeofLargestSeg > 0);
	int blockSizeLocalReduction;
	dim3 dimNumofBlockLocalReduction;
	KernelConf conf;
	conf.ConfKernel(sizeofLargestSeg, blockSizeLocalReduction, dimNumofBlockLocalReduction);
	PROCESS_ERROR(dimNumofBlockLocalReduction.z == 1);
	dimNumofBlockLocalReduction.z = numNode;	//each node per super block
	int numBlockPerSeg = dimNumofBlockLocalReduction.x * dimNumofBlockLocalReduction.y;

	checkCudaErrors(cudaMalloc((void**)&pLocalMax, sizeof(real) * numBlockPerSeg * numNode));
	checkCudaErrors(cudaMalloc((void**)&pLocalMaxKey, sizeof(T) * numBlockPerSeg * numNode));
	checkCudaErrors(cudaMemset(pLocalMaxKey, -1, sizeof(T) * numBlockPerSeg * numNode));
	cudaDeviceSynchronize();
	//find the block level best gain for each node
	LocalReductionEachSeg<<<dimNumofBlockLocalReduction, blockSizeLocalReduction>>>(
									pEachSegSize, pEachSegStartPos, pValueAllSeg, pLocalMax, pLocalMaxKey);
	GETERROR("after PickLocalBestSplitEachNode");

	cudaDeviceSynchronize();
	real *test_d = thrust::min_element(thrust::device, pLocalMax, pLocalMax + numBlockPerSeg * numNode);
	real test;
	checkCudaErrors(cudaMemcpy(&test, test_d, sizeof(real), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
//	printf("max in segmented max=%f; blkPerSeg=%d, numSeg=%d, largestSeg=%d, blksize=%d\n", test, numBlockPerSeg, numNode, sizeofLargestSeg, blockSizeLocalReduction);

	//find the global best gain for each node
	if(numBlockPerSeg > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerSeg, blockSizeBestGain, dimNumofBlockDummy);
		if(blockSizeBestGain < 64)//make sure the reduction is power of two
			blockSizeBestGain = 64;
		GlobalReductionEachSeg<<<numNode, blockSizeBestGain>>>(
				pLocalMax, pLocalMaxKey, pEachNodeMax, pEachSegMaxKey, numBlockPerSeg, numNode);
		cudaDeviceSynchronize();
		GETERROR("after PickGlobalBestSplitEachNode");
	}
	else{//local best fea is the global best fea
		checkCudaErrors(cudaMemcpy(pEachNodeMax, pLocalMax, sizeof(real) * numNode, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(pEachSegMaxKey, pLocalMaxKey, sizeof(T) * numNode, cudaMemcpyDeviceToDevice));
	}

	cudaDeviceSynchronize();
	real *globalMax_end = new real[numNode];
	checkCudaErrors(cudaMemcpy(globalMax_end, pEachNodeMax, sizeof(real) * numNode, cudaMemcpyDeviceToHost));
	T *g_keys = new T[numNode];
	checkCudaErrors(cudaMemcpy(g_keys, pEachSegMaxKey, sizeof(T) * numNode, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	real finalMax = 0;
	T finalKey = -1;
	for(int i = 0; i < numNode; i++){
		if(finalMax < globalMax_end[i]){
			finalMax = globalMax_end[i];
			finalKey = g_keys[i];
		}
	}
//	printf("finalMax=%f, finalKey=%d\n", finalMax, finalKey);

	checkCudaErrors(cudaFree(pLocalMax));
	checkCudaErrors(cudaFree(pLocalMaxKey));
}

#endif /* SEGMENTEDMAX_H_ */
