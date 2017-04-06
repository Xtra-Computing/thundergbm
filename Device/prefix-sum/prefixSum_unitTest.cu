/*
 * testPrefixSum.cu
 *
 *  Created on: 7 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <assert.h>

#include <cuda.h>
#include <helper_cuda.h>
#include "prefixSum.h"
#include "powerOfTwo.h"
#include "../../GetCudaError.h"
#include "../../DeviceHost/MyAssert.h"

using std::cout;
using std::endl;
using std::cerr;

/**
 * @brief: compute block size and the number of blocks
 */
void kernelConf(unsigned int &numBlocks, unsigned int &numThreads, int numElementsLargestArray, int blockSize)
{
    //one thread processes two elements.
    numBlocks = max(1, (int)ceil((float)numElementsLargestArray / (2.f * blockSize)));

    if (numBlocks > 1)
        numThreads = blockSize;
    else
    {
    	smallReductionKernelConf(numThreads, numElementsLargestArray);
    	/*
    	if (isPowerOfTwo(numElementsLargestArray))
    		numThreads = numElementsLargestArray / 2;//only one block and
    	else
    		numThreads = floorPow2(numElementsLargestArray);//a few threads only have one element to process.
    		*/
    }
}

void elementsLastBlock(unsigned int *pnEltsLastBlock, unsigned int *pnThreadLastBlock, unsigned int numBlocks,
					   unsigned int numThreads, const int *pnNumofEltsPerArray, int numArray)
{
	for(int a = 0; a < numArray; a++)
	{
		unsigned int numEltsPerBlock = numThreads * 2;
		// if this is a non-power-of-2 array, the last block will be non-full
		// compute the smallest power of 2 able to compute its scan.
		int numEltsLastBlock;
		if(pnNumofEltsPerArray[a] <= numEltsPerBlock)//one block is enough for this array
			numEltsLastBlock = pnNumofEltsPerArray[a];
		else//multiple blocks are needed
		{
			int tempSubArraySize = pnNumofEltsPerArray[a] - numEltsPerBlock;
			while(tempSubArraySize > numEltsPerBlock)
				tempSubArraySize -= numEltsPerBlock;
			numEltsLastBlock = tempSubArraySize;
		}

		if (isPowerOfTwo(numEltsLastBlock))
			pnThreadLastBlock[a] = numEltsLastBlock / 2;
		else
			pnThreadLastBlock[a] = floorPow2(numEltsLastBlock);

		assert(numEltsLastBlock >= 0);
		pnEltsLastBlock[a] = numEltsLastBlock;
	}
}

/**
  *@brief: compute prefix sum within each block
  */
void prefixSumEachBlock(T *array_d, const long long *pnArrayStartPos_d, const int *pnEachArrayLen_h, int numArray, int numElementsLongestArray,
						unsigned int &numBlkPrescan, dim3 &dim_grid_prescan, dim3 &dim_block_prescan, int &dimZsize, int &dimYsize, T *&out_array_d, unsigned int *&pnEachSubArrayLen_d){
	//the arrays are ordered by their length in ascending order
	int totalNumofEleInArray = 0;
	for(int a = 0; a < numArray; a++)
	{
		totalNumofEleInArray += pnEachArrayLen_h[a];
	}
    unsigned int blockSize = 512; // max size of the thread blocks ############# bugs when 128 for slice_loc.txt (sparse) and YearPredictionMSD (dense)
	//two level scan in case that the array is too large; the first level is called "pre scan"
    unsigned int numThreadsPrescan;//one thread processes two elements.

    //compute kernel configuration
    kernelConf(numBlkPrescan, numThreadsPrescan, numElementsLongestArray, blockSize);//each thread process two elements
    unsigned int numEltsPerBlockPrescan = numThreadsPrescan * 2;//for shared memory allocation

    //when the number of segments (i.e. arrays) is large
    int maxDimSize = 65535;
    dimZsize = max(1, (int)ceil((float)numArray / maxDimSize));
    dimYsize = numArray;
    if(numArray > maxDimSize)
    	dimYsize = 65535;
	dim_grid_prescan = dim3(numBlkPrescan, dimYsize, dimZsize);//numBlkPrescan blocks for an array;
	dim_block_prescan = dim3(numThreadsPrescan, 1, 1);//one grid has one block
	printf("%d blocks for each segment; %d segments in total\n", numBlkPrescan, dimYsize * dimZsize);

    //get info of the last block
    unsigned int *pnEltsLastBlock = new unsigned int[numArray];
    unsigned int *pnEffectiveThreadLastBlock = new unsigned int[numArray];
    elementsLastBlock(pnEltsLastBlock, pnEffectiveThreadLastBlock, numBlkPrescan, numThreadsPrescan, pnEachArrayLen_h, numArray);

	//GPU global memory
	unsigned int *pnEltsLastBlock_d;
	unsigned int *pnThreadLastBlock_d;

	//allocate device memory for prescan 
	cudaMalloc((void **)&out_array_d, numBlkPrescan * numArray * sizeof(T));
	cudaMalloc((void**)&pnEltsLastBlock_d, sizeof(unsigned int) * numArray);
	cudaMalloc((void**)&pnThreadLastBlock_d, sizeof(unsigned int) * numArray);
	cudaMalloc((void**)&pnEachSubArrayLen_d, sizeof(unsigned int) * numArray);

	checkCudaErrors(cudaMemcpy(pnEltsLastBlock_d, pnEltsLastBlock, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pnThreadLastBlock_d, pnEffectiveThreadLastBlock, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pnEachSubArrayLen_d, pnEachArrayLen_h, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice));

	GETERROR("before block level cuda_prefixsum");
	// do prefix sum for each block
	cuda_prefixsum<<<dim_grid_prescan, dim_block_prescan, numEltsPerBlockPrescan * sizeof(T)>>>
			(array_d, totalNumofEleInArray, out_array_d, pnArrayStartPos_d, pnEachSubArrayLen_d, numArray, numBlkPrescan, pnThreadLastBlock_d, pnEltsLastBlock_d);
	cudaDeviceSynchronize();
	GETERROR("after block level cuda_prefixsum");

	delete[] pnEltsLastBlock;
	delete[] pnEffectiveThreadLastBlock;
	cudaFree(pnEltsLastBlock_d);
	cudaFree(pnThreadLastBlock_d);
}

/**
 * @brief: prefix sum for an array in device memory
 * @array_d is used as input and output array.
 */
void prefixsumForDeviceArray(T *array_d, const long long *pnArrayStartPos_d, const int *pnEachArrayLen_h, int numArray, int numElementsLongestArray)
{
	PROCESS_ERROR(array_d != NULL && pnArrayStartPos_d != NULL && pnEachArrayLen_h != NULL && numArray > 0 && numElementsLongestArray > 0);

	//prefix sum for each block
	dim3 dim_grid_prescan, dim_block_prescan;
	int dimZsize, dimYsize;
	T *out_array_d;
    unsigned int numBlkPrescan;
	unsigned int *pnEachSubArrayLen_d;
	prefixSumEachBlock(array_d, pnArrayStartPos_d, pnEachArrayLen_h, numArray, numElementsLongestArray,
					   numBlkPrescan, dim_grid_prescan, dim_block_prescan, dimZsize, dimYsize, out_array_d, pnEachSubArrayLen_d);

	//for block sum
	if(numBlkPrescan > 1){	//more than one block for each array
		unsigned int numBlockForBlockSum = numArray;	//one block for each array for block sum.
		//last block size is the same as the block size; also all blocks have the same # of elts.
		long long *pnBlkSumArrayStartPos_d;		//start position of each set of blocks (i.e. start pos of each segment)
		unsigned int *pnBlockSumEltsLastBlcok_d;//######## can be deleted as it equals to numBlkPreScan
		unsigned int *pnEffectiveThdInBlkSum_d;	//######## can be deleted as it equals to (numBlkPreScan + 1)/2
		unsigned int *pNumBlk_d;	//length of each array//####### can be deleted as it equals to numBlkPrescan

		T *tmp_d;//the result of this variable is not used; it is for satisfying calling the function.
		cudaMalloc((void**)&tmp_d, numBlockForBlockSum * sizeof(T));
		cudaMalloc((void**)&pnBlockSumEltsLastBlcok_d, sizeof(unsigned int) * numBlockForBlockSum);
		cudaMalloc((void**)&pnEffectiveThdInBlkSum_d, sizeof(unsigned int) * numBlockForBlockSum);//all have same # of threads (arrays have same # of blocks in prescan).
		cudaMalloc((void**)&pnBlkSumArrayStartPos_d, sizeof(long long) * numBlockForBlockSum);
		cudaMalloc((void**)&pNumBlk_d, sizeof(unsigned int) * numBlockForBlockSum);

		//compute kernel configuration for prefix sum on among each set of blocks
		int tmpNumThdBlkSum = 0;
		if(isPowerOfTwo(numBlkPrescan))
			tmpNumThdBlkSum = numBlkPrescan / 2;
		else
			tmpNumThdBlkSum = floorPow2(numBlkPrescan);
		if(tmpNumThdBlkSum > 1024){
			cerr << "Bug: the number of thread in a block is " << tmpNumThdBlkSum << " in prefix sum!" << endl;
			exit(0);
		}
		dim3 dim_grid_block_sum(1, dimYsize, dimZsize);//numBlockForBlockSum, 1);//each array only needs one block (i.e. x=1); multiple blocks for multiple arrays.
		dim3 dim_block_block_sum(tmpNumThdBlkSum, 1, 1);
		int numEltsPerBlockForBlockSum = tmpNumThdBlkSum * 2;

		//don't need to get info of the last block, since all of the (last) blocks are the same.
		unsigned int *pnEffectiveThreadForBlockSum = new unsigned int[numBlockForBlockSum];
		unsigned int *pnBlockSumEltsLastBlcok = new unsigned int[numBlockForBlockSum];
		long long *pnBlockSumArrayStartPos = new long long[numBlockForBlockSum];
		unsigned int *pn2ndSubArrayLen_h = new unsigned int[numBlockForBlockSum];
		for(int i = 0; i < numBlockForBlockSum; i++){
			pnEffectiveThreadForBlockSum[i] = tmpNumThdBlkSum;
			pnBlockSumEltsLastBlcok[i] = numBlkPrescan;
			pnBlockSumArrayStartPos[i] = i * numBlkPrescan;//start position of the subarray
			pn2ndSubArrayLen_h[i] = numBlkPrescan;
		}

		cudaMemcpy(pnBlockSumEltsLastBlcok_d, pnBlockSumEltsLastBlcok, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pnEffectiveThdInBlkSum_d, pnEffectiveThreadForBlockSum, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pnBlkSumArrayStartPos_d, pnBlockSumArrayStartPos, sizeof(long long) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pNumBlk_d, pn2ndSubArrayLen_h, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		GETERROR("before second cuda_prefixsum");

		int numofEleInOutArray = numBlkPrescan * numBlockForBlockSum;
		int numofBlockPerSubArray = 1;//only one block for each subarray
		cuda_prefixsum <<<dim_grid_block_sum, dim_block_block_sum, numEltsPerBlockForBlockSum * sizeof(T) >>>(
								out_array_d, numofEleInOutArray, tmp_d, pnBlkSumArrayStartPos_d, pNumBlk_d,
								numArray, numofBlockPerSubArray, pnEffectiveThdInBlkSum_d, pnBlockSumEltsLastBlcok_d);
		cudaDeviceSynchronize();
		GETERROR("in second cuda_prefixsum");

		// update original array using block sum
		//kernel configuration is the same as prescan, since we need to process all the elements
		dim3 dim_grid_updatesum = dim_grid_prescan;
		dim3 dim_block_updatesum = dim_block_prescan;
		if(dim_block_updatesum.y > 1){
			printf("unsupported kernel configuration for cuda_updatesum\n");
			exit(0);
		}
		dim_block_updatesum.y = 2;//different from prescan, one thread processes one element here
		cuda_updatesum<<<dim_grid_updatesum, dim_block_updatesum>>>
				(array_d, pnArrayStartPos_d, pnEachSubArrayLen_d, numArray, out_array_d);

		cudaDeviceSynchronize();
		GETERROR("in cuda_updatesum");

		delete[] pnEffectiveThreadForBlockSum;
		delete[] pnBlockSumEltsLastBlcok;
		delete[] pnBlockSumArrayStartPos;
		delete[] pn2ndSubArrayLen_h;
		cudaFree(tmp_d);
		cudaFree(pnBlockSumEltsLastBlcok_d);
		cudaFree(pnEffectiveThdInBlkSum_d);
		cudaFree(pnBlkSumArrayStartPos_d);
		cudaFree(pNumBlk_d);
	}

	cudaFree(out_array_d);
	cudaFree(pnEachSubArrayLen_d);
}

/**
 * @brief: prefix sum for an array in host memory
 */
//void prefixsumForHostArray(T *array_h, int *pnArrayStartPos, int *pNumofElePerArray, int numArray)
//{
//	T *array_d;
//	int *pnArrayStartPos_d;
//
//	int totalEle = 0;
//	for(int a = 0; a < numArray; a++)
//	{
//		totalEle += pNumofElePerArray[a];
//	}
//	// allocate temp, block sum, and device arrays
//	cudaMalloc((void **)&array_d, totalEle * sizeof(T));
//	cudaMalloc((void**)&pnArrayStartPos_d, sizeof(int) * numArray);
//
//	cudaMemcpy(array_d, array_h, totalEle * sizeof(T), cudaMemcpyHostToDevice);
//	cudaMemcpy(pnArrayStartPos_d, pnArrayStartPos, sizeof(int) * numArray, cudaMemcpyHostToDevice);
//
//  //############ having issues after change position to long long
//	prefixsumForDeviceArray(array_d, pnArrayStartPos_d, pNumofElePerArray, numArray);
//
//	// copy resulting array back to host
//	cudaMemcpy(array_h, array_d, totalEle * sizeof(T), cudaMemcpyDeviceToHost);
//
//	cudaFree(array_d);
//	cudaFree(pnArrayStartPos_d);
//}

///////////////// for testing
void prefixsum_host(T *array_h, int size)
{
	for (int i = 0; i < size; i++) {
		if (i > 0) {
			array_h[i] += array_h[i - 1];
		}
	}
}

void usage(int which)
{
	switch (which) {
	default:
		printf("usage: prefixsum [-h|-b blocks|-t threads] max\n");
		break;
	case 1:
		printf("prefixsum requires numbers <= threads*blocks\n");
		break;
	}
}

void print_array(T *array, int *pnCount, int numArray)
{
	int e = 0;
	for(int a = 0; a < numArray; a++)
	{
		cout << "the " << a << "th array: " << endl;
		for (int i = 0; i < pnCount[a]; i++)
		{
			cout << "i= " << i << " v= " << array[e] << endl;
			e++;
		}
	}
}

void prepare_numbers(T **array, int *pnCount, int numArray)
{
	int totalEle = 0;
	for(int a = 0; a < numArray; a++)
	{
		totalEle += pnCount[a];
	}

	T *numbers = new T[totalEle];

	// load array
	int e = 0;
	for(int a = 0; a < numArray - 1; a++)
	{
		for(int i = 0; i < pnCount[a]; i++)
		{
			numbers[e] = 1;//i + 1.0;
			e++;
		}
	}
	for(int i = 0; i < pnCount[numArray - 1]; i++)
	{
		numbers[e] = 1;
		e++;
	}

	*array = numbers;
}



int TestPrefixSum(int argc, char *argv[])
{
	int opt, host_mode, blocks, threads, max;
	T *array;

	// set options
	host_mode = 0;
	blocks = 1;
	threads = 64;
	while ((opt = getopt(argc, argv, "hd")) != -1) {
		switch (opt) {
		case 'h':
			host_mode = 1;
			break;
		case 'd':
			host_mode = 0;
			break;
		default:
			usage(0);
			return 0;
		}
	}

	// check to make sure we are feeding in correct number of args
	if (argc == optind + 1) {
		max = atoi(argv[optind]);
	} else {
		usage(0);
		return 0;
	}
	// pre-init numbers
	array = NULL;
	int numArray = 16;
	int *pnCount = new int[numArray];
	int *pnArrayStartPos = new int[numArray];

	int currentStartPos = 0;
	for(int i = 0; i < numArray - 2; i++)
	{
		pnArrayStartPos[i] = currentStartPos;
		pnCount[i] = 4176;
		currentStartPos += pnCount[i];
	}


	pnArrayStartPos[numArray - 2] = currentStartPos;
	pnCount[numArray - 2] = max;
	currentStartPos += max;

	pnArrayStartPos[numArray - 1] = currentStartPos;
	pnCount[numArray - 1] = 4177;

	prepare_numbers(&array, pnCount, numArray);

	if (host_mode) {
		printf("prefix sum using host\n");
		prefixsum_host(array, max);
	} else {
		printf("prefix sum using CUDA\n");
//		prefixsumForHostArray(array, pnArrayStartPos, numArray);//deleted each array size on Jul 30 2016
	}

	// print array
	print_array(array, pnCount, numArray);

	free(array);

	return 0;
}
