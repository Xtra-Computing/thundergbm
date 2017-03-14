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
 * @brief: prefix sum for an array in device memory
 */
void prefixsumForDeviceArray(T *array_d, const long long *pnArrayStartPos_d, const int *pnEachArrayLen_h, int numArray, int numElementsLongestArray)
{
	PROCESS_ERROR(array_d != NULL && pnArrayStartPos_d != NULL && pnEachArrayLen_h != NULL && numArray > 0 && numElementsLongestArray > 0);
	//the arrays are ordered by their length in ascending order
	int totalNumofEleInArray = 0;
	for(int a = 0; a < numArray; a++)
	{
		totalNumofEleInArray += pnEachArrayLen_h[a];
	}
    unsigned int blockSize = 256; // max size of the thread blocks ############# bugs when 128 for slice_loc.txt (sparse) and YearPredictionMSD (dense)
    unsigned int numBlocksPrescan;
    unsigned int numThreadsPrescan;//one thread processes two elements.

    //compute kernel configuration
    kernelConf(numBlocksPrescan, numThreadsPrescan, numElementsLongestArray, blockSize);//each thread process two elements
    //when the number of array is large
    int maxDimSize = 65535;
    int dimZsize = max(1, (int)ceil((float)numArray / maxDimSize));
    int dimYsize = numArray;
    if(numArray > maxDimSize)
    	dimYsize = 65535;
	dim3 dim_grid_prescan(numBlocksPrescan, dimYsize, dimZsize);
	dim3 dim_block_prescan(numThreadsPrescan, 1, 1);
    unsigned int numEltsPerBlockPrescan = numThreadsPrescan * 2;//for shared memory allocation

    //get info of the last block
    unsigned int *pnEltsLastBlock = new unsigned int[numArray];
    unsigned int *pnEffectiveThreadLastBlock = new unsigned int[numArray];
    elementsLastBlock(pnEltsLastBlock, pnEffectiveThreadLastBlock, numBlocksPrescan, numThreadsPrescan, pnEachArrayLen_h, numArray);

	T *out_array_d;
	unsigned int *pnEltsLastBlock_d;
	unsigned int *pnThreadLastBlock_d;
	unsigned int *pnEachSubArrayLen_d;

	//for prescan allocate temp, block sum, and device arrays
	cudaMalloc((void **)&out_array_d, numBlocksPrescan * numArray * sizeof(T));
	cudaMalloc((void**)&pnEltsLastBlock_d, sizeof(unsigned int) * numArray);
	cudaMalloc((void**)&pnThreadLastBlock_d, sizeof(unsigned int) * numArray);
	cudaMalloc((void**)&pnEachSubArrayLen_d, sizeof(unsigned int) * numArray);

	cudaMemcpy(pnEltsLastBlock_d, pnEltsLastBlock, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice);
	cudaMemcpy(pnThreadLastBlock_d, pnEffectiveThreadLastBlock, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice);
	cudaMemcpy(pnEachSubArrayLen_d, pnEachArrayLen_h, sizeof(unsigned int) * numArray, cudaMemcpyHostToDevice);


	// do prefix sum for each block
	cuda_prefixsum <<< dim_grid_prescan, dim_block_prescan, numEltsPerBlockPrescan * sizeof(T) >>>
			(array_d, totalNumofEleInArray, out_array_d, pnArrayStartPos_d, pnEachSubArrayLen_d, numArray, numBlocksPrescan, pnThreadLastBlock_d, pnEltsLastBlock_d);
	cudaDeviceSynchronize();
	GETERROR("first cuda_prefixsum");

	//for block sum
	if(numBlocksPrescan > 1)//number of blocks for each array
	{
		T *tmp_d;//the result of this variable is not used; it is for satisfying calling the function.
		unsigned int numBlockForBlockSum = numArray;//one array may have one or more blocks in prescan; one block for each array for block sum.
		unsigned int *pnBlockSumEltsLastBlcok_d;//last block size is the same as the block size, since only one block for each array.
		unsigned int *pnThreadBlockSum_d;
		long long *pnBlockSumArrayStartPos_d;
		unsigned int *pn2ndSubArrayLen_d;//sub array len; for satisfying the function called later.
		cudaMalloc((void **)&tmp_d, numBlockForBlockSum * sizeof(T));
		cudaMalloc((void**)&pnBlockSumEltsLastBlcok_d, sizeof(unsigned int) * numBlockForBlockSum);
		cudaMalloc((void**)&pnThreadBlockSum_d, sizeof(unsigned int) * numBlockForBlockSum);//all have same # of threads (arrays have same # of blocks in prescan).
		cudaMalloc((void**)&pnBlockSumArrayStartPos_d, sizeof(long long) * numBlockForBlockSum);
		cudaMalloc((void**)&pn2ndSubArrayLen_d, sizeof(unsigned int) * numBlockForBlockSum);

		// do prefix sum for block sum
		//compute kernel configuration
		int temNumThreadBlockSum = 0;
		if (isPowerOfTwo(numBlocksPrescan))
			temNumThreadBlockSum = numBlocksPrescan / 2;
		else
			temNumThreadBlockSum = floorPow2(numBlocksPrescan);
		if(temNumThreadBlockSum > 1024)
		{
			cerr << "Bug: the number of thread in a block is " << temNumThreadBlockSum << " in prefix sum!" << endl;
			exit(0);
		}
		dim3 dim_grid_block_sum(1, dimYsize, dimZsize);//numBlockForBlockSum, 1);//each array only needs one block (i.e. x=1); multiple blocks for multiple arrays.
		dim3 dim_block_block_sum(temNumThreadBlockSum, 1, 1);
		int numEltsPerBlockForBlockSum = temNumThreadBlockSum * 2;

		//don't need to get info of the last block, since all of the (last) blocks are the same.
		unsigned int *pnEffectiveThreadForBlockSum = new unsigned int[numBlockForBlockSum];
		unsigned int *pnBlockSumEltsLastBlcok = new unsigned int[numBlockForBlockSum];
		long long *pnBlockSumArrayStartPos = new long long[numBlockForBlockSum];
		unsigned int *pn2ndSubArrayLen_h = new unsigned int[numBlockForBlockSum];
		for(int i = 0; i < numBlockForBlockSum; i++)
		{
			pnEffectiveThreadForBlockSum[i] = temNumThreadBlockSum;
			pnBlockSumEltsLastBlcok[i] = numBlocksPrescan;
			pnBlockSumArrayStartPos[i] = i * numBlocksPrescan;//start position of the subarray
			pn2ndSubArrayLen_h[i] = numBlocksPrescan;
		}

		cudaMemcpy(pnBlockSumEltsLastBlcok_d, pnBlockSumEltsLastBlcok, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pnThreadBlockSum_d, pnEffectiveThreadForBlockSum, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pnBlockSumArrayStartPos_d, pnBlockSumArrayStartPos, sizeof(long long) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		cudaMemcpy(pn2ndSubArrayLen_d, pn2ndSubArrayLen_h, sizeof(unsigned int) * numBlockForBlockSum, cudaMemcpyHostToDevice);
		if(cudaGetLastError() != cudaSuccess)
		{
			cout << "error in before second cuda_prefixsum" << endl;
			exit(0);
		}

		int numofEleInOutArray = numBlocksPrescan * numBlockForBlockSum;
		int numofBlockPerSubArray = 1;//only one block for each subarray
		cuda_prefixsum <<<dim_grid_block_sum, dim_block_block_sum, numEltsPerBlockForBlockSum * sizeof(T) >>>(
								out_array_d, numofEleInOutArray, tmp_d, pnBlockSumArrayStartPos_d, pn2ndSubArrayLen_d,
								numArray, numofBlockPerSubArray, pnThreadBlockSum_d, pnBlockSumEltsLastBlcok_d);
		cudaDeviceSynchronize();
		if(cudaGetLastError() != cudaSuccess)
		{
			cout << "error in second cuda_prefixsum" << endl;
			exit(0);
		}

		// update original array using block sum
		//kernel configuration is the same as prescan, since we need to process all the elements
		dim3 dim_grid_updatesum = dim_grid_prescan;
		dim3 dim_block_updatesum = dim_block_prescan;
		dim_block_updatesum.x *= 2;//different from prescan, one thread processes one element here
		cuda_updatesum <<<dim_grid_updatesum, dim_block_updatesum>>>
				(array_d, pnArrayStartPos_d, pnEachSubArrayLen_d, numArray, out_array_d);

		cudaDeviceSynchronize();
		if(cudaGetLastError() != cudaSuccess)
		{
			cout << "error in cuda_updatesum" << endl;
			exit(0);
		}

		delete[] pnEffectiveThreadForBlockSum;
		delete[] pnBlockSumEltsLastBlcok;
		delete[] pnBlockSumArrayStartPos;
		delete[] pn2ndSubArrayLen_h;
		cudaFree(tmp_d);
		cudaFree(pnBlockSumEltsLastBlcok_d);
		cudaFree(pnThreadBlockSum_d);
		cudaFree(pnBlockSumArrayStartPos_d);
		cudaFree(pn2ndSubArrayLen_d);
	}

	delete[] pnEltsLastBlock;
	delete[] pnEffectiveThreadLastBlock;

	cudaFree(out_array_d);
	cudaFree(pnEltsLastBlock_d);
	cudaFree(pnThreadLastBlock_d);
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
