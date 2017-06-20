/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "IndexComputer.h"
#include "FindFeaKernel.h"
#include "../Hashing.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/HostUtility.h"
#include "../../SharedUtility/powerOfTwo.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;

__global__ void SetKey(uint *pSegStart, int *pSegLen, uint *pnKey){
	uint segmentId = blockIdx.x;//use one x covering multiple ys, because the maximum number of x-dimension is larger.
	__shared__ uint segmentLen, segmentStartPos;
	if(threadIdx.x == 0){//the first thread loads the segment length
		segmentLen = pSegLen[segmentId];
		segmentStartPos = pSegStart[segmentId];
	}
	__syncthreads();

	uint tid0 = blockIdx.y * blockDim.x;
	uint segmentThreadId = tid0 + threadIdx.x;
	if(tid0 >= segmentLen || segmentThreadId >= segmentLen)
		return;

	uint pos = segmentThreadId;
	while(pos < segmentLen){
		pnKey[pos + segmentStartPos] = segmentId;
		pos += blockDim.x;
	}
}

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//reset memory for this bag
	{
		manager.MemsetAsync(bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);

		manager.MemsetAsync(bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(double) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute index for each feature value
	KernelConf conf;
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(bagManager.m_numFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	//# of feature values that need to compute gains; the code below cannot be replaced by indexComp.m_totalNumFeaValue, due to some nodes becoming leaves.
	int numofDenseValue = -1, maxNumFeaValueOneNode = -1;
	if(numofSNode > 1)
	{
		IndexComputer indexComp;
		indexComp.AllocMem(bagManager.m_numFea, numofSNode);
		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t comIdx_start = clock();
		//compute gather index via GPUs
		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, bagId);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		//copy # of feature values of each node
		uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
	
		clock_t start_gd = clock();
		//scatter operation
		//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
		numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_numIns, manager.m_pDInsId, manager.m_pdDFeaValue,
															   bagManager.m_pIndicesEachBag_d, numofDenseValue,
															   bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   	   	manager.m_pDInsId, manager.m_pdDFeaValue, bagManager.m_numFeaValue,
															   		bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//copy # of feature values of a node
		manager.MemcpyHostToDeviceAsync(&manager.m_numFeaValue, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(uint), pStream);
		//copy feature value start position of each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 	 sizeof(uint), pStream);
		//copy each feature start position in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(uint) * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									    sizeof(int) * nNumofFeature, pStream);

		numofDenseValue = manager.m_numFeaValue;//for computing gain of each fvalue
		maxNumFeaValueOneNode = manager.m_numFeaValue;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
	}

//	cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = bagManager.m_numFea * numofSNode;
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled

	//construct keys for exclusive scan
	uint *pnKey_d;
	checkCudaErrors(cudaMalloc((void**)&pnKey_d, bagManager.m_numFeaValue * sizeof(uint)));
	uint *pTempEachFeaStartEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;

	//set keys by GPU
	int maxSegLen = 0;
	int *pTempEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	int *pMaxLen = thrust::max_element(thrust::device, pTempEachFeaLenEachNode, pTempEachFeaLenEachNode + totalNumArray);
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(int), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = totalNumArray;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	SetKey<<<totalNumArray, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, pnKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute prefix sum for gd and hess (more than one arrays)
	double *pTempGDSum = bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	real *pTempHessSum = bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempGDSum, pTempGDSum);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempHessSum, pTempHessSum);


	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//default to left or right
	bool *pDefault2Right;
	checkCudaErrors(cudaMalloc((void**)&pDefault2Right, sizeof(bool) * bagManager.m_numFeaValue));
	checkCudaErrors(cudaMemset(pDefault2Right, 0, sizeof(bool) * bagManager.m_numFeaValue));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
											numofDenseValue, pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, pnKey_d, bagManager.m_numFea,
											bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
											pDefault2Right);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");
	
	//change the gain of the first feature value to 0
	int numFeaStartPos = bagManager.m_numFea * numofSNode;
//	printf("num fea start pos=%d (%d * %d)\n", numFeaStartPos, bagManager.m_numFea, numofSNode);
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numFeaStartPos, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
																bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
																numFeaStartPos, bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
																bagManager.m_numFeaValue);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after FirstFeaGain");

	clock_t end_comp_gain = clock();
	total_com_gain_t += (end_comp_gain - start_comp_gain);

//	cout << "searching" << endl;
	clock_t start_search = clock();
	real *pfLocalBestGain_d, *pfGlobalBestGain_d;
	int *pnLocalBestGainKey_d, *pnGlobalBestGainKey_d;
	//compute # of blocks for each node
	PROCESS_ERROR(maxNumFeaValueOneNode > 0);
	int blockSizeLocalBestGain;
	dim3 dimNumofBlockLocalBestGain;
	conf.ConfKernel(maxNumFeaValueOneNode, blockSizeLocalBestGain, dimNumofBlockLocalBestGain);
	PROCESS_ERROR(dimNumofBlockLocalBestGain.z == 1);
	dimNumofBlockLocalBestGain.z = numofSNode;//each node per super block
	int numBlockPerNode = dimNumofBlockLocalBestGain.x * dimNumofBlockLocalBestGain.y;

	checkCudaErrors(cudaMalloc((void**)&pfLocalBestGain_d, sizeof(real) * numBlockPerNode * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnLocalBestGainKey_d, sizeof(int) * numBlockPerNode * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pfGlobalBestGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnGlobalBestGainKey_d, sizeof(int) * numofSNode));
	//find the block level best gain for each node
	PickLocalBestSplitEachNode<<<dimNumofBlockLocalBestGain, blockSizeLocalBestGain, 0, (*(cudaStream_t*)pStream)>>>(
								bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
								pfLocalBestGain_d,
								pnLocalBestGainKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after PickLocalBestSplitEachNode");

	//find the global best gain for each node
	if(numBlockPerNode > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerNode, blockSizeBestGain, dimNumofBlockDummy);
		PickGlobalBestSplitEachNode<<<numofSNode, blockSizeBestGain, 0, (*(cudaStream_t*)pStream)>>>(
									pfLocalBestGain_d,
									pnLocalBestGainKey_d,
									pfGlobalBestGain_d,
									pnGlobalBestGainKey_d,
								    numBlockPerNode, numofSNode);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		GETERROR("after PickGlobalBestSplitEachNode");
	}
	else{//local best fea is the global best fea
		manager.MemcpyDeviceToDeviceAsync(pfLocalBestGain_d, pfGlobalBestGain_d,
										sizeof(real) * numofSNode, pStream);
		manager.MemcpyDeviceToDeviceAsync(pnLocalBestGainKey_d, pnGlobalBestGainKey_d,
											sizeof(int) * numofSNode, pStream);
	}

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t end_search = clock();
	total_search_t += end_search - start_search;

	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
									 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
									 pfGlobalBestGain_d, pnGlobalBestGainKey_d,
				  	  	  	  	  	 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
				  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 pDefault2Right, pnKey_d,
				  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pnKey_d));
	checkCudaErrors(cudaFree(pDefault2Right));
	checkCudaErrors(cudaFree(pfLocalBestGain_d));
	checkCudaErrors(cudaFree(pfGlobalBestGain_d));
	checkCudaErrors(cudaFree(pnLocalBestGainKey_d));
	checkCudaErrors(cudaFree(pnGlobalBestGainKey_d));
}

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode2(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//reset memory for this bag
	{
		manager.MemsetAsync(bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);

		manager.MemsetAsync(bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(double) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute index for each feature value
	KernelConf conf;
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(bagManager.m_numFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	//# of feature values that need to compute gains; the code below cannot be replaced by indexComp.m_totalNumFeaValue, due to some nodes becoming leaves.
	int numofDenseValue = -1, maxNumFeaValueOneNode = -1;
	if(numofSNode > 1)
	{
		IndexComputer indexComp;
		indexComp.AllocMem(bagManager.m_numFea, numofSNode);
		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t comIdx_start = clock();
		//compute gather index via GPUs
		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, bagId);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		//copy # of feature values of each node
		uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;

		clock_t start_gd = clock();
		//scatter operation
		//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
		numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns,
															   bagManager.m_numIns, manager.m_pDInsId, manager.m_pdDFeaValue,
															   bagManager.m_pIndicesEachBag_d, numofDenseValue,
															   bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   	   	manager.m_pDInsId, manager.m_pdDFeaValue, bagManager.m_numFeaValue,
															   		bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//copy # of feature values of a node
		manager.MemcpyHostToDeviceAsync(&manager.m_numFeaValue, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(uint), pStream);
		//copy feature value start position of each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 	 sizeof(uint), pStream);
		//copy each feature start position in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(uint) * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									    sizeof(int) * nNumofFeature, pStream);

		numofDenseValue = manager.m_numFeaValue;//for computing gain of each fvalue
		maxNumFeaValueOneNode = manager.m_numFeaValue;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
	}

	//compress fvalues ###########
	real *fvalue_h = new real[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMemcpy(fvalue_h, bagManager.m_pDenseFValueEachBag, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	real *csrFvalue = new real[bagManager.m_numFeaValue];
	uint *csrOrgFvalueStartPos = new uint[bagManager.m_numFeaValue];
	memset(csrOrgFvalueStartPos, -1, sizeof(uint) * bagManager.m_numFeaValue);
	uint *eachCsrLen = new uint[bagManager.m_numFeaValue];
	memset(eachCsrLen, -1, sizeof(uint) * bagManager.m_numFeaValue);
	uint *eachFeaLenEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachFeaStartPosEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	checkCudaErrors(cudaMemcpy(eachFeaLenEachNode_h, bagManager.m_pEachFeaLenEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaStartPosEachNode_h, bagManager.m_pEachFeaStartPosEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));
	uint *eachCompressedFeaLen = new uint[bagManager.m_numFea * numofSNode];
	memset(eachCompressedFeaLen, -1, sizeof(uint) * bagManager.m_numFea * numofSNode);
	uint *eachCompressedFeaStartPos = new uint[bagManager.m_numFea * numofSNode];
	memset(eachCompressedFeaStartPos, -1, sizeof(uint) * bagManager.m_numFea * numofSNode);
	uint csrId = 0, curFvalueToCompress = 0;
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++){
		eachCompressedFeaLen[i] = 0;
		uint feaStart = eachFeaStartPosEachNode_h[i];
		uint feaLen = eachFeaLenEachNode_h[i];
		if(feaLen == 0)continue;
		csrFvalue[csrId] = fvalue_h[feaStart];
		csrOrgFvalueStartPos[csrId] = curFvalueToCompress;
		eachCsrLen[csrId] = 1;
		eachCompressedFeaLen[i] = 1;
		for(int l = 1; l < feaLen; l++){
			curFvalueToCompress++;
			if(fabs(fvalue_h[feaStart + l] - csrFvalue[csrId]) > DeviceSplitter::rt_eps){
				eachCompressedFeaLen[i]++;
				csrId++;
				csrFvalue[csrId] = fvalue_h[feaStart + l];
				csrOrgFvalueStartPos[csrId] = curFvalueToCompress;
				eachCsrLen[csrId] = 1;
			}
			else
				eachCsrLen[csrId]++;
		}
		csrId++;
		curFvalueToCompress++;
	}
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++){
		uint prefix = 0;
		for(int l = 0; l < i; l++)
			prefix += eachCompressedFeaLen[l];
		eachCompressedFeaStartPos[i] = prefix;
	}
	uint *eachNodeSizeInCsr = new uint[numofSNode];
	uint *eachCsrNodeStartPos = new uint[numofSNode];
	for(int i = 0; i < numofSNode; i++){
		int posOfLastFeaThisNode = (i + 1) * bagManager.m_numFea - 1;
		int posOfFirstFeaThisNode = i * bagManager.m_numFea;
		eachNodeSizeInCsr[i] = eachCompressedFeaStartPos[posOfLastFeaThisNode] - eachCompressedFeaStartPos[posOfFirstFeaThisNode];
		eachNodeSizeInCsr[i] += eachCompressedFeaLen[posOfLastFeaThisNode];
		eachCsrNodeStartPos[i] = eachCompressedFeaStartPos[posOfFirstFeaThisNode];
		printf("node %d starts %u, len=%u\n", i, eachCsrNodeStartPos[i], eachNodeSizeInCsr[i]);
	}

	uint totalLen = 0;
	for(int i = 0; i < csrId; i++){
		totalLen += eachCsrLen[i];
	}
	uint totalNumCsrFvalue = 0;
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++)
		totalNumCsrFvalue += eachCompressedFeaLen[i];
	printf("csrLen=%u, totalLen=%u, totalLen2=%u; numofFeaValue=%u\n", csrId, totalLen, totalNumCsrFvalue, bagManager.m_numFeaValue);
	PROCESS_ERROR(csrId == totalNumCsrFvalue);
	PROCESS_ERROR(totalNumCsrFvalue < bagManager.m_numFeaValue);
	//PROCESS_ERROR(totalLen == bagManager.m_numFeaValue);
	//update gd and hess
	double *gd_h = new double[bagManager.m_numFeaValue];
	real *hess_h = new real[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMemcpy(gd_h, bagManager.m_pdGDPrefixSumEachBag, sizeof(double) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hess_h, bagManager.m_pHessPrefixSumEachBag, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	double *csrGD_h = new double[bagManager.m_numFeaValue];
	real *csrHess_h = new real[bagManager.m_numFeaValue];
	uint globalPos = 0;
	for(int i = 0; i < csrId; i++){
		csrGD_h[i] = 0;
		csrHess_h[i] = 0;
		uint len = eachCsrLen[i];
		for(int v = 0; v < len; v++){
			csrGD_h[i] += gd_h[globalPos];
			csrHess_h[i] += hess_h[globalPos];
			globalPos++;
		}
	}
	double totalGD = 0, totalHess = 0;
	for(int i = 0; i < csrId; i++){
		totalGD += csrGD_h[i];
		totalHess += csrHess_h[i];
	}
	double totalOrgGD = 0;
	for(int i = 0; i < bagManager.m_numFeaValue; i++){
		totalOrgGD += gd_h[i];
	}
	//printf("total gd=%f, total hess=%f, orgGD=%f\n", totalGD, totalHess, totalOrgGD);
	PROCESS_ERROR(fabs(totalGD - totalOrgGD) < 0.001);
	uint *pnKey_h = new uint[totalNumCsrFvalue];
	uint segStart = 0;
	for(int segId = 0; segId < bagManager.m_numFea * numofSNode; segId++){
		uint segLen = eachCompressedFeaLen[segId];
		for(int i = 0; i < segLen; i++){
			pnKey_h[i + segStart] = segId;
		}
		segStart += segLen;
	}
	thrust::inclusive_scan_by_key(thrust::host, pnKey_h, pnKey_h + totalNumCsrFvalue, csrGD_h, csrGD_h);
	thrust::inclusive_scan_by_key(thrust::host, pnKey_h, pnKey_h + totalNumCsrFvalue, csrHess_h, csrHess_h);
	//compute gain
	nodeStat *snNode_h = new nodeStat[bagManager.m_maxNumSplittable];
	checkCudaErrors(cudaMemcpy(snNode_h, bagManager.m_pSNodeStatEachBag, sizeof(nodeStat) * bagManager.m_maxNumSplittable, cudaMemcpyDeviceToHost));
	int *pid2snPos = new int[bagManager.m_maxNumSplittable];
	checkCudaErrors(cudaMemcpy(pid2snPos, bagManager.m_pPartitionId2SNPosEachBag, sizeof(int) * bagManager.m_maxNumSplittable, cudaMemcpyDeviceToHost));

	real *pGainOnEachFvalue_h = new real[totalNumCsrFvalue];
	bool *pDefault2Right_h = new bool[totalNumCsrFvalue];
	pGainOnEachFvalue_h[0] = 0;
	for(int i = 1; i < totalNumCsrFvalue; i++){
		//forward consideration (fvalues are sorted descendingly)
		double rChildGD = csrGD_h[i - 1];
		double rChildHess = csrHess_h[i - 1];
		uint segId = pnKey_h[i];
		uint pid = segId / bagManager.m_numFea;
		int snPos = pid2snPos[pid];
		PROCESS_ERROR(snPos >= 0 || snPos < bagManager.m_maxNumSplittable);
		double parentGD = snNode_h[snPos].sum_gd;
		double parentHess = snNode_h[snPos].sum_hess;
		PROCESS_ERROR(parentHess > 0);
		double tempGD = parentGD - rChildGD;
		double tempHess = parentHess - rChildHess;
		if(rChildHess >= 1 && tempHess >= 1)//need to compute the gain
		{
			double tempGain = (tempGD * tempGD)/(tempHess + DeviceSplitter::m_lambda) +
								   (rChildGD * rChildGD)/(rChildHess + DeviceSplitter::m_lambda) -
								   (parentGD * parentGD)/(parentHess + DeviceSplitter::m_lambda);
			pGainOnEachFvalue_h[i] = tempGain;
		}
		else{
			//assign gain to 0
			pGainOnEachFvalue_h[i] = 0;
		}

	    //backward consideration
	    int segLen = eachCompressedFeaLen[segId];
	    uint segStartPos = eachCompressedFeaStartPos[segId];
	    PROCESS_ERROR(segLen >= 0);
	    uint lastFvaluePos = segStartPos + segLen - 1;
	    PROCESS_ERROR(lastFvaluePos < totalNumCsrFvalue);
	    double totalMissingGD = parentGD - csrGD_h[lastFvaluePos];
	    double totalMissingHess = parentHess - csrHess_h[lastFvaluePos];
	    if(totalMissingHess < 1)//there is no instance with missing values
	    	continue;
	    //missing values to the right child
	    rChildGD += totalMissingGD;
	    rChildHess += totalMissingHess;
	    tempGD = parentGD - rChildGD;
	    tempHess = parentHess - rChildHess;
	    if(rChildHess >= 1 && tempHess >= 1){
	    	double tempGain = (tempGD * tempGD)/(tempHess + DeviceSplitter::m_lambda) +
				  	   	    (rChildGD * rChildGD)/(rChildHess + DeviceSplitter::m_lambda) -
				  	   	    (parentGD * parentGD)/(parentHess + DeviceSplitter::m_lambda);

	    	if(tempGain > 0 && tempGain - pGainOnEachFvalue_h[i] > 0.1){
	    		pGainOnEachFvalue_h[i] = tempGain;
	    		pDefault2Right_h[i] = true;
	    	}
	    }
	}

	//find best gain for each node
	int *pBestFeaId = new int[numofSNode];
	for(int i = 0; i < numofSNode; i++){
		for(int f = 0; f < bagManager.m_numFea; f++){
			uint segLen = eachCompressedFeaLen[i * bagManager.m_numFea + f];
			if(segLen == 0)continue;
			uint segStartPos = eachCompressedFeaStartPos[i * bagManager.m_numFea + f];
			pGainOnEachFvalue_h[segStartPos] = 0;//fix first gain
		}
	}

	//	cout << "searching" << endl;
	clock_t start_search = clock();
	uint *pEachCsrNodeSize_d;
	uint *pEachCsrNodeStart_d;
	real *pGainEachCsrFvalue_d;
	real *pMaxGain_d;
	uint *pMaxGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeSize_d, sizeof(uint) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeStart_d, sizeof(uint) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pGainEachCsrFvalue_d, sizeof(real) * totalNumCsrFvalue));
	checkCudaErrors(cudaMalloc((void**)&pMaxGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pMaxGainKey_d, sizeof(uint) * numofSNode));
	checkCudaErrors(cudaMemcpy(pEachCsrNodeSize_d, eachNodeSizeInCsr, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pEachCsrNodeStart_d, eachCsrNodeStartPos, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pGainEachCsrFvalue_d, pGainOnEachFvalue_h, sizeof(real) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	real *pfLocalBestGain_d;
	uint *pnLocalBestGainKey_d;
	//compute # of blocks for each node
	uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pEachCsrNodeSize_d, pEachCsrNodeSize_d + numofSNode);
	checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
	PROCESS_ERROR(maxNumFeaValueOneNode > 0);
	int blockSizeLocalBestGain;
	dim3 dimNumofBlockLocalBestGain;
	conf.ConfKernel(maxNumFeaValueOneNode, blockSizeLocalBestGain, dimNumofBlockLocalBestGain);
	PROCESS_ERROR(dimNumofBlockLocalBestGain.z == 1);
	dimNumofBlockLocalBestGain.z = numofSNode;	//each node per super block
	int numBlockPerNode = dimNumofBlockLocalBestGain.x * dimNumofBlockLocalBestGain.y;

	checkCudaErrors(cudaMalloc((void**)&pfLocalBestGain_d, sizeof(real) * numBlockPerNode * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnLocalBestGainKey_d, sizeof(uint) * numBlockPerNode * numofSNode));
	//find the block level best gain for each node
	PickLocalBestSplitEachNode<<<dimNumofBlockLocalBestGain, blockSizeLocalBestGain, 0, (*(cudaStream_t*)pStream)>>>(
									pEachCsrNodeSize_d,
									pEachCsrNodeStart_d,
									pGainEachCsrFvalue_d,
									pfLocalBestGain_d,
									pnLocalBestGainKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after PickLocalBestSplitEachNode");

	//find the global best gain for each node
	if(numBlockPerNode > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerNode, blockSizeBestGain, dimNumofBlockDummy);
		if(blockSizeBestGain < 64)//make sure the reduction is power of two
			blockSizeBestGain = 64;
		PickGlobalBestSplitEachNode<<<numofSNode, blockSizeBestGain, 0, (*(cudaStream_t*)pStream)>>>(
										pfLocalBestGain_d,
										pnLocalBestGainKey_d,
										pMaxGain_d,
										pMaxGainKey_d,
									    numBlockPerNode, numofSNode);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		GETERROR("after PickGlobalBestSplitEachNode");
	}
	else{//local best fea is the global best fea
		manager.MemcpyDeviceToDeviceAsync(pfLocalBestGain_d, pMaxGain_d, sizeof(real) * numofSNode, pStream);
		manager.MemcpyDeviceToDeviceAsync(pnLocalBestGainKey_d, pMaxGainKey_d, sizeof(uint) * numofSNode, pStream);
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pEachCsrNodeSize_d));
	checkCudaErrors(cudaFree(pEachCsrNodeStart_d));
	checkCudaErrors(cudaFree(pGainEachCsrFvalue_d));

	//find the split value and feature
	uint numofSeg = numofSNode * bagManager.m_numFea;
	uint *pEachCompressedFeaStartPos_d;
	uint *pEachCompressedFeaLen_d;
	real *pCsrFvalue_d;
	int *pPartId2SNPos_d;
	double *pCsrGD_d;
	real *pCsrHess_d;
	bool *pCsrDefault2Right_d;
	uint *pnCsrKey_d;

	checkCudaErrors(cudaMalloc((void**)&pEachCompressedFeaStartPos_d, sizeof(uint) * numofSeg));
	checkCudaErrors(cudaMalloc((void**)&pEachCompressedFeaLen_d, sizeof(uint) * numofSeg));
	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue_d, sizeof(real) * totalNumCsrFvalue));
	checkCudaErrors(cudaMalloc((void**)&pPartId2SNPos_d, sizeof(int) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD_d, sizeof(double) * totalNumCsrFvalue));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess_d, sizeof(real) * totalNumCsrFvalue));
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right_d, sizeof(bool) * totalNumCsrFvalue));
	checkCudaErrors(cudaMalloc((void**)&pnCsrKey_d, sizeof(uint) * totalNumCsrFvalue));

	checkCudaErrors(cudaMemcpy(pEachCompressedFeaStartPos_d, eachCompressedFeaStartPos, sizeof(uint) * numofSeg, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pEachCompressedFeaLen_d, eachCompressedFeaLen, sizeof(uint) * numofSeg, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFvalue_d, csrFvalue, sizeof(real) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pPartId2SNPos_d, pid2snPos, sizeof(int) * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrGD_d, csrGD_h, sizeof(double) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrHess_d, csrHess_h, sizeof(real) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrDefault2Right_d, pDefault2Right_h, sizeof(bool) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pnCsrKey_d, pnKey_h, sizeof(uint) * totalNumCsrFvalue, cudaMemcpyHostToDevice));
	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
										 pEachCompressedFeaStartPos_d,
										 pEachCompressedFeaLen_d,
										 pCsrFvalue_d,
										 pMaxGain_d, pMaxGainKey_d,
										 pPartId2SNPos_d, nNumofFeature,
					  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 pCsrGD_d,
					  	  	  	  	  	 pCsrHess_d,
					  	  	  	  	  	 pCsrDefault2Right_d, pnCsrKey_d,
					  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pMaxGain_d));
	checkCudaErrors(cudaFree(pMaxGainKey_d));
	checkCudaErrors(cudaFree(pEachCompressedFeaStartPos_d));
	checkCudaErrors(cudaFree(pEachCompressedFeaLen_d));
	checkCudaErrors(cudaFree(pCsrFvalue_d));
	checkCudaErrors(cudaFree(pPartId2SNPos_d));
	checkCudaErrors(cudaFree(pCsrGD_d));
	checkCudaErrors(cudaFree(pCsrHess_d));
	checkCudaErrors(cudaFree(pCsrDefault2Right_d));
	checkCudaErrors(cudaFree(pnCsrKey_d));
/*
//	cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = bagManager.m_numFea * numofSNode;
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled

	//construct keys for exclusive scan
	uint *pnKey_d;
	checkCudaErrors(cudaMalloc((void**)&pnKey_d, bagManager.m_numFeaValue * sizeof(uint)));
	uint *pTempEachFeaStartEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;

	//set keys by GPU
	int maxSegLen = 0;
	int *pTempEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	int *pMaxLen = thrust::max_element(thrust::device, pTempEachFeaLenEachNode, pTempEachFeaLenEachNode + totalNumArray);
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(int), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = totalNumArray;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	SetKey<<<totalNumArray, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, pnKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute prefix sum for gd and hess (more than one arrays)
	double *pTempGDSum = bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	real *pTempHessSum = bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempGDSum, pTempGDSum);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey_d, pnKey_d + bagManager.m_numFeaValue, pTempHessSum, pTempHessSum);


	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//default to left or right
	bool *pDefault2Right;
	checkCudaErrors(cudaMalloc((void**)&pDefault2Right, sizeof(bool) * bagManager.m_numFeaValue));
	checkCudaErrors(cudaMemset(pDefault2Right, 0, sizeof(bool) * bagManager.m_numFeaValue));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
											numofDenseValue, pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, pnKey_d, bagManager.m_numFea,
											bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
											pDefault2Right);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");

	//change the gain of the first feature value to 0
	int numFeaStartPos = bagManager.m_numFea * numofSNode;
//	printf("num fea start pos=%d (%d * %d)\n", numFeaStartPos, bagManager.m_numFea, numofSNode);
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numFeaStartPos, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
																bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
																numFeaStartPos, bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
																bagManager.m_numFeaValue);
*/
}

