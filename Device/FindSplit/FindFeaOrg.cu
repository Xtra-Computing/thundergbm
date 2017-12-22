/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief:
 */

#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

#include "IndexComputer.h"
#include "FindFeaKernel.h"
#include "../Bagging/BagManager.h"
#include "../Bagging/BagOrgManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/segmentedMax.h"
#include "../../SharedUtility/setSegmentKey.h"

/**
 * @brief: efficient best feature finder
 */
const int BLOCK_SIZE_ = 512;

const int NUM_BLOCKS = 32 * 56;

#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
__global__ void kernel_div(uint *keys, int n_keys, int n_f){
    KERNEL_LOOP(i, n_keys){
        keys[i] = keys[i] / n_f;
    }
}
__global__ void MarkPartition(int preMaxNid, int *pFvToInsId, int *pInsIdToNodeId,
                              int totalNumFv, uint *pParitionMarker, uint *tid2fid, int n_f) {
    int gTid = GLOBAL_TID();
    if (gTid >= totalNumFv)//thread has nothing to mark; note that "totalNumFv" will not decrease!
        return;

    uint insId = pFvToInsId[gTid];
    int nid = pInsIdToNodeId[insId];
    if (nid <= preMaxNid) {//instance in leaf node
//        pParitionMarker[gTid] = 255 * n_f + tid2fid[gTid];//can only support 8 level trees
        pParitionMarker[gTid] = INT_MAX;//can only support 8 level trees
        return;
    }
    int partitionId = nid - preMaxNid - 1;
    ECHECKER(partitionId);
    pParitionMarker[gTid] = partitionId * n_f + tid2fid[gTid];
}
void DeviceSplitter::FeaFinderAllNode(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	BagOrgManager orgManager(bagManager.m_numFeaValue, bagManager.m_numBag);
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//reset memory for this bag
	{
		manager.MemsetAsync(orgManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);

		manager.MemsetAsync(orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(double) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(orgManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
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
//		IndexComputer indexComp;
//		indexComp.AllocMem(bagManager.m_numFea, numofSNode, bagManager.m_maxNumSplittable);
//		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
//		clock_t comIdx_start = clock();
//		//compute gather index via GPUs
////		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, bagId);
//		clock_t comIdx_end = clock();
//		total_com_idx_t += (comIdx_end - comIdx_start);
//
//		//copy # of feature values of each node
//		uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
//
//		clock_t start_gd = clock();
//		//scatter operation
//		//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
//		numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
//		//printf("# of useful fvalue=%d\n", numofDenseValue);

        int *pTmpInsIdToNodeId = bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns;
        int blockSizeForFvalue;
        dim3 dimNumofBlockForFvalue;
        conf.ConfKernel(bagManager.m_numFeaValue, blockSizeForFvalue, dimNumofBlockForFvalue);
        MarkPartition << < dimNumofBlockForFvalue, blockSizeForFvalue >> >
                                                   (bagManager.m_pPreMaxNid_h[bagId], manager.m_pDInsId, pTmpInsIdToNodeId,
                                                           bagManager.m_numFeaValue, orgManager.m_pnKey_d, BagOrgManager::m_pnTid2Fid, bagManager.m_numFea);
        thrust::sequence(thrust::system::cuda::par, bagManager.m_pIndicesEachBag_d,
                         bagManager.m_pIndicesEachBag_d + bagManager.m_numFeaValue, 0);
        thrust::stable_sort_by_key(thrust::system::cuda::par, orgManager.m_pnKey_d, orgManager.m_pnKey_d + bagManager.m_numFeaValue, bagManager.m_pIndicesEachBag_d, thrust::less<uint>());
        thrust::counting_iterator<int> search_begin(0);
        thrust::upper_bound(thrust::cuda::par, orgManager.m_pnKey_d, orgManager.m_pnKey_d + bagManager.m_numFeaValue, search_begin, search_begin + bagManager.m_numFea * numofSNode, bagManager.m_pEachFeaLenEachNodeEachBag_d);
        thrust::adjacent_difference(thrust::cuda::par, bagManager.m_pEachFeaLenEachNodeEachBag_d, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagManager.m_numFea * numofSNode, bagManager.m_pEachFeaLenEachNodeEachBag_d);
        thrust::exclusive_scan(thrust::cuda::par, bagManager.m_pEachFeaLenEachNodeEachBag_d, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagManager.m_numFea * numofSNode, bagManager.m_pEachFeaStartPosEachNodeEachBag_d);

        kernel_div<<<NUM_BLOCKS,BLOCK_SIZE_>>>(orgManager.m_pnKey_d, bagManager.m_numFeaValue, bagManager.m_numFea);
        thrust::upper_bound(thrust::cuda::par, orgManager.m_pnKey_d, orgManager.m_pnKey_d + bagManager.m_numFeaValue, search_begin, search_begin + numofSNode, bagManager.m_pNumFvalueEachNodeEachBag_d);
        thrust::adjacent_difference(thrust::cuda::par, bagManager.m_pNumFvalueEachNodeEachBag_d, bagManager.m_pNumFvalueEachNodeEachBag_d + numofSNode, bagManager.m_pNumFvalueEachNodeEachBag_d);
        thrust::exclusive_scan(thrust::cuda::par, bagManager.m_pNumFvalueEachNodeEachBag_d, bagManager.m_pNumFvalueEachNodeEachBag_d + numofSNode, bagManager.m_pFvalueStartPosEachNodeEachBag_d);

        cudaStreamSynchronize((*(cudaStream_t*)pStream));
//		clock_t end_gd = clock();
//		total_fill_gd_t += (end_gd - start_gd);
        uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
        uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
        checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
        numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
//
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns,
															   bagManager.m_numIns, manager.m_pDInsId, orgManager.m_pdDFeaValue,
															   bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue,
															   orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   orgManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
                                                                orgManager.m_pnKey_d);
//		indexComp.FreeMem();
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   	   	manager.m_pDInsId, bagManager.m_numFeaValue,
															   		orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   	   	orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		checkCudaErrors(cudaMemcpy(orgManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue, orgManager.m_pdDFeaValue, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDefault));
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

	//cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = bagManager.m_numFea * numofSNode;
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled

	//construct keys for exclusive scan
	MEMSET(orgManager.m_pnKey_d, -1, sizeof(uint) * bagManager.m_numFeaValue);
	uint *pTempEachFeaStartEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;

	//set keys by GPU
	int maxSegLen = 0;
	int *pTempEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	int *pMaxLen = thrust::max_element(thrust::device, pTempEachFeaLenEachNode, pTempEachFeaLenEachNode + totalNumArray);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(int), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = totalNumArray;
	uint blockSize = 128;
//	printf("#of sn=%d, maxLen=%u\n", numofSNode, maxSegLen);
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	if(totalNumArray < 1000000)
		SetKey<<<totalNumArray, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, orgManager.m_pnKey_d);
	else{
		int numSegEachBlk = totalNumArray/10000;
		int numofBlkSetKey = (totalNumArray + numSegEachBlk - 1) / numSegEachBlk;
		SetKey<<<numofBlkSetKey, blockSize, 0, (*(cudaStream_t*)pStream)>>>(pTempEachFeaStartEachNode, pTempEachFeaLenEachNode,
				numSegEachBlk, totalNumArray, orgManager.m_pnKey_d);
	}
	if(orgManager.needCopy == true){
		checkCudaErrors(cudaMemcpy(orgManager.m_pnTid2Fid, orgManager.m_pnKey_d, sizeof(uint) * bagManager.m_numFeaValue, cudaMemcpyDeviceToDevice));
		orgManager.needCopy = false;
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute prefix sum for gd and hess (more than one arrays)
	double *pTempGDSum = orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	real *pTempHessSum = orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue;
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, orgManager.m_pnKey_d, orgManager.m_pnKey_d + bagManager.m_numFeaValue, pTempGDSum, pTempGDSum);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, orgManager.m_pnKey_d, orgManager.m_pnKey_d + bagManager.m_numFeaValue, pTempHessSum, pTempHessSum);

	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//default to left or right
	checkCudaErrors(cudaMemset(orgManager.m_pDefault2Right, 0, sizeof(bool) * bagManager.m_numFeaValue));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											orgManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
											numofDenseValue, pTempEachFeaStartEachNode, pTempEachFeaLenEachNode, orgManager.m_pnKey_d, bagManager.m_numFea,
											orgManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
											orgManager.m_pDefault2Right);
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
																numFeaStartPos, orgManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
																bagManager.m_numFeaValue);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after FirstFeaGain");

	clock_t end_comp_gain = clock();
	total_com_gain_t += (end_comp_gain - start_comp_gain);

	//cout << "searching" << endl;
	clock_t start_search = clock();
	real *pfGlobalBestGain_d;
	int *pnGlobalBestGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pfGlobalBestGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnGlobalBestGainKey_d, sizeof(int) * numofSNode));

	SegmentedMax(maxNumFeaValueOneNode, numofSNode, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
			bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
			orgManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue, pStream, pfGlobalBestGain_d, pnGlobalBestGainKey_d);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t end_search = clock();
	total_search_t += end_search - start_search;

	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
									 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 orgManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
									 pfGlobalBestGain_d, pnGlobalBestGainKey_d,
				  	  	  	  	  	 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
				  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 orgManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 orgManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	orgManager.m_pDefault2Right, orgManager.m_pnKey_d,
				  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pfGlobalBestGain_d));
	checkCudaErrors(cudaFree(pnGlobalBestGainKey_d));
}
