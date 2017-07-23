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
#include "../Bagging/BagCsrManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/HostUtility.h"
#include "../../SharedUtility/powerOfTwo.h"
#include "../../SharedUtility/segmentedMax.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;

template<class T>
__global__ void SetKey(uint *pSegStart, T *pSegLen, uint *pnKey){
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
//		if(pos + segmentStartPos == 8756897){
//			print();
//		}
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
		printf("# of useful fvalue=%d\n", numofDenseValue);
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_numIns, manager.m_pDInsId, manager.m_pdDFeaValue,
															   bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue,
															   bagManager.m_pdGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
															   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		//testing
		uint *pTest_h = new uint[numofSNode];
		checkCudaErrors(cudaMemcpy(pTest_h, pTempNumFvalueEachNode, sizeof(uint) * numofSNode, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
//		for(int i = 0; i < numofSNode; i++)
//			printf("############## size of node %d is %u\n", i, pTest_h[i]);
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

	cout << "prefix sum" << endl;
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
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(int), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

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

	cout << "searching" << endl;
	clock_t start_search = clock();
	real *pfGlobalBestGain_d;
	int *pnGlobalBestGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pfGlobalBestGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pnGlobalBestGainKey_d, sizeof(int) * numofSNode));

	SegmentedMax(maxNumFeaValueOneNode, numofSNode, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
			bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
			bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue, pStream, pfGlobalBestGain_d, pnGlobalBestGainKey_d);

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
	checkCudaErrors(cudaFree(pfGlobalBestGain_d));
	checkCudaErrors(cudaFree(pnGlobalBestGainKey_d));
}


#include "CsrSplit.h"
uint numofDenseValue_previous;
void DeviceSplitter::FeaFinderAllNode2(void *pStream, int bagId)
{
	cudaDeviceSynchronize();
	GBDTGPUMemManager manager;
	BagManager bagManager;
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);
	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
	vector<vector<real> > newCsrFvalue(numofSNode * bagManager.m_numFea, vector<real>());

	if(curNumofNode == 1){
		checkCudaErrors(cudaMemcpy(csrManager.preFvalueInsId, manager.m_pDInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDeviceToDevice));
		numofDenseValue_previous = bagManager.m_numFeaValue;//initialise dense value length
	}

	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute index for each feature value
	KernelConf conf;
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(bagManager.m_numFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	//# of feature values that need to compute gains; the code below cannot be replaced by indexComp.m_totalNumFeaValue, due to some nodes becoming leaves.
	int maxNumFeaValueOneNode = -1;
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
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
		PROCESS_ERROR(bagManager.m_numFeaValue >= csrManager.curNumCsr);
		//split nodes
		uint *eachCsrStart;
		checkCudaErrors(cudaMalloc((void**)&eachCsrStart, sizeof(uint) * csrManager.curNumCsr));
		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, eachCsrStart);
		uint *eachNewCsrLen;
		real *eachCsrFvalueSparse;
		checkCudaErrors(cudaMalloc((void**)&eachNewCsrLen, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMalloc((void**)&eachCsrFvalueSparse, sizeof(real) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(eachNewCsrLen, 0, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(csrManager.pEachCsrFeaLen, 0, sizeof(uint) * bagManager.m_numFea * numofSNode));
		checkCudaErrors(cudaMemset(csrManager.pEachNodeSizeInCsr, 0, sizeof(uint) * bagManager.m_maxNumSplittable));
		newCsrLenFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(csrManager.preFvalueInsId, numofDenseValue_previous,
											bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns,
											bagManager.m_pPreMaxNid_h[bagId], eachCsrStart,
											csrManager.getCsrFvalue(), csrManager.curNumCsr,
											csrManager.pEachCsrFeaStartPos, bagManager.m_pPreNumSN_h[bagId],
											bagManager.m_numFea, eachCsrFvalueSparse, eachNewCsrLen, csrManager.pEachCsrFeaLen,
											csrManager.pEachNodeSizeInCsr, numofSNode);
		cudaDeviceSynchronize();
		GETERROR("after newCsrLenFvalue");

		int blockSizeLoadCsrLen;
		dim3 dimNumofBlockToLoadCsrLen;
		conf.ConfKernel(csrManager.curNumCsr * 2, blockSizeLoadCsrLen, dimNumofBlockToLoadCsrLen);
		uint *csrMarker;
		checkCudaErrors(cudaMalloc((void**)&csrMarker, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(csrMarker, 0, sizeof(uint) * csrManager.curNumCsr * 2));
		map2One<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(eachNewCsrLen, csrManager.curNumCsr * 2, csrMarker);
		GETERROR("after map2One");
		thrust::inclusive_scan(thrust::device, csrMarker, csrMarker + csrManager.curNumCsr * 2, csrMarker);
		cudaDeviceSynchronize();
		uint previousNumCsr = csrManager.curNumCsr;
		checkCudaErrors(cudaMemcpy(&csrManager.curNumCsr, csrMarker + csrManager.curNumCsr * 2 - 1, sizeof(uint), cudaMemcpyDefault));

		checkCudaErrors(cudaMemset(csrManager.getMutableCsrLen(), 0, sizeof(uint) * csrManager.curNumCsr));
		loadDenseCsr<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(eachCsrFvalueSparse, eachNewCsrLen, previousNumCsr * 2, csrManager.curNumCsr, csrMarker,
				csrManager.getMutableCsrFvalue(),
				csrManager.getMutableCsrLen());
		GETERROR("after loadDenseCsr");
		cudaDeviceSynchronize();

		thrust::exclusive_scan(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numofSNode * bagManager.m_numFea, csrManager.pEachCsrFeaStartPos);
		checkCudaErrors(cudaMemset(csrManager.preFvalueInsId, -1, sizeof(int) * bagManager.m_numFeaValue));//for testing
		LoadFvalueInsId<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(
						bagManager.m_numIns, manager.m_pDInsId, csrManager.preFvalueInsId, bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue);
		cudaDeviceSynchronize();


		GETERROR("after LoadFvalueInsId");
		thrust::exclusive_scan(thrust::device, csrManager.pEachNodeSizeInCsr, csrManager.pEachNodeSizeInCsr + numofSNode, csrManager.pEachCsrNodeStartPos);//newly added#########
		numofDenseValue_previous = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);//number of dense fvalues.

		PROCESS_ERROR(csrManager.curNumCsr <= bagManager.m_numFeaValue);
		checkCudaErrors(cudaMemset(csrManager.getMutableCsrGD(), 0, sizeof(double) * csrManager.curNumCsr));
		checkCudaErrors(cudaMemset(csrManager.getMutableCsrHess(), 0, sizeof(real) * csrManager.curNumCsr));
		GETERROR("before scan");
		uint *eachCsrStartCurRound;
		checkCudaErrors(cudaMalloc((void**)&eachCsrStartCurRound, sizeof(uint) * csrManager.curNumCsr));
		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, eachCsrStartCurRound);
		GETERROR("before compCsrGDHess");
		compCsrGDHess<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(csrManager.preFvalueInsId, numofDenseValue_previous,
													eachCsrStartCurRound, csrManager.curNumCsr,
													bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
													bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns,
													bagManager.m_numIns,
													csrManager.getMutableCsrGD(),
													csrManager.getMutableCsrHess());
		cudaDeviceSynchronize();
		GETERROR("after compCsrGDHess");

		checkCudaErrors(cudaFree(eachCsrStart));
		checkCudaErrors(cudaFree(eachNewCsrLen));
		checkCudaErrors(cudaFree(eachCsrFvalueSparse));
		checkCudaErrors(cudaFree(eachCsrStartCurRound));
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

		maxNumFeaValueOneNode = manager.m_numFeaValue;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
		//###### compress
		clock_t start_csr = clock();
		CsrCompression(numofSNode, csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen,
				csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos);
		cudaDeviceSynchronize();
		clock_t end_csr = clock();
		printf("compression time: %f\n", double(end_csr - start_csr) / CLOCKS_PER_SEC);
	}

	cudaDeviceSynchronize();
	//	cout << "prefix sum" << endl;
	int numSeg = bagManager.m_numFea * numofSNode;
	clock_t start_scan = clock();

	//construct keys for exclusive scan
	uint *pnCsrKey_d;
	checkCudaErrors(cudaMalloc((void**)&pnCsrKey_d, sizeof(uint) * csrManager.curNumCsr));
	checkCudaErrors(cudaMemset(pnCsrKey_d, -1, sizeof(uint) * csrManager.curNumCsr));

	//set keys by GPU
	uint maxSegLen = 0;
	uint *pMaxLen = thrust::max_element(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numSeg);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(uint), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = numSeg;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	SetKey<<<numSeg, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pnCsrKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute prefix sum for gd and hess (more than one arrays)
	thrust::inclusive_scan_by_key(thrust::device, pnCsrKey_d, pnCsrKey_d + csrManager.curNumCsr,
								  csrManager.getMutableCsrGD(), csrManager.getMutableCsrGD());//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::device, pnCsrKey_d, pnCsrKey_d + csrManager.curNumCsr,
								  csrManager.getMutableCsrHess(), csrManager.getMutableCsrHess());

	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//compute gain
	//default to left or right
	bool *pCsrDefault2Right_d;
	real *pGainEachCsrFvalue_d;
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right_d, sizeof(bool) * csrManager.curNumCsr));
	checkCudaErrors(cudaMemset(pCsrDefault2Right_d, 0, sizeof(bool) * csrManager.curNumCsr));//this is important (i.e. initialisation)
	checkCudaErrors(cudaMalloc((void**)&pGainEachCsrFvalue_d, sizeof(real) * csrManager.curNumCsr));
	checkCudaErrors(cudaMemset(pGainEachCsrFvalue_d, 0, sizeof(real) * csrManager.curNumCsr));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(csrManager.curNumCsr, blockSizeComGain, dimNumofBlockToComGain);
	cudaDeviceSynchronize();
	GETERROR("before ComputeGainDense");
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, csrManager.getCsrGD(), csrManager.getCsrHess(), csrManager.getCsrFvalue(),
											csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pnCsrKey_d, bagManager.m_numFea,
											pGainEachCsrFvalue_d, pCsrDefault2Right_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");

	//change the gain of the first feature value to 0
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numSeg, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
			csrManager.pEachCsrFeaStartPos, numSeg, pGainEachCsrFvalue_d, csrManager.curNumCsr);

	//	cout << "searching" << endl;
	cudaDeviceSynchronize();
	clock_t start_search = clock();
	real *pMaxGain_d;
	uint *pMaxGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pMaxGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pMaxGainKey_d, sizeof(uint) * numofSNode));
	checkCudaErrors(cudaMemset(pMaxGainKey_d, -1, sizeof(uint) * numofSNode));
	//compute # of blocks for each node
	uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, csrManager.pEachNodeSizeInCsr, csrManager.pEachNodeSizeInCsr + numofSNode);
	checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));

	SegmentedMax(maxNumFeaValueOneNode, numofSNode, csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos,
					  pGainEachCsrFvalue_d, pStream, pMaxGain_d, pMaxGainKey_d);

	//find the split value and feature
	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
										 csrManager.pEachCsrFeaStartPos,
										 csrManager.pEachCsrFeaLen,
										 csrManager.getCsrFvalue(),
										 pMaxGain_d, pMaxGainKey_d,
										 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
					  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 csrManager.getCsrGD(),
					  	  	  	  	  	 csrManager.getCsrHess(),
					  	  	  	  	  	 pCsrDefault2Right_d, pnCsrKey_d,
					  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	checkCudaErrors(cudaFree(pGainEachCsrFvalue_d));
	checkCudaErrors(cudaFree(pMaxGain_d));
	checkCudaErrors(cudaFree(pMaxGainKey_d));
	checkCudaErrors(cudaFree(pCsrDefault2Right_d));
	checkCudaErrors(cudaFree(pnCsrKey_d));
}
