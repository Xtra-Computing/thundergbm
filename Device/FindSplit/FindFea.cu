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

void CsrCompression(int numofSNode, uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos, uint *eachCompressedFeaLen,
		uint *eachNodeSizeInCsr, uint *eachCsrNodeStartPos, real *csrFvalue, double *csrGD_h, real *csrHess_h, uint *eachCsrLen){
	BagManager bagManager;
	real *fvalue_h = new real[bagManager.m_numFeaValue];
	uint *eachFeaLenEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachFeaStartPosEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	checkCudaErrors(cudaMemcpy(fvalue_h, bagManager.m_pDenseFValueEachBag, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaLenEachNode_h, bagManager.m_pEachFeaLenEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaStartPosEachNode_h, bagManager.m_pEachFeaStartPosEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));

	uint csrId = 0, curFvalueToCompress = 0;
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++){
		eachCompressedFeaLen[i] = 0;
		uint feaLen = eachFeaLenEachNode_h[i];
		uint feaStart = eachFeaStartPosEachNode_h[i];
		if(feaLen == 0)continue;
		csrFvalue[csrId] = fvalue_h[feaStart];
		eachCsrLen[csrId] = 1;
		eachCompressedFeaLen[i] = 1;
		for(int l = 1; l < feaLen; l++){
			curFvalueToCompress++;
			if(fabs(fvalue_h[feaStart + l] - csrFvalue[csrId]) > DeviceSplitter::rt_eps){
				eachCompressedFeaLen[i]++;
				csrId++;
				csrFvalue[csrId] = fvalue_h[feaStart + l];
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

	for(int i = 0; i < numofSNode; i++){
		int posOfLastFeaThisNode = (i + 1) * bagManager.m_numFea - 1;
		int posOfFirstFeaThisNode = i * bagManager.m_numFea;
		eachNodeSizeInCsr[i] = eachCompressedFeaStartPos[posOfLastFeaThisNode] - eachCompressedFeaStartPos[posOfFirstFeaThisNode];
		eachNodeSizeInCsr[i] += eachCompressedFeaLen[posOfLastFeaThisNode];
		eachCsrNodeStartPos[i] = eachCompressedFeaStartPos[posOfFirstFeaThisNode];
//		printf("node %d starts %u, len=%u\n", i, eachCsrNodeStartPos[i], eachNodeSizeInCsr[i]);
	}

	totalNumCsrFvalue = csrId;
//	printf("csrLen=%u, totalLen=%u, numofFeaValue=%u\n", csrId, totalLen, bagManager.m_numFeaValue);
	PROCESS_ERROR(totalNumCsrFvalue < bagManager.m_numFeaValue);
	//compute csr gd and hess
	double *gd_h = new double[bagManager.m_numFeaValue];
	real *hess_h = new real[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMemcpy(gd_h, bagManager.m_pdGDPrefixSumEachBag, sizeof(double) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hess_h, bagManager.m_pHessPrefixSumEachBag, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));

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

	printf("org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNumCsrFvalue);

	delete[] fvalue_h;
	delete[] eachFeaLenEachNode_h;
	delete[] eachFeaStartPosEachNode_h;
	delete[] gd_h;
	delete[] hess_h;
}

/**
 * @brief: efficient best feature finder
 */
__global__ void LoadFvalueInsId(const int *pOrgFvalueInsId, int *pNewFvalueInsId, const unsigned int *pDstIndexEachFeaValue, int numFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	//index for scatter
	int idx = pDstIndexEachFeaValue[gTid];
	if(idx == -1)//instance is in a leaf node
		return;

	CONCHECKER(idx >= 0);
	CONCHECKER(idx < numFeaValue);

	//scatter: store GD, Hess and the feature value.
	pNewFvalueInsId[idx] = pOrgFvalueInsId[gTid];
}

__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, real *csrFvalue, uint numCsr, const uint *preRoundEachCsrFeaStartPos, const uint preRoundNumSN, int numFea,
						  real *eachCsrFvalueSparse, uint *csrNewLen, uint *eachCsrFeaLen, uint *eachNodeSizeInCsr){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	if(gTid >= numFeaValue)//thread has nothing to do
		return;

	int insId = preFvalueInsId[gTid];
	int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node
	uint csrId = numCsr;
	RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	uint segId = numFea * preRoundNumSN;
	RangeBinarySearch(csrId, preRoundEachCsrFeaStartPos, numFea * preRoundNumSN, segId);
	uint nodeId = segId / numFea;
	uint nodeStartPos = preRoundEachCsrFeaStartPos[nodeId * numFea];
	uint sizePreNodesAhead = nodeStartPos;
	uint curNodeSize;
	if(nodeId == preRoundNumSN - 1)
		curNodeSize = numCsr - nodeStartPos;
	else
		curNodeSize = preRoundEachCsrFeaStartPos[(nodeId + 1) * numFea] - nodeStartPos;
	uint localId = csrId - sizePreNodesAhead;
	uint orgValue;
	if(pid % 2 == 1){
		orgValue = atomicAdd(csrNewLen + sizePreNodesAhead * 2 + curNodeSize + localId, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[sizePreNodesAhead * 2 + curNodeSize + localId] = csrFvalue[csrId];
	}
	else{
		orgValue = atomicAdd(csrNewLen + sizePreNodesAhead * 2 + localId, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[sizePreNodesAhead * 2 + localId] = csrFvalue[csrId];
	}

	if(orgValue == 0){
		uint feaId = segId % numFea;
		CONCHECKER(feaId < numFea);
		atomicAdd(eachCsrFeaLen + pid * numFea + feaId, 1);
		atomicAdd(eachNodeSizeInCsr + pid, 1);
	}
}

__global__ void map2One(const uint *eachCsrFeaLen, uint numCsr, uint *csrMarker){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrFeaLen[gTid] != 0)
		csrMarker[gTid] = 1;
	else
		csrMarker[gTid] = 0;
}

__global__ void loadDenseCsr(const real *eachCsrFvalueSparse, const uint *eachCsrFeaLen, uint numCsr, const uint *csrIdx, real *eachCsrFvalueDense, uint *eachCsrFeaLenDense){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrFeaLen[gTid] != 0){
		uint idx = csrIdx[gTid] - 1;//inclusive scan is used to compute indices.
		eachCsrFeaLenDense[idx] = eachCsrFeaLen[gTid];
		eachCsrFvalueDense[idx] = eachCsrFvalueSparse[gTid];
	}
}

int *preFvalueInsId = NULL;
uint totalNumCsrFvalue_merge;
uint *eachCompressedFeaStartPos_merge;
uint *eachCompressedFeaLen_merge;
double *csrGD_h_merge;
real *csrHess_h_merge;
uint *eachNodeSizeInCsr_merge;
uint *eachCsrNodeStartPos_merge;
real *csrFvalue_merge;
uint *eachCsrLen_merge;
uint *eachNewCompressedFeaStart_merge;
void DeviceSplitter::FeaFinderAllNode2(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	printf("preSN=%u, curSN=%u\n", bagManager.m_pPreNumSN_h[bagId], numofSNode);
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);
	//################
	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
	vector<vector<real> > newCsrFvalue(numofSNode * bagManager.m_numFea, vector<real>());

	if(preFvalueInsId == NULL || curNumofNode == 1){
		eachNewCompressedFeaStart_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
		eachCompressedFeaStartPos_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
		eachCompressedFeaLen_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
		csrGD_h_merge = new double[bagManager.m_numFeaValue];
		csrHess_h_merge = new real[bagManager.m_numFeaValue];
		eachCsrNodeStartPos_merge = new uint[bagManager.m_maxNumSplittable];
		eachCsrLen_merge = new uint[bagManager.m_numFeaValue];
		checkCudaErrors(cudaMallocHost((void**)&eachNodeSizeInCsr_merge, sizeof(uint) * bagManager.m_maxNumSplittable));
		checkCudaErrors(cudaMallocHost((void**)&csrFvalue_merge, sizeof(real) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMallocHost((void**)&preFvalueInsId, sizeof(int) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMemcpy(preFvalueInsId, manager.m_pDInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	}
	//split nodes
	int *pInsId2Nid = new int[bagManager.m_numIns];//ins id to node id
	checkCudaErrors(cudaMemcpy(pInsId2Nid, bagManager.m_pInsIdToNodeIdEachBag, sizeof(int) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
	//################3

	//reset memory for this bag
	{
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
		//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
		numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
		//###########
		printf("total csr fvalue=%u\n", totalNumCsrFvalue_merge);/**/
		PROCESS_ERROR(bagManager.m_numFeaValue >= totalNumCsrFvalue_merge);
		//split nodes
		uint *eachCsrStart;
		checkCudaErrors(cudaMallocHost((void**)&eachCsrStart, sizeof(uint) * totalNumCsrFvalue_merge));
		thrust::exclusive_scan(thrust::host, eachCsrLen_merge, eachCsrLen_merge + totalNumCsrFvalue_merge, eachCsrStart);
		uint *firstCsrLen;
		real *eachCsrFvalueSparse;
		uint *eachCsrFeaLen;
		uint *eachCsrFeaStartPos;
		checkCudaErrors(cudaMallocHost((void**)&firstCsrLen, sizeof(uint) * totalNumCsrFvalue_merge * 2));
		checkCudaErrors(cudaMallocHost((void**)&eachCsrFvalueSparse, sizeof(real) * totalNumCsrFvalue_merge * 2));
		checkCudaErrors(cudaMallocHost((void**)&eachCsrFeaLen, sizeof(uint) * bagManager.m_numFea * numofSNode));
		checkCudaErrors(cudaMallocHost((void**)&eachCsrFeaStartPos, sizeof(uint) * bagManager.m_numFea * bagManager.m_pPreNumSN_h[bagId]));
		checkCudaErrors(cudaMemset(firstCsrLen, 0, sizeof(uint) * totalNumCsrFvalue_merge * 2));
		checkCudaErrors(cudaMemset(eachCsrFeaLen, 0, sizeof(uint) * bagManager.m_numFea * numofSNode));
		checkCudaErrors(cudaMemcpy(eachCsrFeaStartPos, eachCompressedFeaStartPos_merge, sizeof(uint) * bagManager.m_numFea * bagManager.m_pPreNumSN_h[bagId],
						cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(eachNodeSizeInCsr_merge, 0, sizeof(uint) * bagManager.m_maxNumSplittable));

		newCsrLenFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(preFvalueInsId, bagManager.m_numFeaValue,
											bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns,
											bagManager.m_pPreMaxNid_h[bagId], eachCsrStart,
											csrFvalue_merge, totalNumCsrFvalue_merge,
											eachCsrFeaStartPos, bagManager.m_pPreNumSN_h[bagId],
											bagManager.m_numFea, eachCsrFvalueSparse, firstCsrLen, eachCsrFeaLen,
											eachNodeSizeInCsr_merge);

		int blockSizeLoadCsrLen;
		dim3 dimNumofBlockToLoadCsrLen;
		conf.ConfKernel(totalNumCsrFvalue_merge * 2, blockSizeLoadCsrLen, dimNumofBlockToLoadCsrLen);
		uint *csrMarker;
		checkCudaErrors(cudaMallocHost((void**)&csrMarker, sizeof(uint) * totalNumCsrFvalue_merge * 2));
		checkCudaErrors(cudaMemset(csrMarker, 0, sizeof(uint) * totalNumCsrFvalue_merge * 2));
		map2One<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(firstCsrLen, totalNumCsrFvalue_merge * 2, csrMarker);
		thrust::inclusive_scan(thrust::device, csrMarker, csrMarker + totalNumCsrFvalue_merge * 2, csrMarker);
		cudaDeviceSynchronize();
		uint totalNumCsrBest = csrMarker[totalNumCsrFvalue_merge * 2 - 1];
		printf("num csr=%u, dense csr=%u\n", totalNumCsrFvalue_merge * 2, totalNumCsrBest);
		uint *eachCsrFeaLenDense;
		real *eachCsrFvalueDense;
		checkCudaErrors(cudaMallocHost((void**)&eachCsrFeaLenDense, sizeof(uint) * totalNumCsrBest));
		checkCudaErrors(cudaMallocHost((void**)&eachCsrFvalueDense, sizeof(real) * totalNumCsrBest));
		checkCudaErrors(cudaMemset(eachCsrFeaLenDense, -1, sizeof(uint) * totalNumCsrBest));
		loadDenseCsr<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(eachCsrFvalueSparse, firstCsrLen, totalNumCsrFvalue_merge * 2, csrMarker, eachCsrFvalueDense, eachCsrFeaLenDense);

		cudaDeviceSynchronize();

		printf("hello world org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNumCsrBest);
		thrust::exclusive_scan(thrust::host, eachCsrFeaLen, eachCsrFeaLen + numofSNode * bagManager.m_numFea, eachNewCompressedFeaStart_merge);
		delete[] pInsId2Nid;
		//###############################
		LoadFvalueInsId<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(
						manager.m_pDInsId, preFvalueInsId, bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		thrust::exclusive_scan(thrust::host, eachNodeSizeInCsr_merge, eachNodeSizeInCsr_merge + numofSNode, eachCsrNodeStartPos_merge);//newly added#########
		totalNumCsrFvalue_merge = totalNumCsrBest;
		memcpy(eachCompressedFeaStartPos_merge, eachNewCompressedFeaStart_merge, sizeof(uint) * bagManager.m_numFea * numofSNode);
		memcpy(eachCompressedFeaLen_merge, eachCsrFeaLen, sizeof(uint) * bagManager.m_numFea * numofSNode);

		checkCudaErrors(cudaMemcpy(csrFvalue_merge, eachCsrFvalueDense, sizeof(real) * totalNumCsrBest, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(eachCsrLen_merge, eachCsrFeaLenDense, sizeof(uint) * totalNumCsrBest, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		real *pInsGrad = new real[bagManager.m_numIns];
		real *pInsHess = new real[bagManager.m_numIns];
		checkCudaErrors(cudaMemcpy(pInsGrad, bagManager.m_pInsGradEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(pInsHess, bagManager.m_pInsHessEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost));

		uint globalPos = 0;
		for(int i = 0; i < totalNumCsrFvalue_merge; i++){
			csrGD_h_merge[i] = 0;
			csrHess_h_merge[i] = 0;
			uint len = eachCsrLen_merge[i];
			for(int v = 0; v < len; v++){
				int insId = preFvalueInsId[globalPos];
				csrGD_h_merge[i] += pInsGrad[insId];
				csrHess_h_merge[i] += pInsHess[insId];
				globalPos++;
			}
		}
		delete[] pInsGrad;
		//##############
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
		//###### compress
		CsrCompression(numofSNode, totalNumCsrFvalue_merge, eachCompressedFeaStartPos_merge, eachCompressedFeaLen_merge,
				   eachNodeSizeInCsr_merge, eachCsrNodeStartPos_merge, csrFvalue_merge, csrGD_h_merge, csrHess_h_merge, eachCsrLen_merge);
		printf("total csr fvalue=%u\n", totalNumCsrFvalue_merge);
	}

	//	cout << "prefix sum" << endl;
	int numSeg = bagManager.m_numFea * numofSNode;
	real *pCsrFvalue_d;
	uint *pEachCompressedFeaStartPos_d;
	uint *pEachCompressedFeaLen_d;
	double *pCsrGD_d;
	real *pCsrHess_d;
	uint *pEachCsrNodeSize_d;
	uint *pEachCsrNodeStart_d;
	checkCudaErrors(cudaMalloc((void**)&pEachCompressedFeaStartPos_d, sizeof(uint) * numSeg));
	checkCudaErrors(cudaMalloc((void**)&pEachCompressedFeaLen_d, sizeof(uint) * numSeg));
	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue_d, sizeof(real) * totalNumCsrFvalue_merge));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD_d, sizeof(double) * totalNumCsrFvalue_merge));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess_d, sizeof(real) * totalNumCsrFvalue_merge));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeSize_d, sizeof(uint) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeStart_d, sizeof(uint) * numofSNode));

	checkCudaErrors(cudaMemcpy(pEachCompressedFeaStartPos_d, eachCompressedFeaStartPos_merge, sizeof(uint) * numSeg, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pEachCompressedFeaLen_d, eachCompressedFeaLen_merge, sizeof(uint) * numSeg, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFvalue_d, csrFvalue_merge, sizeof(real) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrHess_d, csrHess_h_merge, sizeof(real) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrGD_d, csrGD_h_merge, sizeof(double) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pEachCsrNodeSize_d, eachNodeSizeInCsr_merge, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pEachCsrNodeStart_d, eachCsrNodeStartPos_merge, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled

	//construct keys for exclusive scan
	uint *pnCsrKey_d;
	checkCudaErrors(cudaMalloc((void**)&pnCsrKey_d, sizeof(uint) * totalNumCsrFvalue_merge));

	//set keys by GPU
	uint maxSegLen = 0;
	uint *pMaxLen = thrust::max_element(thrust::device, pEachCompressedFeaLen_d, pEachCompressedFeaLen_d + numSeg);
	checkCudaErrors(cudaMemcpyAsync(&maxSegLen, pMaxLen, sizeof(uint), cudaMemcpyDeviceToHost, (*(cudaStream_t*)pStream)));

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = numSeg;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	SetKey<<<numSeg, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(pEachCompressedFeaStartPos_d, pEachCompressedFeaLen_d, pnCsrKey_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute prefix sum for gd and hess (more than one arrays)
	thrust::inclusive_scan_by_key(thrust::device, pnCsrKey_d, pnCsrKey_d + totalNumCsrFvalue_merge, pCsrGD_d, pCsrGD_d);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::device, pnCsrKey_d, pnCsrKey_d + totalNumCsrFvalue_merge, pCsrHess_d, pCsrHess_d);

	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//compute gain
	//default to left or right
	bool *pCsrDefault2Right_d;
	real *pGainEachCsrFvalue_d;
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right_d, sizeof(bool) * totalNumCsrFvalue_merge));
	checkCudaErrors(cudaMalloc((void**)&pGainEachCsrFvalue_d, sizeof(real) * totalNumCsrFvalue_merge));

	//cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(totalNumCsrFvalue_merge, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, pCsrGD_d, pCsrHess_d, pCsrFvalue_d,
											totalNumCsrFvalue_merge, pEachCompressedFeaStartPos_d, pEachCompressedFeaLen_d, pnCsrKey_d, bagManager.m_numFea,
											pGainEachCsrFvalue_d, pCsrDefault2Right_d);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");

	//change the gain of the first feature value to 0
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numSeg, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
										pEachCompressedFeaStartPos_d, numSeg, pGainEachCsrFvalue_d, totalNumCsrFvalue_merge);

	//	cout << "searching" << endl;
	clock_t start_search = clock();
	real *pMaxGain_d;
	uint *pMaxGainKey_d;
	checkCudaErrors(cudaMalloc((void**)&pMaxGain_d, sizeof(real) * numofSNode));
	checkCudaErrors(cudaMalloc((void**)&pMaxGainKey_d, sizeof(uint) * numofSNode));
	//compute # of blocks for each node
	uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pEachCsrNodeSize_d, pEachCsrNodeSize_d + numofSNode);
	checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));

	SegmentedMax(maxNumFeaValueOneNode, numofSNode, pEachCsrNodeSize_d, pEachCsrNodeStart_d,
					  pGainEachCsrFvalue_d, pStream, pMaxGain_d, pMaxGainKey_d);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//find the split value and feature
	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
										 pEachCompressedFeaStartPos_d,
										 pEachCompressedFeaLen_d,
										 pCsrFvalue_d,
										 pMaxGain_d, pMaxGainKey_d,
										 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
					  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 pCsrGD_d,
					  	  	  	  	  	 pCsrHess_d,
					  	  	  	  	  	 pCsrDefault2Right_d, pnCsrKey_d,
					  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	checkCudaErrors(cudaFree(pEachCsrNodeSize_d));
	checkCudaErrors(cudaFree(pEachCsrNodeStart_d));
	checkCudaErrors(cudaFree(pGainEachCsrFvalue_d));
	checkCudaErrors(cudaFree(pMaxGain_d));
	checkCudaErrors(cudaFree(pMaxGainKey_d));
	checkCudaErrors(cudaFree(pEachCompressedFeaStartPos_d));
	checkCudaErrors(cudaFree(pEachCompressedFeaLen_d));
	checkCudaErrors(cudaFree(pCsrFvalue_d));
	checkCudaErrors(cudaFree(pCsrGD_d));
	checkCudaErrors(cudaFree(pCsrHess_d));
	checkCudaErrors(cudaFree(pCsrDefault2Right_d));
	checkCudaErrors(cudaFree(pnCsrKey_d));
}

