/*
 * FindFeaCsr.cu
 *
 *  Created on: Jul 28, 2017
 *      Author: zeyi
 */

#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <unistd.h>
#include <set>
#include <fstream>

#include "IndexComputer.h"
#include "FindFeaKernel.h"
#include "../Bagging/BagManager.h"
#include "../CSR/BagCsrManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/segmentedMax.h"
#include "../../SharedUtility/segmentedSum.h"
#include "../../SharedUtility/setSegmentKey.h"

#include "../CSR/CsrSplit.h"
#include "../CSR/CsrCompressor.h"
#include "../../syncarray.h"
using std::set;


void CsrCompression(int numofSNode, uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos, uint *eachCompressedFeaLen,
		uint *eachNodeSizeInCsr, uint *eachCsrNodeStartPos, real *csrFvalue, double *csrGD_h, real *csrHess_h, uint *eachCsrLen){
	BagManager bagManager;
	real *fvalue_h = new real[bagManager.m_numFeaValue];
	uint *eachFeaLenEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachFeaStartPosEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	checkCudaErrors(cudaMemcpy(fvalue_h, fvalue_d, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
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
	checkCudaErrors(cudaMemcpy(gd_h, fgd_d, sizeof(double) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hess_h, fhess_d, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));

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


uint numofDenseValue_previous;
bool firstTime = true;

void AfterCompression(GBDTGPUMemManager &manager, BagCsrManager &csrManager, BagManager &bagManager, IndexComputer &indexComp, KernelConf &conf, void *pStream, double &total_scan_t,
					  int &maxNumFeaValueOneNode){
	//cout << "prefix sum" << endl;
	printf("prefix sum\n");
	clock_t start_scan = clock();
	int nNumofFeature = manager.m_numofFea;
	double *pGD_d = (double*)indexComp.histogram_d.addr;//reuse memory; must be here, as curNumCsr may change in different level.
	real *pHess_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 2);//reuse memory
	real *pGain_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * (sizeof(real)/sizeof(uint) + 2));

	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[0];
	int numSeg = bagManager.m_numFea * numofSNode;
	//construct keys for exclusive scan
	checkCudaErrors(cudaMemset(csrManager.getMutableCsrKey(), -1, sizeof(uint) * csrManager.curNumCsr));
//	checkCudaErrors(cudaMemset(pCSRMultableKey, -1, sizeof(uint) * csrManager.curNumCsr));
	printf("done constructing key... number of segments is %d\n", numSeg);

	//set keys by GPU
	uint maxSegLen = 0;
	uint *pMaxLen = thrust::max_element(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numSeg);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(&maxSegLen, pMaxLen, sizeof(uint), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = numSeg;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
	if(optimiseSetKey == false)
		SetKey<<<numSeg, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
			(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, csrManager.getMutableCsrKey());
//		(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pCSRMultableKey);
	else{
		if(numSeg < 1000000)
			SetKey<<<numSeg, blockSize, sizeof(uint) * 2, (*(cudaStream_t*)pStream)>>>
				(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, csrManager.getMutableCsrKey());
//			(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pCSRMultableKey);
		else{
			int numSegEachBlk = numSeg/10000;
			int numofBlkSetKey = (numSeg + numSegEachBlk - 1) / numSegEachBlk;
			SetKey<<<numofBlkSetKey, blockSize, 0, (*(cudaStream_t*)pStream)>>>(csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen,
					numSegEachBlk, numSeg, csrManager.getMutableCsrKey());
//					numSegEachBlk, numSeg, pCSRMultableKey);
		}
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	cudaDeviceSynchronize();


	//compute prefix sum for gd and hess (more than one arrays)
	thrust::inclusive_scan_by_key(thrust::device, csrManager.getCsrKey(), csrManager.getCsrKey() + csrManager.curNumCsr, pGD_d, pGD_d);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::device, csrManager.getCsrKey(), csrManager.getCsrKey() + csrManager.curNumCsr, pHess_d, pHess_d);

	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//compute gain; default to left or right
	bool *default2Right = (bool*)indexComp.partitionMarker.addr;
	checkCudaErrors(cudaMemset(default2Right, 0, sizeof(bool) * csrManager.curNumCsr));//this is important (i.e. initialisation)
	checkCudaErrors(cudaMemset(pGain_d, 0, sizeof(real) * csrManager.curNumCsr));

//	cout << "compute gain" << endl;
	printf("compute gain\n");
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(csrManager.curNumCsr, blockSizeComGain, dimNumofBlockToComGain);
	cudaDeviceSynchronize();
	GETERROR("before ComputeGainDense");

	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag,
											bagManager.m_pPartitionId2SNPosEachBag,
											DeviceSplitter::m_lambda, pGD_d, pHess_d, csrManager.getCsrFvalue(),
											csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, csrManager.getCsrKey(), bagManager.m_numFea,
//											csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pCSRKey, bagManager.m_numFea,
											pGain_d, default2Right);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");

	//change the gain of the first feature value to 0
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numSeg, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
											csrManager.pEachCsrFeaStartPos, numSeg, pGain_d, csrManager.curNumCsr);

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
	printf("max fvalue one node=%d\n", maxNumFeaValueOneNode);
	SegmentedMax(maxNumFeaValueOneNode, numofSNode, csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos,
				 pGain_d, pStream, pMaxGain_d, pMaxGainKey_d);

	printf("finding split info\n");
	//find the split value and feature
	FindSplitInfo<<<1, numofSNode, 0>>>(
										 csrManager.pEachCsrFeaStartPos,
										 csrManager.pEachCsrFeaLen,
										 csrManager.getCsrFvalue(),
										 pMaxGain_d, pMaxGainKey_d,
										 bagManager.m_pPartitionId2SNPosEachBag, nNumofFeature,
					  	  	  	  	  	 bagManager.m_pSNodeStatEachBag,
					  	  	  	  	  	 pGD_d, pHess_d,
					  	  	  	  	  	 default2Right, csrManager.getCsrKey(),
//					  	  	  	  	  	 default2Right, pCSRKey,
					  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag,
					  	  	  	  	  	 bagManager.m_pRChildStatEachBag,
					  	  	  	  	  	 bagManager.m_pLChildStatEachBag);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(pMaxGain_d));
	checkCudaErrors(cudaFree(pMaxGainKey_d));
}

void AllNode2CompGD(GBDTGPUMemManager &manager, BagCsrManager &csrManager, BagManager &bagManager, IndexComputer &indexComp, KernelConf &conf, void *pStream,
		int &curNumofNode, double &total_com_idx_t, double &total_fill_gd_t, int &maxNumFeaValueOneNode, double &total_csr_len_t){
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
	int nNumofFeature = manager.m_numofFea;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[0];
	PROCESS_ERROR(nNumofFeature > 0);
	if(curNumofNode == 1){
		checkCudaErrors(cudaMemcpy(csrManager.preFvalueInsId, manager.m_pDInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDeviceToDevice));
		numofDenseValue_previous = bagManager.m_numFeaValue;//initialise dense value length
	}

	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//compute index for each feature value
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(bagManager.m_numFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	clock_t csr_len_t = clock();
	if(numofSNode > 1)
	{
		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t comIdx_start = clock();
		//compute gather index via GPUs
		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, 0);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		//copy # of feature values of each node
		uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d;
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		PROCESS_ERROR(bagManager.m_numFeaValue >= csrManager.curNumCsr);
		//split nodes
		csr_len_t = clock();

		if(indexComp.partitionMarker.reservedSize < csrManager.curNumCsr * 8){//make sure enough memory for reuse
			printf("reallocate memory for marker (sn=%d): %u v.s. %u.......\n", numofSNode, indexComp.partitionMarker.reservedSize/8, csrManager.curNumCsr);
			indexComp.partitionMarker.reserveSpace(csrManager.curNumCsr * 8, sizeof(unsigned char));
		}
		uint *pOldCsrLen_d = (uint*)indexComp.partitionMarker.addr;
		unsigned char *pCsrId2Pid = (unsigned char*)(((uint*)indexComp.partitionMarker.addr) + csrManager.curNumCsr);
		checkCudaErrors(cudaMemcpy(pOldCsrLen_d, csrManager.getCsrLen(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemset(pCsrId2Pid, (int)-1, sizeof(char) * csrManager.curNumCsr));

		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, csrManager.getMutableCsrStart());

uint *pCsrNewLen_d;// = (uint*)(indexComp.histogram_d.addr);
//uint *pCsrNewLen_d = (uint*)(indexComp.histogram_d.addr);
checkCudaErrors(cudaMallocHost((void**)&pCsrNewLen_d, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(pCsrNewLen_d, 0, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(csrManager.pEachCsrFeaLen, 0, sizeof(uint) * bagManager.m_numFea * numofSNode));
		dim3 dimNumofBlockToCsrLen;
		uint blockSizeCsrLen = 128;

cudaDeviceSynchronize();
		dimNumofBlockToCsrLen.x = (numofDenseValue_previous + blockSizeCsrLen - 1) / blockSizeCsrLen;
		newCsrLenFvalue<<<dimNumofBlockToCsrLen, blockSizeCsrLen, blockSizeCsrLen * sizeof(uint)>>>(
				csrManager.preFvalueInsId, numofDenseValue_previous,
				bagManager.m_pInsIdToNodeIdEachBag,
				bagManager.m_pPreMaxNid_h[0], csrManager.getCsrStart(),
				csrManager.getCsrFvalue(), csrManager.curNumCsr,
				csrManager.pEachCsrFeaStartPos, bagManager.m_pPreNumSN_h[0],
				bagManager.m_numFea, csrManager.getCsrKey(), pCsrNewLen_d, pCsrId2Pid);
//				bagManager.m_numFea, pCSRKey, pCsrNewLen_d, pCsrId2Pid);

		GETERROR("after newCsrLenFvalue");
		LoadFvalueInsId<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(
						bagManager.m_numIns, manager.m_pDInsId, csrManager.preFvalueInsId, bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue);
		GETERROR("after LoadFvalueInsId");

		printf("filling fvalue\n");
		cudaDeviceSynchronize();

		real *pCsrFvalueSpare = (real*)(((int*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 2);//reuse memory

		int blockSizeFillFvalue;
		dim3 dimNumBlockToFillFvalue;
		conf.ConfKernel(csrManager.curNumCsr, blockSizeFillFvalue, dimNumBlockToFillFvalue);
//fid hess sum
uint *hess_cnt_d;
checkCudaErrors(cudaMalloc((void**)&hess_cnt_d, sizeof(uint) * bagManager.m_numFea));
checkCudaErrors(cudaMemset(hess_cnt_d, 0, sizeof(uint) * bagManager.m_numFea));
		fillFvalue<<<dimNumBlockToFillFvalue, blockSizeFillFvalue>>>(csrManager.getCsrFvalue(), csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos,
				   bagManager.m_pPreNumSN_h[0], bagManager.m_numFea, csrManager.getCsrKey(), pOldCsrLen_d, pCsrId2Pid,
//				   bagManager.m_pPreNumSN_h[bagId], bagManager.m_numFea, pCSRKey, pOldCsrLen_d, pCsrId2Pid,
				   pCsrFvalueSpare, pCsrNewLen_d, csrManager.pEachCsrFeaLen);
		GETERROR("after fillFvalue");
		cudaDeviceSynchronize();

		//compute number of CSR in each node
		checkCudaErrors(cudaMemset(csrManager.pEachNodeSizeInCsr, 0, sizeof(uint) * bagManager.m_maxNumSplittable));
		printf("done filling\n");
		dim3 dimNumSeg;
		dimNumSeg.x = numofSNode;
		uint blockSize = 128;
		segmentedSum<<<dimNumSeg, blockSize, blockSize * sizeof(uint)>>>(csrManager.pEachCsrFeaLen, bagManager.m_numFea, csrManager.pEachNodeSizeInCsr);
		GETERROR("after segmentedSum");

		int blockSizeLoadCsrLen;
		dim3 dimNumofBlockToLoadCsrLen;
		conf.ConfKernel(csrManager.curNumCsr * 2, blockSizeLoadCsrLen, dimNumofBlockToLoadCsrLen);
		//uint *pCsrMarker = (uint*)indexComp.partitionMarker.addr;
uint *pCsrMarker;
checkCudaErrors(cudaMalloc((void**)&pCsrMarker, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(pCsrMarker, 0, sizeof(uint) * csrManager.curNumCsr * 2));
		map2One<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(pCsrNewLen_d, csrManager.curNumCsr * 2, pCsrMarker);
		GETERROR("after map2One");
		cudaDeviceSynchronize();
		thrust::inclusive_scan(thrust::device, pCsrMarker, pCsrMarker + csrManager.curNumCsr * 2, pCsrMarker);
		cudaDeviceSynchronize();
		uint previousNumCsr = csrManager.curNumCsr;
		checkCudaErrors(cudaMemcpy(&csrManager.curNumCsr, pCsrMarker + csrManager.curNumCsr * 2 - 1, sizeof(uint), cudaMemcpyDefault));

		checkCudaErrors(cudaMemset(csrManager.getMutableCsrLen(), 0, sizeof(uint) * csrManager.curNumCsr));
cudaDeviceSynchronize();
		loadDenseCsr<<<dimNumofBlockToLoadCsrLen, blockSizeLoadCsrLen>>>(pCsrFvalueSpare, pCsrNewLen_d,
				previousNumCsr * 2, csrManager.curNumCsr, pCsrMarker,
				csrManager.getMutableCsrFvalue(), csrManager.getMutableCsrLen());
		GETERROR("after loadDenseCsr");
		printf("done load dense csr: number of csr is %d\n", csrManager.curNumCsr);
		thrust::exclusive_scan(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numofSNode * bagManager.m_numFea, csrManager.pEachCsrFeaStartPos);
cudaDeviceSynchronize();


		thrust::exclusive_scan(thrust::device, csrManager.pEachNodeSizeInCsr, csrManager.pEachNodeSizeInCsr + numofSNode, csrManager.pEachCsrNodeStartPos);
		numofDenseValue_previous = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);//number of dense fvalues.
		uint *pCsrStartCurRound = (uint*)indexComp.partitionMarker.addr;
		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, pCsrStartCurRound);
		PROCESS_ERROR(csrManager.curNumCsr <= bagManager.m_numFeaValue);
		cudaDeviceSynchronize();
checkCudaErrors(cudaFree(pCsrMarker));
		printf("exit if\n");
	}
	else
	{
		clock_t start_gd = clock();
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//copy # of feature values of a node
		manager.MemcpyHostToDeviceAsync(&manager.m_numFeaValue, bagManager.m_pNumFvalueEachNodeEachBag_d,
										sizeof(uint), pStream);
		//copy feature value start position of each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d,
									 	 sizeof(uint), pStream);
		//copy each feature start position in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d,
										sizeof(uint) * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d,
									    sizeof(int) * nNumofFeature, pStream);

		maxNumFeaValueOneNode = manager.m_numFeaValue;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
		//###### compress
		cudaDeviceSynchronize();
		CsrCompressor compressor;
		csrManager.curNumCsr = compressor.totalOrgNumCsr;
		compressor.CsrCompression(csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen,
								  csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos, csrManager.getMutableCsrFvalue(), csrManager.getMutableCsrLen());
	}
	//need to compute for every new tree
	printf("reserve memory\n");
	if(indexComp.histogram_d.reservedSize < csrManager.curNumCsr * (2 + 2 * sizeof(real)/sizeof(uint))){//make sure enough memory for reuse
		printf("reallocate memory for histogram (sn=%u): %u v.s. %u.......\n", numofSNode, indexComp.histogram_d.reservedSize,
				csrManager.curNumCsr * (2 + 2 * sizeof(real)/sizeof(uint)));
		indexComp.histogram_d.reserveSpace(csrManager.curNumCsr * (2 + 2 * sizeof(real)/sizeof(uint)), sizeof(uint));
	}
	cudaDeviceSynchronize();
	double *pGD_d = (double*)indexComp.histogram_d.addr;//reuse memory; must be here, as curNumCsr may change in different level.
	real *pHess_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 2);//reuse memory
	checkCudaErrors(cudaMemset(pGD_d, 0, sizeof(double) * csrManager.curNumCsr));
	checkCudaErrors(cudaMemset(pHess_d, 0, sizeof(real) * csrManager.curNumCsr));
	dim3 dimNumofBlockForGD;
	dimNumofBlockForGD.x = csrManager.curNumCsr;
	uint blockSizeForGD = 64;
	uint sharedMemSizeForGD = blockSizeForGD * (sizeof(double) + sizeof(double));
	const uint *pCsrStartPos_d;
	if(numofSNode == 1)
		pCsrStartPos_d = CsrCompressor::pCsrStart_d;
	else
		pCsrStartPos_d = (uint*)indexComp.partitionMarker.addr;
	printf("comp gd and hess\n");


	ComputeGDHess<<<dimNumofBlockForGD, blockSizeForGD, sharedMemSizeForGD>>>(csrManager.getCsrLen(), pCsrStartPos_d,
			bagManager.m_pInsGradEachBag,
			bagManager.m_pInsHessEachBag,
			csrManager.preFvalueInsId, pGD_d, pHess_d);
	cudaDeviceSynchronize();
	GETERROR("after ComputeGD");
	clock_t csr_len_end = clock();
	total_csr_len_t += (csr_len_end - csr_len_t);
}

void DeviceSplitter::FeaFinderAllNode2(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);

	IndexComputer indexComp;
	indexComp.AllocMem(bagManager.m_numFea, numofSNode, bagManager.m_maxNumSplittable);
	KernelConf conf;
	cudaDeviceSynchronize();

	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
	int maxNumFeaValueOneNode = -1;
	AllNode2CompGD(manager, csrManager, bagManager, indexComp, conf, pStream, curNumofNode, total_com_idx_t, total_fill_gd_t, maxNumFeaValueOneNode, total_csr_len_t);

	AfterCompression(manager, csrManager, bagManager, indexComp, conf, pStream, total_scan_t, maxNumFeaValueOneNode);
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
uint *eachNewCompressedFeaLen_merge;
uint *eachNewCompressedFeaStart_merge;

void AllNode3CompGD(GBDTGPUMemManager &manager, BagCsrManager &csrManager, BagManager &bagManager, IndexComputer &indexComp, KernelConf &conf, void *pStream,
					int &curNumofNode, double &total_com_idx_t, double &total_fill_gd_t, int &maxNumFeaValueOneNode){
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[0];
	int nNumofFeature = manager.m_numofFea;
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
	vector<vector<real> > newCsrFvalue(numofSNode * bagManager.m_numFea, vector<real>());
		vector<vector<uint> > eachNewCsrLen(numofSNode * bagManager.m_numFea, vector<uint>());

		if(preFvalueInsId == NULL || curNumofNode == 1){
			eachNewCompressedFeaLen_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
			eachNewCompressedFeaStart_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
			eachCompressedFeaStartPos_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
			eachCompressedFeaLen_merge = new uint[bagManager.m_numFea * bagManager.m_maxNumSplittable];
			csrGD_h_merge = new double[bagManager.m_numFeaValue];
			csrHess_h_merge = new real[bagManager.m_numFeaValue];
			eachNodeSizeInCsr_merge = new uint[bagManager.m_maxNumSplittable];
			eachCsrNodeStartPos_merge = new uint[bagManager.m_maxNumSplittable];
			csrFvalue_merge = new real[bagManager.m_numFeaValue];
			eachCsrLen_merge = new uint[bagManager.m_numFeaValue];
			checkCudaErrors(cudaMallocHost((void**)&preFvalueInsId, sizeof(int) * bagManager.m_numFeaValue));
			checkCudaErrors(cudaMemcpy(preFvalueInsId, manager.m_pDInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
		}
		//split nodes
		int *pInsId2Nid = new int[bagManager.m_numIns];//ins id to node id
		checkCudaErrors(cudaMemcpy(pInsId2Nid, bagManager.m_pInsIdToNodeIdEachBag, sizeof(int) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
		//################3

		//reset memory for this bag
		{
			manager.MemsetAsync(fgd_d, 0, sizeof(double) * bagManager.m_numFeaValue, pStream);
			manager.MemsetAsync(fhess_d, 0, sizeof(real) * bagManager.m_numFeaValue, pStream);
			manager.MemsetAsync(fgain_d, 0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		}
		cudaStreamSynchronize((*(cudaStream_t*)pStream));

		//compute index for each feature value
		int blockSizeLoadGD;
		dim3 dimNumofBlockToLoadGD;
		conf.ConfKernel(bagManager.m_numFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
		//# of feature values that need to compute gains; the code below cannot be replaced by indexComp.m_totalNumFeaValue, due to some nodes becoming leaves.
		int numofDenseValue = -1;
		if(numofSNode > 1)
		{
			//####################
			printf("total csr fvalue=%u\n", totalNumCsrFvalue_merge);/**/
			//split nodes
			PROCESS_ERROR(bagManager.m_numFeaValue >= totalNumCsrFvalue_merge);
			memset(eachNewCompressedFeaLen_merge, 0, sizeof(uint) * bagManager.m_numFea * numofSNode);
			uint globalFvalueId = 0;
			clock_t extra_start = clock();
			for(int csrId = 0; csrId < totalNumCsrFvalue_merge; csrId++){
				uint csrLen = eachCsrLen_merge[csrId];
				//fid of this csr
				int fid = -1;
				for(int segId = 0; segId < bagManager.m_numFea * numofSNode; segId++){
					uint segStart = eachCompressedFeaStartPos_merge[segId];
					uint feaLen = eachCompressedFeaLen_merge[segId];
					if(csrId >= segStart && csrId < segStart + feaLen){
						fid = segId % bagManager.m_numFea;//may have problem here??
						break;
					}
				}
				PROCESS_ERROR(fid != -1 && fid < bagManager.m_numFea);

				//decompressed
				for(int i = 0; i < csrLen; i++){
					int insId = preFvalueInsId[globalFvalueId];
					globalFvalueId++;
					PROCESS_ERROR(insId >= 0);
					int pid = pInsId2Nid[insId] - bagManager.m_pPreMaxNid_h[0] - 1;//mapping to new node
					if(csrFvalue_merge[csrId] == 0.369250 && i == 0){
						printf("pid=%d, insId=%d, old csrLen=%d\n", pid, insId, csrLen);
					}
					if(pid < 0)
						continue;//############## this way okay?
					PROCESS_ERROR(pid >= 0 && pid < numofSNode);
					if(i == 0 || newCsrFvalue[pid * bagManager.m_numFea + fid].empty() ||
					   fabs(newCsrFvalue[pid * bagManager.m_numFea + fid].back() - csrFvalue_merge[csrId]) > DeviceSplitter::rt_eps){
						newCsrFvalue[pid * bagManager.m_numFea + fid].push_back(csrFvalue_merge[csrId]);
						eachNewCsrLen[pid * bagManager.m_numFea + fid].push_back(1);
						eachNewCompressedFeaLen_merge[pid * bagManager.m_numFea + fid]++;
					}
					else
						eachNewCsrLen[pid * bagManager.m_numFea + fid].back()++;
				}
			}
			clock_t extra_end = clock();
			total_extra_time += (double(extra_end - extra_start)/CLOCKS_PER_SEC);

			uint totalNewCsr = 0;
			for(int i = 0; i < numofSNode * bagManager.m_numFea; i++)
				totalNewCsr += eachNewCsrLen[i].size();
			printf("hello world org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNewCsr);
			thrust::exclusive_scan(thrust::host, eachNewCompressedFeaLen_merge, eachNewCompressedFeaLen_merge + numofSNode * bagManager.m_numFea, eachNewCompressedFeaStart_merge);
			delete[] pInsId2Nid;
			//###############################
			PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
			clock_t comIdx_start = clock();
			//compute gather index via GPUs
			indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, 0);
			clock_t comIdx_end = clock();
			total_com_idx_t += (comIdx_end - comIdx_start);

			//copy # of feature values of each node
			uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d;

			clock_t start_gd = clock();
			//scatter operation
			//total fvalue to load may be smaller than m_totalFeaValue, due to some nodes becoming leaves.
			numofDenseValue = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
			LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag,
																   bagManager.m_pInsHessEachBag,
																   bagManager.m_numIns, manager.m_pDInsId, fvalue_org_d,
																   bagManager.m_pIndicesEachBag_d, numofDenseValue,
																   fgd_d, fhess_d, fvalue_d);
			cudaStreamSynchronize((*(cudaStream_t*)pStream));
			clock_t end_gd = clock();
			total_fill_gd_t += (end_gd - start_gd);
			uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
			checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
			//###########
			LoadFvalueInsId<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(
							manager.m_pDInsId, preFvalueInsId, bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue);
			cudaStreamSynchronize((*(cudaStream_t*)pStream));
			//##############
		}
		else
		{
			clock_t start_gd = clock();
			LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag,
																   	   	bagManager.m_pInsHessEachBag, bagManager.m_numIns,
																   	   	manager.m_pDInsId, bagManager.m_numFeaValue,
																   		fgd_d, fhess_d);
			checkCudaErrors(cudaMemcpy(fvalue_d, fvalue_org_d, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDefault));
			cudaStreamSynchronize((*(cudaStream_t*)pStream));
			clock_t end_gd = clock();
			total_fill_gd_t += (end_gd - start_gd);

			clock_t comIdx_start = clock();
			//copy # of feature values of a node
			manager.MemcpyHostToDeviceAsync(&manager.m_numFeaValue, bagManager.m_pNumFvalueEachNodeEachBag_d, sizeof(uint), pStream);
			//copy feature value start position of each node
			manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d, sizeof(uint), pStream);
			//copy each feature start position in each node
			manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d,
											sizeof(uint) * nNumofFeature, pStream);
			//copy # of feature values of each feature in each node
			manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d,
										    sizeof(int) * nNumofFeature, pStream);

			numofDenseValue = manager.m_numFeaValue;//for computing gain of each fvalue
			maxNumFeaValueOneNode = manager.m_numFeaValue;
			clock_t comIdx_end = clock();
			total_com_idx_t += (comIdx_end - comIdx_start);
		}

		//###### compress
		CsrCompression(numofSNode, totalNumCsrFvalue_merge, eachCompressedFeaStartPos_merge, eachCompressedFeaLen_merge,
					   eachNodeSizeInCsr_merge, eachCsrNodeStartPos_merge, csrFvalue_merge, csrGD_h_merge, csrHess_h_merge, eachCsrLen_merge);
		printf("total csr fvalue=%u\n", totalNumCsrFvalue_merge);

//test two ways of computing csr
		int segIdCnt = 0, innerCsrIdCnt = 0;
		if(numofSNode > 1)
		for(int csrId = 0; csrId < totalNumCsrFvalue_merge; csrId++){
			if(eachCsrLen_merge[csrId] != eachNewCsrLen[segIdCnt][innerCsrIdCnt]){
				printf("csrId=%d, segIdCnt=%d, innerCsrIdCnt=%d, segSize=%d\n", csrId, segIdCnt, innerCsrIdCnt, eachNewCsrLen[segIdCnt].size());
				printf("right csr len=%d v.s. wrong len=%d\n", eachCsrLen_merge[csrId], eachNewCsrLen[segIdCnt][innerCsrIdCnt]);
				printf("right fvalue=%f v.s. wrong fvalue=%f\n", csrFvalue_merge[csrId], newCsrFvalue[segIdCnt][innerCsrIdCnt]);
				printf("previous right fvalue=%f v.s. wrong fvalue=%f; right csr len=%d v.s. wrong len=%d\n", csrFvalue_merge[csrId - 1],
						newCsrFvalue[segIdCnt][innerCsrIdCnt - 1], eachCsrLen_merge[csrId - 1], eachNewCsrLen[segIdCnt][innerCsrIdCnt - 1]);
				printf("next right fvalue=%f v.s. wrong fvalue=%f; right csr len=%d v.s. wrong len=%d\n", csrFvalue_merge[csrId + 1],
						newCsrFvalue[segIdCnt][innerCsrIdCnt + 1], eachCsrLen_merge[csrId + 1], eachNewCsrLen[segIdCnt][innerCsrIdCnt + 1]);
				//exit(0);
			}
			innerCsrIdCnt++;
			if(eachNewCsrLen[segIdCnt].size() == innerCsrIdCnt){
				segIdCnt++;
				innerCsrIdCnt = 0;
			}
		}
//end test

		//	cout << "prefix sum" << endl;
		int numSeg = bagManager.m_numFea * numofSNode;

		csrManager.curNumCsr = totalNumCsrFvalue_merge;
		checkCudaErrors(cudaMemcpy(csrManager.pEachCsrFeaStartPos, eachCompressedFeaStartPos_merge, sizeof(uint) * numSeg, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(csrManager.pEachCsrFeaLen, eachCompressedFeaLen_merge, sizeof(uint) * numSeg, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(csrManager.pCsrFvalue, csrFvalue_merge, sizeof(real) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrLen(), eachCsrLen_merge, sizeof(uint) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
		if(indexComp.partitionMarker.reservedSize < csrManager.curNumCsr){//make sure enough memory for reuse
			indexComp.partitionMarker.reserveSpace(csrManager.curNumCsr, sizeof(bool));
		}
		if(indexComp.histogram_d.reservedSize < csrManager.curNumCsr * 4){//make sure enough memory for reuse
			indexComp.histogram_d.reserveSpace(csrManager.curNumCsr * 4, sizeof(uint));
		}
		double *pGD_d = (double*)indexComp.histogram_d.addr;//reuse memory; must be here, as curNumCsr may change in different level.
		real *pHess_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 2);//reuse memory

		checkCudaErrors(cudaMemcpy(pHess_d, csrHess_h_merge, sizeof(real) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pGD_d, csrGD_h_merge, sizeof(double) * totalNumCsrFvalue_merge, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(csrManager.pEachNodeSizeInCsr, eachNodeSizeInCsr_merge, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(csrManager.pEachCsrNodeStartPos, eachCsrNodeStartPos_merge, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
		//compute the feature with the maximum number of values
		cudaDeviceSynchronize();//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled
}

void DeviceSplitter::FeaFinderAllNode3(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);

	IndexComputer indexComp;
	indexComp.AllocMem(bagManager.m_numFea, numofSNode, bagManager.m_maxNumSplittable);
	KernelConf conf;
	cudaDeviceSynchronize();

//	cout << bagManager.m_maxNumSplittable << endl;
	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
	int maxNumFeaValueOneNode = -1;

	AllNode2CompGD(manager, csrManager, bagManager, indexComp, conf, pStream, curNumofNode, total_com_idx_t, total_fill_gd_t, maxNumFeaValueOneNode, total_csr_len_t);
SyncArray<real> csrFvalue(csrManager.curNumCsr);
csrFvalue.set_device_data(csrManager.getMutableCsrFvalue());
real *csrFvalue_h = csrFvalue.host_data();
SyncArray<uint> csrLen(csrManager.curNumCsr);
csrLen.set_device_data(csrManager.getMutableCsrLen());
uint *csrLen_h = csrLen.host_data();

	printf("done with fast csr\n");
	AllNode3CompGD(manager, csrManager, bagManager, indexComp, conf, pStream, curNumofNode, total_com_idx_t, total_fill_gd_t, maxNumFeaValueOneNode);

SyncArray<real> csrFvalue3(csrManager.curNumCsr);
csrFvalue3.set_device_data(csrManager.getMutableCsrFvalue());
real *csrFvalue_h3 = csrFvalue3.host_data();
SyncArray<uint> csrLen3(csrManager.curNumCsr);
csrLen3.set_device_data(csrManager.getMutableCsrLen());
uint *csrLen_h3 = csrLen3.host_data();
	printf("done with naive csr\n");

//compare two compression methods
	for(int i = 0; i < csrManager.curNumCsr; i++){
		if(csrFvalue_h[i] == 0.369250){
			printf("%f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i], csrFvalue_h3[i], i, csrLen_h[i], csrLen_h3[i]);
		}
	}
	for(int i = 0; i < csrManager.curNumCsr; i++){
		if(csrFvalue_h3[i] == 0.369250){
			printf("%f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i], csrFvalue_h3[i], i, csrLen_h[i], csrLen_h3[i]);
		}
	}

	int cnt = 0;
	for(int i = 0; i < csrManager.curNumCsr; i++){
		if(csrFvalue_h[i] != csrFvalue_h3[i]){
			printf("%f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i], csrFvalue_h3[i], i, csrLen_h[i], csrLen_h3[i]);
			if(cnt == 0)
				printf("previous %f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i-1], csrFvalue_h3[i-1], i-1, csrLen_h[i-1], csrLen_h3[i-1]);
			cnt++;
		}
		if(csrLen_h[i] != csrLen_h3[i]){
			printf("%f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i], csrFvalue_h3[i], i, csrLen_h[i], csrLen_h3[i]);
			if(cnt == 0)
				printf("previous %f v.s. %f, id=%d; len: %d v.s. %d\n", csrFvalue_h[i-1], csrFvalue_h3[i-1], i-1, csrLen_h[i-1], csrLen_h3[i-1]);
			cnt++;
		}

		if(cnt == 10)
			exit(0);
	}
//end comparison

	AfterCompression(manager, csrManager, bagManager, indexComp, conf, pStream, total_scan_t, maxNumFeaValueOneNode);
}


