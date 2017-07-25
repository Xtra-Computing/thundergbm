/*
 * CsrCompressor.cu
 *
 *  Created on: Jul 25, 2017
 *      Author: zeyi
 */
#include <vector>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "CsrCompressor.h"
#include "BagCsrManager.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"

using std::vector;

real *CsrCompressor::fvalue_h = NULL;
uint *CsrCompressor::eachFeaLenEachNode_h = NULL;
uint *CsrCompressor::eachFeaStartPosEachNode_h = NULL;
uint *CsrCompressor::eachCsrFeaStartPos_h = NULL;
uint *CsrCompressor::eachCompressedFeaLen_h = NULL;
uint *CsrCompressor::eachCsrLen_h = NULL;
uint CsrCompressor::eachNodeSizeInCsr_h = 0;
real *CsrCompressor::csrFvalue_h = NULL;
int *CsrCompressor::insId_h = NULL;
uint CsrCompressor::totalNumCsr = 0;

uint *CsrCompressor::pCsrFeaStartPos_d = NULL;
uint *CsrCompressor::pCsrFeaLen_d = NULL;
uint *CsrCompressor::pCsrLen_d = NULL;
uint CsrCompressor::eachNodeSizeInCsr_d = 0;
real *CsrCompressor::pCsrFvalue_d;

CsrCompressor::CsrCompressor(){
	if(fvalue_h != NULL)
		return;
	BagManager bagManager;
	GBDTGPUMemManager manager;
	uint numFea = bagManager.m_numFea;
	uint numFeaValue = bagManager.m_numFeaValue;

	fvalue_h = new real[numFeaValue];
	eachFeaLenEachNode_h = new uint[numFea];
	eachFeaStartPosEachNode_h = new uint[numFea];
	eachCsrFeaStartPos_h = new uint[numFea];
	eachCompressedFeaLen_h = new uint[numFea];
	eachCsrLen_h = new uint[numFeaValue];
	eachNodeSizeInCsr_h = 0;
	csrFvalue_h = new real[numFeaValue];

	checkCudaErrors(cudaMemcpy(fvalue_h, manager.m_pdDFeaValue, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaLenEachNode_h, bagManager.m_pEachFeaLenEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaStartPosEachNode_h, bagManager.m_pEachFeaStartPosEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToHost));

	uint csrId = 0, curFvalueToCompress = 0;
	for(int i = 0; i < bagManager.m_numFea; i++){
		eachCompressedFeaLen_h[i] = 0;
		uint feaLen = eachFeaLenEachNode_h[i];
		uint feaStart = eachFeaStartPosEachNode_h[i];
		if(feaLen == 0)continue;
		csrFvalue_h[csrId] = fvalue_h[feaStart];
		eachCsrLen_h[csrId] = 1;
		eachCompressedFeaLen_h[i] = 1;
		for(int l = 1; l < feaLen; l++){
			curFvalueToCompress++;
			if(fabs(fvalue_h[feaStart + l] - csrFvalue_h[csrId]) > DeviceSplitter::rt_eps){
				eachCompressedFeaLen_h[i]++;
				csrId++;
				csrFvalue_h[csrId] = fvalue_h[feaStart + l];
				eachCsrLen_h[csrId] = 1;
			}
			else
				eachCsrLen_h[csrId]++;
		}
		csrId++;
		curFvalueToCompress++;
	}
	for(int i = 0; i < bagManager.m_numFea; i++){
		uint prefix = 0;
		for(int l = 0; l < i; l++)
			prefix += eachCompressedFeaLen_h[l];
		eachCsrFeaStartPos_h[i] = prefix;
	}

	totalNumCsr = csrId;
	eachNodeSizeInCsr_h = totalNumCsr;
	printf("org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNumCsr);
	PROCESS_ERROR(totalNumCsr < bagManager.m_numFeaValue);

	insId_h = new int[numFeaValue];
	checkCudaErrors(cudaMemcpy(insId_h, manager.m_pDInsId, sizeof(int) * numFeaValue, cudaMemcpyDefault));

	checkCudaErrors(cudaMalloc((void**)&pCsrFeaStartPos_d, sizeof(uint) * numFea));
	checkCudaErrors(cudaMalloc((void**)&pCsrFeaLen_d, sizeof(uint) * numFea));
	checkCudaErrors(cudaMalloc((void**)&pCsrLen_d, sizeof(uint) * totalNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue_d, sizeof(real) * totalNumCsr));
	checkCudaErrors(cudaMemcpy(pCsrFeaStartPos_d, eachCsrFeaStartPos_h, sizeof(uint) * bagManager.m_numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFeaLen_d, eachCompressedFeaLen_h, sizeof(uint) * bagManager.m_numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFvalue_d, csrFvalue_h, sizeof(real) * totalNumCsr, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(pCsrLen_d, eachCsrLen_h, sizeof(uint) * totalNumCsr, cudaMemcpyDefault));
}

void CsrCompressor::CsrCompression(uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos_d, uint *eachCompressedFeaLen_d,
								   uint *eachNodeSizeInCsr_d, uint *eachCsrNodeStartPos_d){
	BagManager bagManager;
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);
	totalNumCsrFvalue = totalNumCsr;
	//compute csr gd and hess
	checkCudaErrors(cudaMemcpy(eachCompressedFeaStartPos_d, pCsrFeaStartPos_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(eachCompressedFeaLen_d, pCsrFeaLen_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrFvalue(), pCsrFvalue_d, sizeof(real) * totalNumCsr, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrLen(), pCsrLen_d, sizeof(uint) * totalNumCsr, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemset(eachCsrNodeStartPos_d, 0, sizeof(uint)));
	checkCudaErrors(cudaMemcpy(eachNodeSizeInCsr_d, &eachNodeSizeInCsr_h, sizeof(uint), cudaMemcpyHostToDevice));

	//need to compute for every new tree
	real *insGD_h = new real[bagManager.m_numIns];
	real *insHess_h = new real[bagManager.m_numIns];
	checkCudaErrors(cudaMemcpy(insGD_h, bagManager.m_pInsGradEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(insHess_h, bagManager.m_pInsHessEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
	clock_t start = clock();
	vector<double> v_csrGD(totalNumCsr, 0);
	vector<real> v_csrHess(totalNumCsr, 0);
	//uint globalPos = 0;
	uint *eachCsrStartPos = new uint[totalNumCsr];
	uint *eachCsrStartCurRound;
	checkCudaErrors(cudaMalloc((void**)&eachCsrStartCurRound, sizeof(uint) * totalNumCsr));
	thrust::exclusive_scan(thrust::device, pCsrLen_d, pCsrLen_d + totalNumCsr, eachCsrStartCurRound);
	checkCudaErrors(cudaMemcpy(eachCsrStartPos, eachCsrStartCurRound, sizeof(uint) * totalNumCsr, cudaMemcpyDeviceToHost));
	int i, v;
	uint len, startPos;
//#pragma omp parallel for private(i, v, len, startPos) schedule(dynamic) nowait
	for(i = 0; i < totalNumCsr; i++){
		len = eachCsrLen_h[i];
		startPos = eachCsrStartPos[i];
		for(v = 0; v < len; v++){
			v_csrGD[i] += insGD_h[insId_h[startPos + v]];
			v_csrHess[i] += insHess_h[insId_h[startPos + v]];
			//globalPos++;
		}
	}
	clock_t end = clock();
	printf("compute gd & hess time: %f\n", double(end - start) / CLOCKS_PER_SEC);
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrGD(), v_csrGD.data(), sizeof(double) * totalNumCsr, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrHess(), v_csrHess.data(), sizeof(real) * totalNumCsr, cudaMemcpyHostToDevice));
	delete[] insGD_h;
	delete[] insHess_h;
	delete[] eachCsrStartPos;
	checkCudaErrors(cudaFree(eachCsrStartCurRound));
}
