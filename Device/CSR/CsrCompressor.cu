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
#include "CsrSplit.h"
#include "BagCsrManager.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"

using std::vector;

uint *CsrCompressor::eachFeaLenEachNode_h = NULL;
uint *CsrCompressor::eachFeaStartPosEachNode_h = NULL;
uint CsrCompressor::eachNodeSizeInCsr_h = 0;
int *CsrCompressor::insId_h = NULL;
uint CsrCompressor::totalOrgNumCsr = 0;

uint *CsrCompressor::pCsrFeaStartPos_d = NULL;
uint *CsrCompressor::pCsrFeaLen_d = NULL;
uint *CsrCompressor::pCsrLen_d = NULL;
real *CsrCompressor::pCsrFvalue_d = NULL;
uint *CsrCompressor::pCsrStart_d = NULL;
bool CsrCompressor::bUseCsr = false;
real *CsrCompressor::pOrgFvalue = NULL;

CsrCompressor::CsrCompressor(){
	if(pCsrFeaStartPos_d != NULL || bUseCsr == false)
		return;
	GBDTGPUMemManager manager;
	uint numFea = manager.m_numofFea;
	uint numFeaValue = manager.m_numFeaValue;

	uint *eachCsrFeaStartPos_h = new uint[numFea];
	uint *eachCompressedFeaLen_h = new uint[numFea];
	uint *eachCsrLen_h = new uint[numFeaValue];
	real *csrFvalue_h = new real[numFeaValue];

	eachNodeSizeInCsr_h = 0;
	uint csrId = 0, curFvalueToCompress = 0;
	for(int i = 0; i < numFea; i++){
		eachCompressedFeaLen_h[i] = 0;
		uint feaLen = eachFeaLenEachNode_h[i];
		uint feaStart = eachFeaStartPosEachNode_h[i];
		if(feaLen == 0)continue;
		csrFvalue_h[csrId] = pOrgFvalue[feaStart];
		eachCsrLen_h[csrId] = 1;
		eachCompressedFeaLen_h[i] = 1;
		for(int l = 1; l < feaLen; l++){
			curFvalueToCompress++;
			if(fabs(pOrgFvalue[feaStart + l] - csrFvalue_h[csrId]) > DeviceSplitter::rt_eps){
				eachCompressedFeaLen_h[i]++;
				csrId++;
				csrFvalue_h[csrId] = pOrgFvalue[feaStart + l];
				eachCsrLen_h[csrId] = 1;
			}
			else
				eachCsrLen_h[csrId]++;
		}
		csrId++;
		curFvalueToCompress++;
	}
	for(int i = 0; i < numFea; i++){
		uint prefix = 0;
		for(int l = 0; l < i; l++)
			prefix += eachCompressedFeaLen_h[l];
		eachCsrFeaStartPos_h[i] = prefix;
	}

	totalOrgNumCsr = csrId;
	eachNodeSizeInCsr_h = totalOrgNumCsr;
	printf("org=%u v.s. csr=%u\n", manager.m_numFeaValue, totalOrgNumCsr);
//	PROCESS_ERROR(totalOrgNumCsr < manager.m_numFeaValue);
	if(totalOrgNumCsr * 4 > manager.m_numFeaValue){
		bUseCsr = false;

		delete[] eachCsrFeaStartPos_h;
		delete[] eachCompressedFeaLen_h;
		delete[] eachCsrLen_h;
		delete[] csrFvalue_h;
		return;
	}

	checkCudaErrors(cudaMalloc((void**)&pCsrFeaStartPos_d, sizeof(uint) * numFea));
	checkCudaErrors(cudaMalloc((void**)&pCsrFeaLen_d, sizeof(uint) * numFea));
	checkCudaErrors(cudaMalloc((void**)&pCsrLen_d, sizeof(uint) * totalOrgNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue_d, sizeof(real) * totalOrgNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrStart_d, sizeof(uint) * totalOrgNumCsr));
	checkCudaErrors(cudaMemcpy(pCsrFeaStartPos_d, eachCsrFeaStartPos_h, sizeof(uint) * numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFeaLen_d, eachCompressedFeaLen_h, sizeof(uint) * numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pCsrLen_d, eachCsrLen_h, sizeof(uint) * totalOrgNumCsr, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(pCsrFvalue_d, csrFvalue_h, sizeof(real) * totalOrgNumCsr, cudaMemcpyDefault));
	thrust::exclusive_scan(thrust::device, pCsrLen_d, pCsrLen_d + totalOrgNumCsr, pCsrStart_d);

	delete[] eachCsrFeaStartPos_h;
	delete[] eachCompressedFeaLen_h;
	delete[] eachCsrLen_h;
	delete[] csrFvalue_h;
}

void CsrCompressor::CsrCompression(uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos_d, uint *eachCompressedFeaLen_d,
								   uint *eachNodeSizeInCsr_d, uint *eachCsrNodeStartPos_d, real *pCsrFvalue, uint *pCsrLen){
	BagManager bagManager;
	GBDTGPUMemManager manager;
	BagCsrManager csrManager(manager.m_numofFea, bagManager.m_maxNumSplittable, manager.m_numFeaValue);
	totalNumCsrFvalue = totalOrgNumCsr;
	//compute csr gd and hess
	checkCudaErrors(cudaMemcpy(eachCompressedFeaStartPos_d, pCsrFeaStartPos_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(eachCompressedFeaLen_d, pCsrFeaLen_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(pCsrFvalue, pCsrFvalue_d, sizeof(real) * totalOrgNumCsr, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(pCsrLen, pCsrLen_d, sizeof(uint) * totalOrgNumCsr, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemset(eachCsrNodeStartPos_d, 0, sizeof(uint)));
	checkCudaErrors(cudaMemcpy(eachNodeSizeInCsr_d, &eachNodeSizeInCsr_h, sizeof(uint), cudaMemcpyHostToDevice));
}
