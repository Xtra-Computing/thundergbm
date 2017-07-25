/*
 * CsrCompressor.cu
 *
 *  Created on: Jul 25, 2017
 *      Author: zeyi
 */
#include <iostream>
#include <vector>

#include "CsrCompressor.h"
#include "BagCsrManager.h"
#include "../Bagging/BagManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"

using std::vector;

void CsrCompressor::CsrCompression(uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos_d, uint *eachCompressedFeaLen_d,
								   uint *eachNodeSizeInCsr_d, uint *eachCsrNodeStartPos_d){
	BagManager bagManager;
	GBDTGPUMemManager manager;
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);
	real *fvalue_h = new real[bagManager.m_numFeaValue];
	uint *eachFeaLenEachNode_h = new uint[bagManager.m_numFea];
	uint *eachFeaStartPosEachNode_h = new uint[bagManager.m_numFea];
	uint *eachCsrFeaStartPos_h = new uint[bagManager.m_numFea];
	uint *eachCompressedFeaLen_h = new uint[bagManager.m_numFea];
	uint *eachCsrLen_h = new uint[bagManager.m_numFeaValue];
	uint eachCsrNodeStartPos_h;
	uint eachNodeSizeInCsr_h;
	real *csrFvalue_h = new real[bagManager.m_numFeaValue];
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

	int posOfLastFeaThisNode = bagManager.m_numFea - 1;
	int posOfFirstFeaThisNode = 0;
	eachNodeSizeInCsr_h = eachCsrFeaStartPos_h[posOfLastFeaThisNode] - eachCsrFeaStartPos_h[posOfFirstFeaThisNode];
	eachNodeSizeInCsr_h += eachCompressedFeaLen_h[posOfLastFeaThisNode];
	eachCsrNodeStartPos_h = eachCsrFeaStartPos_h[posOfFirstFeaThisNode];
//	printf("node %d starts %u, len=%u\n", i, eachCsrNodeStartPos[i], eachNodeSizeInCsr[i]);

	totalNumCsrFvalue = csrId;
	printf("org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNumCsrFvalue);
//	printf("csrLen=%u, totalLen=%u, numofFeaValue=%u\n", csrId, totalLen, bagManager.m_numFeaValue);
	PROCESS_ERROR(totalNumCsrFvalue < bagManager.m_numFeaValue);
	//compute csr gd and hess
	int *insId_h = new int[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMemcpy(insId_h, manager.m_pDInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(eachCompressedFeaStartPos_d, eachCsrFeaStartPos_h, sizeof(uint) * bagManager.m_numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachCompressedFeaLen_d, eachCompressedFeaLen_h, sizeof(uint) * bagManager.m_numFea, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachCsrNodeStartPos_d, &eachCsrNodeStartPos_h, sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachNodeSizeInCsr_d, &eachNodeSizeInCsr_h, sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrFvalue(), csrFvalue_h, sizeof(real) * totalNumCsrFvalue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrLen(), eachCsrLen_h, sizeof(uint) * totalNumCsrFvalue, cudaMemcpyDefault));

	delete[] fvalue_h;
	delete[] eachFeaLenEachNode_h;
	delete[] eachFeaStartPosEachNode_h;
	delete[] eachCsrFeaStartPos_h;
	delete[] eachCompressedFeaLen_h;
	delete[] csrFvalue_h;

	//need to compute for every new tree
	real *insGD_h = new real[bagManager.m_numIns];
	real *insHess_h = new real[bagManager.m_numIns];
	checkCudaErrors(cudaMemcpy(insGD_h, bagManager.m_pInsGradEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(insHess_h, bagManager.m_pInsHessEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDefault));
	vector<double> v_csrGD(totalNumCsrFvalue, 0);
	vector<real> v_csrHess(totalNumCsrFvalue, 0);
	uint globalPos = 0;
	for(int i = 0; i < totalNumCsrFvalue; i++){
		uint len = eachCsrLen_h[i];
		for(int v = 0; v < len; v++){
			v_csrGD[i] += insGD_h[insId_h[globalPos]];
			v_csrHess[i] += insHess_h[insId_h[globalPos]];
			globalPos++;
		}
	}
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrGD(), v_csrGD.data(), sizeof(double) * totalNumCsrFvalue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrHess(), v_csrHess.data(), sizeof(real) * totalNumCsrFvalue, cudaMemcpyDefault));
	delete[] insGD_h;
	delete[] insHess_h;
	delete[] eachCsrLen_h;
	delete[] insId_h;
}
