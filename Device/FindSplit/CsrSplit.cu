/*
 * CsrSplit.cu
 *
 *  Created on: Jul 11, 2017
 *      Author: zeyi
 */

#include "CsrSplit.h"
#include "../Bagging/BagManager.h"
#include "../Bagging/BagCsrManager.h"
#include "../Splitter/DeviceSplitter.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/binarySearch.h"

void CsrCompression(int numofSNode, uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos_d, uint *eachCompressedFeaLen_d,
		uint *eachNodeSizeInCsr_d, uint *eachCsrNodeStartPos_d){
	BagManager bagManager;
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);
	real *fvalue_h = new real[bagManager.m_numFeaValue];
	uint *eachFeaLenEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachFeaStartPosEachNode_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachCsrFeaStartPos_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachCompressedFeaLen_h = new uint[bagManager.m_numFea * numofSNode];
	uint *eachCsrLen_h = new uint[bagManager.m_numFeaValue];
	uint *eachCsrNodeStartPos_h = new uint[numofSNode];
	double *csrGD_h = new double[bagManager.m_numFeaValue];
	real *csrHess_h = new real[bagManager.m_numFeaValue];
	uint *eachNodeSizeInCsr_h = new uint[numofSNode];
	real *csrFvalue_h = new real[bagManager.m_numFeaValue];
	checkCudaErrors(cudaMemcpy(fvalue_h, bagManager.m_pDenseFValueEachBag, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaLenEachNode_h, bagManager.m_pEachFeaLenEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(eachFeaStartPosEachNode_h, bagManager.m_pEachFeaStartPosEachNodeEachBag_d, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyDeviceToHost));

	uint csrId = 0, curFvalueToCompress = 0;
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++){
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
	for(int i = 0; i < bagManager.m_numFea * numofSNode; i++){
		uint prefix = 0;
		for(int l = 0; l < i; l++)
			prefix += eachCompressedFeaLen_h[l];
		eachCsrFeaStartPos_h[i] = prefix;
	}

	for(int i = 0; i < numofSNode; i++){
		int posOfLastFeaThisNode = (i + 1) * bagManager.m_numFea - 1;
		int posOfFirstFeaThisNode = i * bagManager.m_numFea;
		eachNodeSizeInCsr_h[i] = eachCsrFeaStartPos_h[posOfLastFeaThisNode] - eachCsrFeaStartPos_h[posOfFirstFeaThisNode];
		eachNodeSizeInCsr_h[i] += eachCompressedFeaLen_h[posOfLastFeaThisNode];
		eachCsrNodeStartPos_h[i] = eachCsrFeaStartPos_h[posOfFirstFeaThisNode];
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
		uint len = eachCsrLen_h[i];
		for(int v = 0; v < len; v++){
			csrGD_h[i] += gd_h[globalPos];
			csrHess_h[i] += hess_h[globalPos];
			globalPos++;
		}
	}
	checkCudaErrors(cudaMemcpy(eachCompressedFeaStartPos_d, eachCsrFeaStartPos_h, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachCompressedFeaLen_d, eachCompressedFeaLen_h, sizeof(uint) * bagManager.m_numFea * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachCsrNodeStartPos_d, eachCsrNodeStartPos_h, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(eachNodeSizeInCsr_d, eachNodeSizeInCsr_h, sizeof(uint) * numofSNode, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrLen(), eachCsrLen_h, sizeof(uint) * totalNumCsrFvalue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrGD(), csrGD_h, sizeof(double) * totalNumCsrFvalue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrHess(), csrHess_h, sizeof(real) * totalNumCsrFvalue, cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(csrManager.getMutableCsrFvalue(), csrFvalue_h, sizeof(real) * totalNumCsrFvalue, cudaMemcpyDefault));

	printf("org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNumCsrFvalue);

	delete[] fvalue_h;
	delete[] eachFeaLenEachNode_h;
	delete[] eachFeaStartPosEachNode_h;
	delete[] gd_h;
	delete[] hess_h;
	delete[] eachCsrFeaStartPos_h;
	delete[] eachCompressedFeaLen_h;
	delete[] eachCsrLen_h;
	delete[] eachCsrNodeStartPos_h;
	delete[] csrGD_h;
	delete[] csrHess_h;
	delete[] eachNodeSizeInCsr_h;
	delete[] csrFvalue_h;
}

/**
 * @brief: efficient best feature finder
 */
__global__ void LoadFvalueInsId(int numIns, const int *pOrgFvalueInsId, int *pNewFvalueInsId, const unsigned int *pDstIndexEachFeaValue, int numFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load; note that "numFeaValue" needs to be the length of whole dataset
		return;

	//index for scatter
	uint idx = pDstIndexEachFeaValue[gTid];
	if(idx == LARGE_4B_UINT)//instance is in a leaf node
		return;

	CONCHECKER(idx < numFeaValue);

	//scatter: store the feature value ins id.
	CONCHECKER(numIns >= pOrgFvalueInsId[gTid] && pOrgFvalueInsId[gTid] >= 0);
	pNewFvalueInsId[idx] = pOrgFvalueInsId[gTid];
}

__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr, const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea,
						  real *eachCsrFvalueSparse, uint *csrNewLen, uint *eachNewSegLen, uint *eachNodeSizeInCsr, int numSN, uint *eachNodeFvalue){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	if(gTid >= numFeaValue)//thread has nothing to do
		return;

	int insId = preFvalueInsId[gTid];//insId is not -1, as preFvalueInsId is dense.
	if(pInsId2Nid[insId] <= maxNid)//leaf node
		return;
	int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node
	atomicAdd(eachNodeFvalue + pid, 1);
	CONCHECKER(pid < numSN && pid >= 0);
	uint csrId = numCsr;
	RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	uint segId = numFea * preRoundNumSN;
	RangeBinarySearch(csrId, preRoundSegStartPos, numFea * preRoundNumSN, segId);
	uint prePid = segId / numFea;
	uint prePartStartPos = preRoundSegStartPos[prePid * numFea];
	uint numCsrPrePartsAhead = prePartStartPos;
	uint numCsrCurPart;
	if(prePid == preRoundNumSN - 1)
		numCsrCurPart = numCsr - prePartStartPos;
	else
		numCsrCurPart = preRoundSegStartPos[(prePid + 1) * numFea] - prePartStartPos;
	uint posInPart = csrId - numCsrPrePartsAhead;//id in the partition
	uint orgValue;
	//compute len of each csr
	if(pid % 2 == 1){
		uint testLen1 = csrNewLen[numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart];
		orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart] = csrFvalue[csrId];
	}
	else{
		uint testLen2 = csrNewLen[numCsrPrePartsAhead * 2 + posInPart];
		orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + posInPart, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[numCsrPrePartsAhead * 2 + posInPart] = csrFvalue[csrId];
	}

	//compute len of each segment
	if(orgValue == 0){
		uint feaId = segId % numFea;
		CONCHECKER(feaId < numFea);
		uint testLen3 = eachNewSegLen[pid * numFea + feaId];
		uint tempLen = atomicAdd(eachNewSegLen + pid * numFea + feaId, 1);
		uint testLen4 = eachNodeSizeInCsr[pid];
		atomicAdd(eachNodeSizeInCsr + pid, 1);
	}
}

__global__ void map2One(const uint *eachCsrLen, uint numCsr, uint *csrMarker){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrLen[gTid] > 0)
		csrMarker[gTid] = 1;
	else
		csrMarker[gTid] = 0;
}

__global__ void loadDenseCsr(const real *eachCsrFvalueSparse, const uint *eachCsrFeaLen, uint numCsr, uint numCsrThisRound,
							 const uint *csrIdx, real *eachCsrFvalueDense, uint *eachCsrFeaLenDense){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrFeaLen[gTid] != 0){
		uint idx = csrIdx[gTid] - 1;//inclusive scan is used to compute indices.
		CONCHECKER(csrIdx[gTid] <= numCsrThisRound);
		eachCsrFeaLenDense[idx] = eachCsrFeaLen[gTid];
		eachCsrFvalueDense[idx] = eachCsrFvalueSparse[gTid];
	}
}

__global__ void compCsrGDHess(const int *preFvalueInsId, uint numUsefulFvalue, const uint *eachCsrStart, uint numCsr,
							  const real *pInsGrad, const real *pInsHess, int numIns,
							  double *csrGD, real *csrHess){
	uint gTid = GLOBAL_TID();
	if(gTid >= numUsefulFvalue)
		return;
	uint csrId = numCsr;
	RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	int insId = preFvalueInsId[gTid];
	CONCHECKER(insId >= 0 && insId < numIns);
	double temp = pInsGrad[insId];
	atomicAdd(csrGD + csrId, temp);
	atomicAdd(csrHess + csrId, pInsHess[insId]);
}
