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
using std::set;
uint numofDenseValue_previous;
bool firstTime = true;
uint *pCSRKey, *pCSRMultableKey;
uint num_key = 29053924;
void DeviceSplitter::FeaFinderAllNode2(void *pStream, int bagId)
{
	cudaDeviceSynchronize();
	GBDTGPUMemManager manager;
	BagManager bagManager;
	BagCsrManager csrManager(bagManager.m_numFea, bagManager.m_maxNumSplittable, bagManager.m_numFeaValue);
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];

//	if(csrManager.curNumCsr > num_key)
//	{
//		printf("oh shitttttttttttttttttttttttttttttttttttttttttttttttttttttt\n");
//		exit(0);
//	}
//	if(firstTime == true){
//		checkCudaErrors(cudaMalloc((void**)&pCSRKey, sizeof(uint) * num_key));
//		checkCudaErrors(cudaMalloc((void**)&pCSRMultableKey, sizeof(uint) * num_key));
//		firstTime = false;
//	}

	IndexComputer indexComp;
	indexComp.AllocMem(bagManager.m_numFea, numofSNode, bagManager.m_maxNumSplittable);
	
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);
	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);

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
	int maxNumFeaValueOneNode = -1;
	clock_t csr_len_t = clock();
	if(numofSNode > 1)
	{
		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t comIdx_start = clock();
		//compute gather index via GPUs
		indexComp.ComputeIdxGPU(numofSNode, maxNumofSplittable, bagId);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		//copy # of feature values of each node
		uint *pTempNumFvalueEachNode = bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
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
//comp each fea hess
uint *oldCsrLen_h = new uint[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(oldCsrLen_h, pOldCsrLen_d, sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
uint *eachFeaStart_h = new uint[numofSNode * bagManager.m_numFea];
checkCudaErrors(cudaMemcpy(eachFeaStart_h, csrManager.pEachCsrFeaStartPos, sizeof(uint) * numofSNode * bagManager.m_numFea, cudaMemcpyDeviceToHost));
uint *eachFeaLen_h = new uint[numofSNode * bagManager.m_numFea];
checkCudaErrors(cudaMemcpy(eachFeaLen_h, csrManager.pEachCsrFeaLen, sizeof(uint) * numofSNode * bagManager.m_numFea, cudaMemcpyDeviceToHost));

for(int f = 0; f < bagManager.m_numFea; f++){
	uint sum = 0;
	for(int n = 0; n < numofSNode; n++){
		uint start = eachFeaStart_h[n * bagManager.m_numFea + f];
		uint len = eachFeaLen_h[n * bagManager.m_numFea + f];
		for(int v = start; v < start + len; v++){
			sum += oldCsrLen_h[v];
		}
	}
	printf("f=%d: old cnt=%d\t", f, sum);
}
printf("--------------------------------------------------------\n");

		printf("done index comp\n");

		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, csrManager.getMutableCsrStart());

uint *pCsrNewLen_d;// = (uint*)(indexComp.histogram_d.addr);
//uint *pCsrNewLen_d = (uint*)(indexComp.histogram_d.addr);
checkCudaErrors(cudaMallocHost((void**)&pCsrNewLen_d, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(pCsrNewLen_d, 0, sizeof(uint) * csrManager.curNumCsr * 2));
		checkCudaErrors(cudaMemset(csrManager.pEachCsrFeaLen, 0, sizeof(uint) * bagManager.m_numFea * numofSNode));
		dim3 dimNumofBlockToCsrLen;
		uint blockSizeCsrLen = 128;

//test
int *fvInsId = new int[bagManager.m_numFeaValue];
int *insId2Nid = new int[bagManager.m_numIns];
real *insHess = new real[bagManager.m_numIns];
checkCudaErrors(cudaMemcpy(fvInsId, csrManager.preFvalueInsId, sizeof(int) * bagManager.m_numFeaValue, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(insId2Nid, bagManager.m_pInsIdToNodeIdEachBag, sizeof(int) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(insHess, bagManager.m_pInsHessEachBag, sizeof(real) * bagManager.m_numIns, cudaMemcpyDeviceToHost));
real *sumHess = new real[numofSNode];
memset(sumHess, 0, sizeof(real) * numofSNode);
/*
vector<vector<int> > vInsId;
for(int i = 0; i < numofSNode; i++){
	vector<int> temp(bagManager.m_numIns, 0);
	vInsId.push_back(temp);
}
for(int fv = 0; fv < bagManager.m_numFeaValue; fv++){
	int insId = fvInsId[fv];
	int pid = insId2Nid[insId] - bagManager.m_pPreMaxNid_h[bagId] - 1;
	vInsId[pid][insId] = 1;
	real hess = insHess[insId];
	if(pid < 0 || pid >= numofSNode){
		printf("oh shittttttttttttttttttt pid=%d, numofSN=%d, premax=%d\n", pid, numofSNode, bagManager.m_pPreMaxNid_h[bagId]);
	}
	else 
		sumHess[pid] += 1; 
}
for(int i = 0; i < numofSNode; i++){
	int nodeSum = 0;
	for(int ins = 0; ins < bagManager.m_numIns; ins++)
		nodeSum += vInsId[i][ins];
	printf("pid=%d, sum_hess=%f, sum_ins=%d\t", i, sumHess[i], nodeSum);
}
printf("###################### , numfeavalue=%d\n", bagManager.m_numFeaValue);
*/
cudaDeviceSynchronize();
		dimNumofBlockToCsrLen.x = (numofDenseValue_previous + blockSizeCsrLen - 1) / blockSizeCsrLen;
		newCsrLenFvalue<<<dimNumofBlockToCsrLen, blockSizeCsrLen, blockSizeCsrLen * sizeof(uint)>>>(
				csrManager.preFvalueInsId, numofDenseValue_previous,
				bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns,
				bagManager.m_pPreMaxNid_h[bagId], csrManager.getCsrStart(),
				csrManager.getCsrFvalue(), csrManager.curNumCsr,
				csrManager.pEachCsrFeaStartPos, bagManager.m_pPreNumSN_h[bagId],
				bagManager.m_numFea, csrManager.getCsrKey(), pCsrNewLen_d, pCsrId2Pid);
//				bagManager.m_numFea, pCSRKey, pCsrNewLen_d, pCsrId2Pid);

		int *p_key_h = new int[csrManager.curNumCsr];
//		checkCudaErrors(cudaMemcpy(p_key_h, pCSRKey, sizeof(int) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
//		for(int i = 0; i < csrManager.curNumCsr; i++)
//			if(p_key_h[i] != 0)
//				printf("key=%d \t", p_key_h[i]);

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
				   bagManager.m_pPreNumSN_h[bagId], bagManager.m_numFea, csrManager.getCsrKey(), pOldCsrLen_d, pCsrId2Pid,
//				   bagManager.m_pPreNumSN_h[bagId], bagManager.m_numFea, pCSRKey, pOldCsrLen_d, pCsrId2Pid,
				   pCsrFvalueSpare, pCsrNewLen_d, csrManager.pEachCsrFeaLen, csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos,
				   hess_cnt_d);
		GETERROR("after fillFvalue");
		cudaDeviceSynchronize();
uint *hess_cnt_h = new uint[bagManager.m_numFea];
checkCudaErrors(cudaMemcpy(hess_cnt_h, hess_cnt_d, sizeof(uint) * bagManager.m_numFea, cudaMemcpyDeviceToHost));
for(int i = 0; i < bagManager.m_numFea; i++){
	printf("f=%d: cnt=%d\t", i, hess_cnt_h[i]);
}
printf("\n******************************************************************** numofCsr=%d\n", csrManager.curNumCsr);

uint *csrNewLen_h = new uint[csrManager.curNumCsr * 2];
checkCudaErrors(cudaMemcpy(csrNewLen_h, pCsrNewLen_d, sizeof(uint) * csrManager.curNumCsr * 2, cudaMemcpyDeviceToHost));

//double check new length and old length
uint *csrId2SegId_h = new uint[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(csrId2SegId_h, csrManager.getCsrKey(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
uint preNumSeg = bagManager.m_numFea * bagManager.m_pPreNumSN_h[bagId];
uint *preRoundSegStartPos_h = new uint[preNumSeg];
checkCudaErrors(cudaMemcpy(preRoundSegStartPos_h, csrManager.pEachCsrFeaStartPos, sizeof(uint) * preNumSeg, cudaMemcpyDeviceToHost));

bool temp = false;
for(int csrId = 0; csrId < csrManager.curNumCsr; csrId++){
	int numCsr = csrManager.curNumCsr;
	uint numFea = bagManager.m_numFea;
	uint segId = csrId2SegId_h[csrId];
	uint prePid = segId / numFea;
	uint prePartStartPos = preRoundSegStartPos_h[prePid * numFea];
	uint numCsrPrePartsAhead = prePartStartPos;
	if(csrId - numCsrPrePartsAhead < 0){
		printf("oh shit in checking if new length equals to old length\n");
		exit(0);
	}
	uint posInPart = csrId - numCsrPrePartsAhead;//id in the partition
	uint numCsrCurPart;
	uint preRoundNumSN = bagManager.m_pPreNumSN_h[bagId];
	if(prePid == preRoundNumSN - 1)
		numCsrCurPart = numCsr - prePartStartPos;
	else
		numCsrCurPart = preRoundSegStartPos_h[(prePid + 1) * numFea] - prePartStartPos;

	uint basePos = numCsrPrePartsAhead * 2 + posInPart;
	uint first = basePos;
	uint second = basePos + numCsrCurPart;
	uint feaId = segId % numFea;
	if(oldCsrLen_h[csrId] != csrNewLen_h[first] + csrNewLen_h[second]){
		printf("oh shift: old=%d, new1=%d, new2=%d; "
				"csrId=%d, first=%d, second=%d\n",
				oldCsrLen_h[csrId], csrNewLen_h[first], csrNewLen_h[second],
				csrId, first, second);
		printf("pre num seg=%d, feaId=%u, prePid=%u, segId=%u\n", preNumSeg, feaId, prePid, segId);
		temp = true;
	}
	if(csrId == 927002){
		printf("oh shift: old=%d, new1=%d, new2=%d; "
				"csrId=%d, first=%d, second=%d\n",
				oldCsrLen_h[csrId], csrNewLen_h[first], csrNewLen_h[second],
				csrId, first, second);
		printf("pre num seg=%d, feaId=%u, prePid=%u, segId=%u\n", preNumSeg, feaId, prePid, segId);
	}
	if(temp == true)
		exit(0);
}

//end checking new length

uint numOldNode = bagManager.m_pPreNumSN_h[bagId];
uint *oldNodeSize = new uint[numOldNode];
checkCudaErrors(cudaMemcpy(oldNodeSize, csrManager.pEachNodeSizeInCsr, sizeof(uint) * numOldNode, cudaMemcpyDeviceToHost));
uint *oldNodeStartPos_h = new uint[numOldNode];
checkCudaErrors(cudaMemcpy(oldNodeStartPos_h, csrManager.pEachCsrNodeStartPos, sizeof(uint) * numOldNode, cudaMemcpyDeviceToHost));
uint total_csr = 0;
for(int f = 0; f < bagManager.m_numFea; f++){
	uint sum = 0;
	uint csr_cnt = 0;
	for(int n = 0; n < numofSNode; n++){
		uint nodeStart = oldNodeStartPos_h[n] * 2;
		uint previousNodesSize = oldNodeStartPos_h[n];
		uint curNodeSize = oldNodeSize[n];
		uint start = eachFeaStart_h[n * bagManager.m_numFea + f] + previousNodesSize;
		uint len = eachFeaLen_h[n * bagManager.m_numFea + f];
		for(int v = start; v < start + len; v++){
			sum += csrNewLen_h[v];
			sum += csrNewLen_h[v + curNodeSize];
			if(csrNewLen_h[v] > 0)csr_cnt++;
			if(csrNewLen_h[v + curNodeSize] > 0)csr_cnt++;
		}
	}
	printf("f=%d: sparse cnt=%d, # csr=%d\t", f, sum, csr_cnt);
	total_csr += csr_cnt;
}
printf("\n################################################################################### csr total=%d\n", total_csr);

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
uint *pCsrDest_h = new uint[previousNumCsr * 2];
checkCudaErrors(cudaMemcpy(pCsrDest_h, pCsrMarker, sizeof(uint) * previousNumCsr * 2, cudaMemcpyDeviceToHost));
uint *h_eachCsrLen = new uint[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(h_eachCsrLen, csrManager.getMutableCsrLen(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(pCsrMarker));
uint total_csr_dense = 0;
for(int f = 0; f < bagManager.m_numFea; f++){
	uint sum = 0;
	uint csr_cnt = 0;
	for(int n = 0; n < numofSNode; n++){
		uint nodeStart = oldNodeStartPos_h[n] * 2;
		uint previousNodesSize = oldNodeStartPos_h[n];
		uint curNodeSize = oldNodeSize[n];
		uint start = eachFeaStart_h[n * bagManager.m_numFea + f] + previousNodesSize;
		uint len = eachFeaLen_h[n * bagManager.m_numFea + f];
		for(int v = start; v < start + len; v++){
			sum += h_eachCsrLen[pCsrDest_h[v]];
			sum += h_eachCsrLen[pCsrDest_h[v + curNodeSize]];
			if(h_eachCsrLen[pCsrDest_h[v]] > 0)csr_cnt++;
			if(h_eachCsrLen[pCsrDest_h[v + curNodeSize]] > 0)csr_cnt++;
		}
	}
	printf("f=%d: desne by sparse cnt=%d, # csr=%d\t", f, sum, csr_cnt);
	total_csr_dense += csr_cnt;
}
printf("\n &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& csr total=%d\n", total_csr_dense);

//print dense csr len
if(numofSNode > 2){
	for(int i = 927000; i < 927005; i++){
		printf("i=%d, len=%d\n", i, h_eachCsrLen[i]);
	}
	printf("totalCsr=%d\n", csrManager.curNumCsr);
	exit(0);
}


uint *h_startPos = new uint[numofSNode * bagManager.m_numFea];
uint *h_len = new uint[numofSNode * bagManager.m_numFea];
checkCudaErrors(cudaMemcpy(h_startPos, csrManager.pEachCsrFeaStartPos, sizeof(uint) * numofSNode * bagManager.m_numFea, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(h_len, csrManager.pEachCsrFeaLen, sizeof(uint) * numofSNode * bagManager.m_numFea, cudaMemcpyDeviceToHost));
for(int f = 0; f < bagManager.m_numFea; f++){
	uint sum = 0;
	uint csr_cnt = 0;
	for(int n = 0; n < numofSNode; n++){
		uint start = h_startPos[n * bagManager.m_numFea + f];
		uint len = h_len[n * bagManager.m_numFea + f];
		for(int v = start; v < start + len; v++){
			sum += h_eachCsrLen[v];
			csr_cnt++;
		}
	}
	printf("f=%d: new cnt=%d, # csr=%d\t", f, sum, csr_cnt);
}
printf("\n===============================================================================\n");

uint *h_csrNewLen = new uint[previousNumCsr * 2];
checkCudaErrors(cudaMemcpy(h_csrNewLen, pCsrNewLen_d, sizeof(uint) * previousNumCsr * 2, cudaMemcpyDeviceToHost));
uint totalInCsr = 0;
for(int i = 0; i < previousNumCsr * 2; i++)
	totalInCsr += h_csrNewLen[i];
uint totalHessLen = 0;
for(int f = 0; f < numofSNode * bagManager.m_numFea; f++)
	totalHessLen += h_len[f];
printf("................................................. fea len sum=%d, total in css=%d\n", totalHessLen, totalInCsr);

		thrust::exclusive_scan(thrust::device, csrManager.pEachNodeSizeInCsr, csrManager.pEachNodeSizeInCsr + numofSNode, csrManager.pEachCsrNodeStartPos);
		numofDenseValue_previous = thrust::reduce(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);//number of dense fvalues.
		uint *pCsrStartCurRound = (uint*)indexComp.partitionMarker.addr;
		thrust::exclusive_scan(thrust::device, csrManager.getCsrLen(), csrManager.getCsrLen() + csrManager.curNumCsr, pCsrStartCurRound);
		PROCESS_ERROR(csrManager.curNumCsr <= bagManager.m_numFeaValue);
		cudaDeviceSynchronize();
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
		cudaDeviceSynchronize();
		CsrCompressor compressor;
		csrManager.curNumCsr = compressor.totalOrgNumCsr;
		compressor.CsrCompression(csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen,
								  csrManager.pEachNodeSizeInCsr, csrManager.pEachCsrNodeStartPos, csrManager.getMutableCsrFvalue(), csrManager.getMutableCsrLen());
	}
	//need to compute for every new tree
	printf("reserve memory\n");
	if(indexComp.histogram_d.reservedSize < csrManager.curNumCsr * 4){//make sure enough memory for reuse
		printf("reallocate memory for histogram (sn=%u): %u v.s. %u.......\n", numofSNode, indexComp.histogram_d.reservedSize, csrManager.curNumCsr * 4);
		indexComp.histogram_d.reserveSpace(csrManager.curNumCsr * 4, sizeof(uint));
	}
	cudaDeviceSynchronize();
	double *pGD_d = (double*)indexComp.histogram_d.addr;//reuse memory; must be here, as curNumCsr may change in different level.
//real *pHess_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 2);//reuse memory
double *pHess_d;
checkCudaErrors(cudaMalloc((void**)&pHess_d, sizeof(double) * csrManager.curNumCsr));
	real *pGain_d = (real*)(((uint*)indexComp.histogram_d.addr) + csrManager.curNumCsr * 3);
	checkCudaErrors(cudaMemset(pGD_d, 0, sizeof(double) * csrManager.curNumCsr));
	checkCudaErrors(cudaMemset(pHess_d, 0, sizeof(double) * csrManager.curNumCsr));
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
	/*printf("csrkey add = %d\n", csrManager.getCsrKey());
	if(9594328 == csrManager.curNumCsr){
		printf("try copying before compGDHess\n");
		int *tempKey = new int[csrManager.curNumCsr];
		checkCudaErrors(cudaMemcpy((void**)&tempKey, csrManager.getCsrKey(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
		printf("done copying\n");
	}*/
//expanding csr
uint *eachCsrLen_h = new uint[csrManager.curNumCsr];
uint *eachCsrStart_h = new uint[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(eachCsrLen_h, csrManager.getCsrLen(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(eachCsrStart_h, pCsrStartPos_d, sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
uint totalEle1 = 0, totalEle2 = 0;
totalEle1 = eachCsrStart_h[csrManager.curNumCsr - 1] + eachCsrLen_h[csrManager.curNumCsr - 1];
for(int i = 0; i < csrManager.curNumCsr; i++){
	totalEle2 += eachCsrLen_h[i];
}
printf("expanding csr ...... total1=%d, total2=%d, dimNumofBlockForGD=%d\n", totalEle1, totalEle2, dimNumofBlockForGD.x);
uint *access_cnt_d;
checkCudaErrors(cudaMallocManaged((void**)&access_cnt_d, sizeof(uint) * totalEle1));
checkCudaErrors(cudaMemset(access_cnt_d, 0, sizeof(uint) * totalEle1));

	ComputeGDHess<<<dimNumofBlockForGD, blockSizeForGD, sharedMemSizeForGD>>>(csrManager.getCsrLen(), pCsrStartPos_d,
			bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
			bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns,
			csrManager.preFvalueInsId, pGD_d, pHess_d, access_cnt_d);
	cudaDeviceSynchronize();
	GETERROR("after ComputeGD");
	clock_t csr_len_end = clock();
	total_csr_len_t += (csr_len_end - csr_len_t);
int totalAccess = 0;
for(int i = 0; i < totalEle1; i++){
	totalAccess += access_cnt_d[i];
}
printf("oh shittt.............................., total access cnt=%d\n", totalAccess);
double *allHess_h = new double[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(allHess_h, pHess_d, sizeof(double) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
double testTotalHess = 0;
uint totalEle3 = 0;
for(int i = 0; i < csrManager.curNumCsr; i++){
	testTotalHess += allHess_h[i];
	totalEle3 += eachCsrLen_h[i];
	if(eachCsrLen_h[i] != allHess_h[i])
		printf("len=%d, hess=%f, idx=%d\n", eachCsrLen_h[i], allHess_h[i], i);
}
printf("########################## test total hess =%f, totalEle3=%d\n", testTotalHess, totalEle3);
checkCudaErrors(cudaFree(access_cnt_d));
	//cout << "prefix sum" << endl;
	printf("prefix sum\n");
	int numSeg = bagManager.m_numFea * numofSNode;
	clock_t start_scan = clock();

	//construct keys for exclusive scan
/*	printf("csrkey add = %d\n", csrManager.getCsrKey());
	if(9594328 == csrManager.curNumCsr){
		int *tempKey = new int[csrManager.curNumCsr];
		checkCudaErrors(cudaMemcpy((void**)&tempKey, csrManager.getCsrKey(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
	}
	*/
	checkCudaErrors(cudaMemset(csrManager.getMutableCsrKey(), -1, sizeof(uint) * csrManager.curNumCsr));
//	checkCudaErrors(cudaMemset(pCSRMultableKey, -1, sizeof(uint) * csrManager.curNumCsr));
	printf("done constructing key... number of segments is %d\n", numSeg);
//	printf("csrkey add = %d\n", csrManager.getCsrKey());

	//set keys by GPU
	uint maxSegLen = 0;
	uint *pMaxLen = thrust::max_element(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numSeg);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(&maxSegLen, pMaxLen, sizeof(uint), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	MEMSET(csrManager.getMutableCsrKey(), -1, csrManager.curNumCsr * sizeof(uint));
	dim3 dimNumofBlockToSetKey;
	dimNumofBlockToSetKey.x = numSeg;
	uint blockSize = 128;
	dimNumofBlockToSetKey.y = (maxSegLen + blockSize - 1) / blockSize;
//	printf("%%%%%%%%%%%%%%%%%%% kernel configurations %d segments, blksize=%d\n", numSeg, blockSize);
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
//	checkCudaErrors(cudaMemcpy(pCSRKey, pCSRMultableKey, sizeof(int) * csrManager.curNumCsr, cudaMemcpyDeviceToDevice));
	cudaDeviceSynchronize();
	int *h_key = new int[csrManager.curNumCsr];
	memset(h_key, -1, sizeof(int) * csrManager.curNumCsr);
//	checkCudaErrors(cudaMemcpy(h_key, pCSRKey, sizeof(int) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
//	for(int i = 0; i < csrManager.curNumCsr; i++)
//		if(h_key[i] != 0)
//			printf("???????????????????? key=%d \t", h_key[i]);
	cudaDeviceSynchronize();
//	if(csrManager.curNumCsr > 8479362)
//		printf("done set key, key=%d\n", h_key[8479362]);
	uint *h_startPos = new uint[numSeg];
	uint *h_len = new uint[numSeg];
	checkCudaErrors(cudaMemcpy(h_startPos, csrManager.pEachCsrFeaStartPos, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_len, csrManager.pEachCsrFeaLen, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));

	bool first = true;
	int keyChange = 1;
	for(int i = 0; i < csrManager.curNumCsr; i++){
		if(h_key[i] == keyChange || h_key[i] > 32){
			keyChange++;
//			printf("############$$$$$$$$$ my key=%d, i=%d, key+1=%d\n", h_key[i], i, h_key[i + 1]);
		}
		if(i == 8479362 || i == 12848)
		{
//			printf("############ my key=%d, i=%d, key+1=%d\n", h_key[i], i, h_key[i + 1]);
//			if(i == 8479362)
//			for(int f = 0; f < numSeg; f++)
//				printf("fea %d: start = %d, len=%d\n", f, h_startPos[f], h_len[f]);
		}
//		if(h_key[i] != 0 && curNumofNode == 3)
//			printf("key = %d\t", h_key[i]);
		/*if((h_key[i] >= numSeg || h_key[i] < 0)){
			printf("########### key=%d, i=%d, numSeg=%d, numCsr=%d, h_startPos[%d]=%d, len=%d, h_startPos[%d]=%d, len=%d, h_startPos[%d]=%d, len=%d\n",
					h_key[i], i, numSeg, csrManager.curNumCsr,
					numSeg-2, h_startPos[numSeg-2], h_len[numSeg - 2],
					numSeg-1, h_startPos[numSeg-1], h_len[numSeg - 1],
					0, h_startPos[0], h_len[0]);
			exit(0);
		}*/
	}

double *gd_h = new double[csrManager.curNumCsr];
double *hess_h = new double[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(gd_h, pGD_d, sizeof(double) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(hess_h, pHess_d, sizeof(double) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
double totalHess = 0;
for(int i = 0; i < csrManager.curNumCsr; i++){
	totalHess += hess_h[i];
}
printf("####################################### totalHess=%f\n", totalHess);

	//compute prefix sum for gd and hess (more than one arrays)
	thrust::inclusive_scan_by_key(thrust::device, csrManager.getCsrKey(), csrManager.getCsrKey() + csrManager.curNumCsr, pGD_d, pGD_d);//in place prefix sum
	thrust::inclusive_scan_by_key(thrust::device, csrManager.getCsrKey(), csrManager.getCsrKey() + csrManager.curNumCsr, pHess_d, pHess_d);
//	thrust::inclusive_scan_by_key(thrust::device, pCSRKey, pCSRKey + csrManager.curNumCsr, pGD_d, pGD_d);//in place prefix sum
//	thrust::inclusive_scan_by_key(thrust::device, pCSRKey, pCSRKey + csrManager.curNumCsr, pHess_d, pHess_d);
double *sum_gd_h = new double[csrManager.curNumCsr];
double *sum_hess_h = new double[csrManager.curNumCsr];
checkCudaErrors(cudaMemcpy(sum_gd_h, pGD_d, sizeof(double) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(sum_hess_h, pHess_d, sizeof(double) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));

	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

	//compute gain; default to left or right
	bool *default2Right = (bool*)indexComp.partitionMarker.addr;
	checkCudaErrors(cudaMemset(default2Right, 0, sizeof(bool) * csrManager.curNumCsr));//this is important (i.e. initialisation)
	checkCudaErrors(cudaMemset(pGain_d, 0, sizeof(real) * csrManager.curNumCsr));

//	cout << "compute gain" << endl;
	printf("compute gain\n");
	uint test = thrust::reduce(thrust::device, csrManager.pEachCsrFeaLen, csrManager.pEachCsrFeaLen + numSeg);
	clock_t start_comp_gain = clock();
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(csrManager.curNumCsr, blockSizeComGain, dimNumofBlockToComGain);
	cudaDeviceSynchronize();
	GETERROR("before ComputeGainDense");
	uint *pid2SNPos = new uint[10];
	checkCudaErrors(cudaMemcpy(pid2SNPos, bagManager.m_pPartitionId2SNPosEachBag, sizeof(uint) * 2, cudaMemcpyDeviceToHost));
//	for(int i = 0; i < 2; i++)
//		printf("pid2SNPos[%d]=%d\t", i, pid2SNPos[i]);
//	printf("\n");
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable,
											DeviceSplitter::m_lambda, pGD_d, pHess_d, csrManager.getCsrFvalue(),
											csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, csrManager.getCsrKey(), bagManager.m_numFea,
//											csrManager.curNumCsr, csrManager.pEachCsrFeaStartPos, csrManager.pEachCsrFeaLen, pCSRKey, bagManager.m_numFea,
											pGain_d, default2Right);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	GETERROR("after ComputeGainDense");
	checkCudaErrors(cudaMemcpy(pid2SNPos, bagManager.m_pPartitionId2SNPosEachBag, sizeof(uint) * 2, cudaMemcpyDeviceToHost));
//	for(int i = 0; i < 2; i++)
//		printf("pid2SNPos[%d]=%d\t", i, pid2SNPos[i]);
//	printf("\n");
	real *gain_h = new real[csrManager.curNumCsr];
	checkCudaErrors(cudaMemcpy(gain_h, pGain_d, sizeof(real) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));
	real maxGain = 0;
	uint maxKey = 0;
	for(int i = 0; i < csrManager.curNumCsr; i++){
		if(gain_h[i] > maxGain){
			maxGain = gain_h[i];
			maxKey = i;
		}
//		if(gain_h[i] >= 13472360447){
//			printf("################### gain=%f, i=%d\n", gain_h[i], i);
//		}
	}
	uint key = 1;//26012679;
	int segId = -1, segPos = -1;
	uint *feaStartPos = new uint[numSeg];
	uint *feaLen = new uint[numSeg];
	checkCudaErrors(cudaMemcpy(feaStartPos, csrManager.pEachCsrFeaStartPos, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(feaLen, csrManager.pEachCsrFeaLen, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));
	uint *csrKey_h = new uint[csrManager.curNumCsr];
	checkCudaErrors(cudaMemcpy(csrKey_h, csrManager.getCsrKey(), sizeof(uint) * csrManager.curNumCsr, cudaMemcpyDeviceToHost));

	uint *nodeStart = new uint[numofSNode];
	uint *nodeSize = new uint[numofSNode];
	checkCudaErrors(cudaMemcpy(nodeStart, csrManager.pEachCsrNodeStartPos, sizeof(uint) * numofSNode, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(nodeSize, csrManager.pEachNodeSizeInCsr, sizeof(uint) * numofSNode, cudaMemcpyDeviceToHost));
	double total_sum_hess = 0;
	int total_csrlen = 0;

//##################

	if(csrManager.curNumCsr >= 17538600){
		uint segId_h = csrKey_h[17538600];
		int segLen = feaLen[segId_h];
		uint segStartPos = feaStartPos[segId_h];
		uint lastFvaluePos = segStartPos + segLen - 1;
		for(int i = segStartPos; i <= segStartPos + 15; i++){
			//if(i % 200 == 0)
			printf("%f\t", hess_h[i]);
		}
		printf("\t segLen=%d\n", segLen);
	}
	for(int l = 0; l < numSeg; l++){
		printf("%d\t", feaLen[l]);
	}
	printf("\n");
//######################

//each fea hess
double total_hess_all_fea = 0;
for(int f = 0; f < bagManager.m_numFea; f++){
	double fea_total_hess = 0;
	for(int n = 0; n < numofSNode; n++){
		uint fea_start = feaStartPos[n * bagManager.m_numFea + f];
		uint fea_len = feaLen[n * bagManager.m_numFea + f];
		for(int s = fea_start; s < fea_start + fea_len; s++){
			fea_total_hess += eachCsrLen_h[s];
		}
	}
	total_hess_all_fea += fea_total_hess;
	printf("f=%d: hess=%f\t", f, fea_total_hess);
}
printf("total hess all fea=%f\n", total_hess_all_fea);
	for(int i = 0; i < numofSNode; i++){
		uint start = feaStartPos[i * bagManager.m_numFea];
		uint len = feaLen[i * bagManager.m_numFea];
		double hess = 0;
		for(int f = 0; f < bagManager.m_numFea; f++)
			total_csrlen += feaLen[i * bagManager.m_numFea + f];
		for(int s = start; s < start + len; s++){
			hess += hess_h[s];
		}
		total_sum_hess += hess;
		printf("hess%d=%f (%f)\t", i, hess, sum_hess_h[start + len - 1]);
//		if(start + len >= key){
//			real hess_left = 0, hess_right = 0;
//			for(int s = start; s < start + len; s++){
//				printf("key=%d\t", csrKey_h[s]);
//				if(s < key)
//					hess_left += hess_h[s];
//				else
//					hess_right += hess_h[s];
//			}
//			printf("hess_l=%f, hess_r=%f\n", hess_left, hess_right);
//			printf(".................................. start=%d, len=%d, offset=%d numSeg=%d, segId=%d gd=%f, hess=%f, sum_gd=%f, sum_hess=%f, gain=%f\n",
//					start, len, key - start, numSeg, i, gd_h[key], hess_h[key], sum_gd_h[key - 1], sum_hess_h[key - 1], gain_h[key]);
//			break;
//		}
	}
	printf("total=%f, total len=%d\n", total_sum_hess, total_csrlen);
	printf("max gain = %f, key=%d\n", maxGain, maxKey);
	if(csrManager.curNumCsr >= 17538600)
		printf("gain of key %d is %f\n", 17538600, gain_h[17538600]);

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
	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
										 csrManager.pEachCsrFeaStartPos,
										 csrManager.pEachCsrFeaLen,
										 csrManager.getCsrFvalue(),
										 pMaxGain_d, pMaxGainKey_d,
										 bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
					  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 pGD_d, pHess_d,
					  	  	  	  	  	 default2Right, csrManager.getCsrKey(),
//					  	  	  	  	  	 default2Right, pCSRKey,
					  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
					  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	checkCudaErrors(cudaFree(pMaxGain_d));
	checkCudaErrors(cudaFree(pMaxGainKey_d));
}

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
void DeviceSplitter::FeaFinderAllNode3(void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
	BagManager bagManager;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = bagManager.m_maxNumSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);
	//################
	int curNumofNode;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
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
		manager.MemsetAsync(fgd_d + bagId * bagManager.m_numFeaValue,
							0, sizeof(double) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(fhess_d + bagId * bagManager.m_numFeaValue,
							0, sizeof(real) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(fgain_d + bagId * bagManager.m_numFeaValue,
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
					fid = segId % bagManager.m_numFea;
					break;
				}
			}
			PROCESS_ERROR(fid != -1 && fid < bagManager.m_numFea);

			//decompressed
			for(int i = 0; i < csrLen; i++){
				int insId = preFvalueInsId[globalFvalueId];
				globalFvalueId++;
				PROCESS_ERROR(insId >= 0);
				int pid = pInsId2Nid[insId] - bagManager.m_pPreMaxNid_h[bagId] - 1;//mapping to new node
				if(pid < 0)
					continue;//############## this way okay?
				PROCESS_ERROR(pid >= 0 && pid < numofSNode);
				if(i == 0 || newCsrFvalue[pid * bagManager.m_numFea + fid].empty() || fabs(csrFvalue_merge[csrId] - newCsrFvalue[pid * bagManager.m_numFea + fid].back()) > DeviceSplitter::rt_eps){
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
			totalNewCsr += newCsrFvalue[i].size();
		printf("hello world org=%u v.s. csr=%u\n", bagManager.m_numFeaValue, totalNewCsr);
		thrust::exclusive_scan(thrust::host, eachNewCompressedFeaLen_merge, eachNewCompressedFeaLen_merge + numofSNode * bagManager.m_numFea, eachNewCompressedFeaStart_merge);
		delete[] pInsId2Nid;
		//###############################
		IndexComputer indexComp;
		indexComp.AllocMem(bagManager.m_numFea, numofSNode, bagManager.m_maxNumSplittable);
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
															   bagManager.m_numIns, manager.m_pDInsId, fvalue_org_d,
															   bagManager.m_pIndicesEachBag_d, numofDenseValue,
															   fgd_d + bagId * bagManager.m_numFeaValue,
															   fhess_d + bagId * bagManager.m_numFeaValue,
															   fvalue_d + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
		uint *pMaxNumFvalueOneNode = thrust::max_element(thrust::device, pTempNumFvalueEachNode, pTempNumFvalueEachNode + numofSNode);
		checkCudaErrors(cudaMemcpy(&maxNumFeaValueOneNode, pMaxNumFvalueOneNode, sizeof(int), cudaMemcpyDeviceToHost));
		indexComp.FreeMem();
		//###########
		LoadFvalueInsId<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(
						manager.m_pDInsId, preFvalueInsId, bagManager.m_pIndicesEachBag_d, bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		//##############
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   	   	manager.m_pDInsId, bagManager.m_numFeaValue,
															   		fgd_d + bagId * bagManager.m_numFeaValue,
															   	   	fhess_d + bagId * bagManager.m_numFeaValue);
		checkCudaErrors(cudaMemcpy(fvalue_d, fvalue_org_d, sizeof(real) * bagManager.m_numFeaValue, cudaMemcpyDefault));
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

	//###### compress
	CsrCompression(numofSNode, totalNumCsrFvalue_merge, eachCompressedFeaStartPos_merge, eachCompressedFeaLen_merge,
				   eachNodeSizeInCsr_merge, eachCsrNodeStartPos_merge, csrFvalue_merge, csrGD_h_merge, csrHess_h_merge, eachCsrLen_merge);
	printf("total csr fvalue=%u\n", totalNumCsrFvalue_merge);

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

//print dense csr len
/*	if(numofSNode > 4){
		for(int i = 927000; i < 927005; i++){
			printf("i=%d, len=%d\n", i, eachCsrLen_merge[i]);
		}
		printf("i=%d, len=%d\n", 7325427, eachCsrLen_merge[7325427]);
		//find segment id
		for(int s = 0; s < numSeg; s++){

			printf("");
		}
		printf("total csr=%d\n", totalNumCsrFvalue_merge);
		exit(0);
	}
*/
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

//##################

		if(totalNumCsrFvalue_merge >= 17538600){
			uint *csrKey_h = new uint[totalNumCsrFvalue_merge];
			checkCudaErrors(cudaMemcpy(csrKey_h, pnCsrKey_d, sizeof(uint) * totalNumCsrFvalue_merge, cudaMemcpyDeviceToHost));
			uint segId_h = csrKey_h[17538600];
			uint *feaLen = new uint[numSeg];
			uint *feaStartPos = new uint[numSeg];
			checkCudaErrors(cudaMemcpy(feaStartPos, pEachCompressedFeaStartPos_d, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(feaLen, pEachCompressedFeaLen_d, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));

			int segLen = feaLen[segId_h];
			uint segStartPos = feaStartPos[segId_h];
			uint lastFvaluePos = segStartPos + segLen - 1;

			//real *hess_h = new real[totalNumCsrFvalue_merge];
			//checkCudaErrors(cudaMemcpy(hess_h, pCsrHess_d, sizeof(real) * totalNumCsrFvalue_merge, cudaMemcpyDeviceToHost));
			real *hess_h = csrHess_h_merge;
			for(int i = segStartPos; i <= segStartPos + 15; i++){
				//if(i % 200 == 0)
					printf("%f\t", hess_h[i]);
			}
			printf("\t segLen=%d\n", segLen);
		}
		uint *feaLen = new uint[numSeg];
		checkCudaErrors(cudaMemcpy(feaLen, pEachCompressedFeaLen_d, sizeof(uint) * numSeg, cudaMemcpyDeviceToHost));
		for(int l = 0; l < numSeg; l++){
			printf("%d\t", feaLen[l]);
		}
		printf("\n");
//######################

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


