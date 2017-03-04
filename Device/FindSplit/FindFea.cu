/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "../../DeviceHost/MyAssert.h"
#include "../../DeviceHost/svm-shared/HostUtility.h"
#include "../KernelConf.h"
#include "../Hashing.h"
#include "../Splitter/DeviceSplitter.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/findFeaMemManager.h"
#include "../Memory/SNMemManager.h"
#include "../Bagging/BagManager.h"
#include "FindFeaKernel.h"
#include "IndexComputer.h"

using std::cout;
using std::endl;
using std::make_pair;
using std::cerr;

#ifdef testing
#undef testing
#endif

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat, void *pStream, int bagId)
{
	GBDTGPUMemManager manager;
//	SNGPUManager snManager;
	BagManager bagManager;
//	int numofSNode = manager.m_curNumofSplitable;
	int numofSNode = bagManager.m_curNumofSplitableEachBag_h[bagId];
	int maxNumofSplittable = manager.m_maxNumofSplittable;
//	cout << bagManager.m_maxNumSplittable << endl;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//reset memory for this bag
	{
		manager.MemsetAsync(bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);

		manager.MemsetAsync(bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);
		manager.MemsetAsync(bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
							0, sizeof(float_point) * bagManager.m_numFeaValue, pStream);
	}
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	//process in a few rounds each of which has a subset of splittable nodes
#if testing
	int numRound = Ceil(numofSNode, ffManager.maxNumofSNodeInFF);
	if(numRound > 1)
		cout << "FindFea in " << numRound << " rounds." << endl;

	SplitPoint *testBestSplitPoint1 = new SplitPoint[maxNumofSplittable];
	nodeStat *testpRChildStat = new nodeStat[maxNumofSplittable];
	nodeStat *testpLChildStat = new nodeStat[maxNumofSplittable];
	SplitPoint *testBestSplitPoint3 = new SplitPoint[maxNumofSplittable];
	nodeStat *testpRChildStat3 = new nodeStat[maxNumofSplittable];
	nodeStat *testpLChildStat3 = new nodeStat[maxNumofSplittable];
#endif

	//compute index for each feature value
	IndexComputer indexComp;
	KernelConf conf;
	int blockSizeLoadGD;
	dim3 dimNumofBlockToLoadGD;
	conf.ConfKernel(indexComp.m_totalFeaValue, blockSizeLoadGD, dimNumofBlockToLoadGD);
	if(numofSNode > 1)
	{
		clock_t comIdx_start = clock();
		//prepare data for computing indices
		manager.MemcpyDeviceToHostAsync(bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable, pBuffIdVec_h, sizeof(int) * numofSNode, pStream);
		manager.MemcpyDeviceToHostAsync(bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable, pSNIdToBuffId_h, sizeof(int) * bagManager.m_maxNumSplittable, pStream);
		manager.MemcpyDeviceToHostAsync(bagManager.m_pInsIdToNodeIdEachBag, indexComp.m_insIdToNodeId_dh, sizeof(int) * bagManager.m_numIns, pStream);
		//compute indices
		indexComp.ComputeIndex(numofSNode, pSNIdToBuffId_h, maxNumofSplittable, pBuffIdVec_h);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);

		//copy scatter index to device memory
		manager.MemcpyHostToDeviceAsync(indexComp.m_pIndices_dh, bagManager.m_pIndicesEachBag_d + bagId * bagManager.m_numFeaValue, sizeof(int) * bagManager.m_numFeaValue, pStream);
		//copy # of feature values of each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pNumFeaValueEachNode_dh, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(long long) * bagManager.m_maxNumSplittable, pStream);
		//copy feature value start position of each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pFeaValueStartPosEachNode_dh, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable, 
										sizeof(long long) * maxNumofSplittable, pStream);
		//copy each feature start position in each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pEachFeaStartPosEachNode_dh, bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(long long) * maxNumofSplittable * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pEachFeaLenEachNode_dh, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(int) * bagManager.m_maxNumSplittable * bagManager.m_numFea, pStream);
		//copy (in pinned mem) of feature values for each feature in each node
		manager.MemcpyHostToDeviceAsync(indexComp.m_pEachFeaLenEachNode_dh, bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(int) * bagManager.m_maxNumSplittable * bagManager.m_numFea, pStream);

		PROCESS_ERROR(nNumofFeature == bagManager.m_numFea);
		clock_t start_gd = clock();
		//scatter operation
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, 
															   bagManager.m_numIns, manager.m_pDInsId, manager.m_pdDFeaValue, 
															   bagManager.m_pIndicesEachBag_d, indexComp.m_totalFeaValue,
															   bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue, 
															   bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue, 
															   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   	bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   		manager.m_pDInsId, manager.m_pdDFeaValue, indexComp.m_totalFeaValue,
															   		bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
															   	   	bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//copy # of feature values of a node
		manager.MemcpyHostToDeviceAsync(&manager.m_totalNumofValues, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(long long), pStream);
		//copy feature value start position of each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 	 sizeof(long long), pStream);
		//copy each feature start position in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pFeaStartPos, bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(long long) * nNumofFeature, pStream);
		//copy # of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									    sizeof(int) * nNumofFeature, pStream);

		//copy # (in pinned mem) of feature values of each feature in each node
		manager.MemcpyDeviceToDeviceAsync(manager.m_pDNumofKeyValue, bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(int) * nNumofFeature, pStream);

		//set indexComp
		indexComp.m_pFeaValueStartPosEachNode_dh[0] = 0;
		indexComp.m_pNumFeaValueEachNode_dh[0] = manager.m_totalNumofValues;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
	}

	//initialise values for gd and hess prefix sum computing
	manager.MemcpyDeviceToDeviceAsync(bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
									bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
									sizeof(float_point) * manager.m_totalNumofValues, pStream);
	manager.MemcpyDeviceToDeviceAsync(bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
									bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
									sizeof(float_point) * manager.m_totalNumofValues, pStream);


#if 0
	float_point deltaTest = 0.01;
	float_point *pfGDEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	float_point *pfHessEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	manager.MemcpyDeviceToHostAsync(bagManager.m_pGDEachFvalueEachBag, pfGDEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	manager.MemcpyDeviceToHostAsync(bagManager.m_pHessEachFvalueEachBag, pfHessEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	for(int i = 0; i < manager.m_totalNumofValues; i++){
		cout << pfHessEachFeaValue_h[i] << "\t";
		if(i % 20 == 0 && i > 0)
			cout << endl;
	}
#endif

//	cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = indexComp.m_numFea * numofSNode;
	cudaStreamSynchronize((*(cudaStream_t*)pStream));//wait until the pinned memory (m_pEachFeaLenEachNodeEachBag_dh) is filled
	ComputeMaxNumValuePerFea(bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea, totalNumArray, bagId);
	cout << "max # of values per fea is " << bagManager.m_pMaxNumValuePerFeaEachBag[bagId] <<"; # of arrays is " << totalNumArray << endl;

	cudaDeviceSynchronize();
	//compute prefix sum for gd and hess (handing more than one arrays)
	PrefixSumForEachNode(totalNumArray, bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
						 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
						 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
						 bagManager.m_pEachFeaLenEachNodeEachBag_dh + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
						 bagManager.m_pMaxNumValuePerFeaEachBag[bagId], pStream);//last parameter is a host variable

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

#if 0
	cout << "############### of sn " << numofSNode << endl;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pHessPrefixSumEachBag, pfHessEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	for(int i = 0; i < manager.m_totalNumofValues; i++){
		cout << pfHessEachFeaValue_h[i] << "\t";
		if(i % 20 == 0 && i > 0)
			cout << endl;
	}
#endif
#if 0 
	float_point *pfGDScanEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	float_point *pfHessScanEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	manager.MemcpyDeviceToHostAsync(bagManager.m_pGDPrefixSumEachBag, pfGDScanEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	manager.MemcpyDeviceToHostAsync(bagManager.m_pHessPrefixSumEachBag, pfHessScanEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	for(int n = 0; n < numofSNode; n++)
	{
		for(int fid = 0; fid < manager.m_numofFea; fid++)
		{
			float_point fGDScan = 0;
			float_point fHessScan = 0;
			int nodeStartPos = indexComp.m_pEachFeaStartPosEachNode_dh[n * manager.m_numofFea + fid];
			for(int fv = 0; fv < indexComp.m_pEachFeaLenEachNode_dh[n * manager.m_numofFea + fid]; fv++)
			{
				int pos = nodeStartPos + fv;
				fGDScan += pfGDEachFeaValue_h[pos];
				fHessScan += pfHessEachFeaValue_h[pos];
				if(abs(pfGDScanEachFeaValue_h[pos] - fGDScan) > deltaTest)
				{
					cout << "scan gd diff " << pfGDScanEachFeaValue_h[pos] << " v.s. " << fGDScan << endl;
				}
				if(abs(pfHessScanEachFeaValue_h[pos] - fHessScan) > deltaTest)
				{
					cout << "scan hess diff " << pfHessScanEachFeaValue_h[pos] << " v.s. " << fHessScan << endl;
				}
			}
		}
	}
	
#endif

//	cout << "compute gain" << endl;
	clock_t start_comp_gain = clock();
	//# of feature values that need to compute gains.
	int numofDenseValue = indexComp.m_pFeaValueStartPosEachNode_dh[numofSNode - 1] + indexComp.m_pNumFeaValueEachNode_dh[numofSNode - 1];
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain, 0, (*(cudaStream_t*)pStream)>>>(
											//manager.m_pSNodeStat, ffManager.m_pFeaValueStartPosEachNode_d, numofSNode,
											bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
											numofSNode,
											//manager.m_pBuffIdVec,
											bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable,
											//DeviceSplitter::m_lambda, ffManager.pGDPrefixSum, ffManager.pHessPrefixSum,
											DeviceSplitter::m_lambda, bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
											//ffManager.pDenseFeaValue, numofDenseValue, ffManager.pGainEachFeaValue);
											bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue, numofDenseValue,
											bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	if(cudaGetLastError() != cudaSuccess){
		cerr << "error after ComputeGainDense" << endl;
		exit(-1);
	}

#if 1
	float_point *pGainDense = new float_point[manager.m_totalNumofValues];
	memset(pGainDense, 0, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHostAsync(bagManager.m_pGainEachFvalueEachBag, pGainDense, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	float_point maxGain = -1;
	int max_id = -1;
	for(int i = 0; i < manager.m_totalNumofValues; i++)
	{//find the max gain
		if(pGainDense[i] > maxGain)
		{
			maxGain = pGainDense[i];
			max_id = i;
		}
	}
	cout << "max gain before fixing is " << maxGain << " id = " << max_id << endl;
#endif

//	cout << "first fea gain removal" << endl;
	//change the gain of the first feature value to 0
	int numFeaStartPos = indexComp.m_numFea * numofSNode;
	printf("num of feature start positions=%d\n", numFeaStartPos);
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numFeaStartPos, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain, 0, (*(cudaStream_t*)pStream)>>>(
																bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
																numFeaStartPos, bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	if(cudaGetLastError() != cudaSuccess){
		cerr << "error after FirstFeaGain" << endl;
		exit(-1);
	}
	clock_t end_comp_gain = clock();
	total_com_gain_t += (end_comp_gain - start_comp_gain);
#if 1
	manager.MemcpyDeviceToHostAsync(bagManager.m_pGainEachFvalueEachBag, pGainDense, sizeof(float_point) * manager.m_totalNumofValues, pStream);
	maxGain = -1;
	for(int i = 0; i < manager.m_totalNumofValues; i++)
	{//find the max gain
		if(pGainDense[i] > maxGain)
			maxGain = pGainDense[i];
	}
	cout << "max gain after fixing is " << maxGain << endl;
	delete []pGainDense;
#endif

//	cout << "searching" << endl;
	clock_t start_search = clock();
	//compute # of blocks for each node
	int maxNumFeaValueOneNode = -1;
	for(int n = 0; n < numofSNode; n++)
	{//find the node with the max number of element
		if(maxNumFeaValueOneNode < indexComp.m_pNumFeaValueEachNode_dh[n])
			maxNumFeaValueOneNode = indexComp.m_pNumFeaValueEachNode_dh[n];
	}
	PROCESS_ERROR(maxNumFeaValueOneNode > 0);
	int blockSizeLocalBestGain;
	dim3 dimNumofBlockLocalBestGain;
	conf.ConfKernel(maxNumFeaValueOneNode, blockSizeLocalBestGain, dimNumofBlockLocalBestGain);
	PROCESS_ERROR(dimNumofBlockLocalBestGain.z == 1);
	dimNumofBlockLocalBestGain.z = numofSNode;//each node per super block
	int numBlockPerNode = dimNumofBlockLocalBestGain.x * dimNumofBlockLocalBestGain.y;
	//find the block level best gain for each node
	PickLocalBestSplitEachNode<<<dimNumofBlockLocalBestGain, blockSizeLocalBestGain, 0, (*(cudaStream_t*)pStream)>>>(
								bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
								bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
								bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));

	if(cudaGetLastError() != cudaSuccess){
		cerr << "error after PickLocalBestSplitEachNode" << endl;
		exit(-1);
	}

	//find the global best gain for each node
	if(numBlockPerNode > 1){
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerNode, blockSizeBestGain, dimNumofBlockDummy);
		PickGlobalBestSplitEachNode<<<numofSNode, blockSizeBestGain, 0, (*(cudaStream_t*)pStream)>>>(
									bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
									bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
								    numBlockPerNode, numofSNode);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		if(cudaGetLastError() != cudaSuccess){
			cerr << "error after PickGlobalBestSplitEachNode" << endl;
			exit(-1);
		}
	}
	else{//local best fea is the global best fea
		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										sizeof(float_point) * numofSNode, pStream);
		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
											bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
											sizeof(int) * numofSNode, pStream);
	}

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t end_search = clock();
	total_search_t += end_search - start_search;

//	cout << "construct split point" << endl;
	//construct split points; memset for split points
	manager.MemcpyHostToDeviceAsync(manager.m_pBestPointHost, bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
									sizeof(SplitPoint) * bagManager.m_maxNumSplittable, pStream);
	FindSplitInfo<<<1, numofSNode, 0, (*(cudaStream_t*)pStream)>>>(
									 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
									 bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
				  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
//	cout << "Done find split" << endl;

#if testing
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint3, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pRChildStat, testpRChildStat3, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pLChildStat, testpLChildStat3, sizeof(nodeStat) * maxNumofSplittable);

#endif
}

