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

	FFMemManager ffManager;
	ffManager.resetMemForFindFea();
	//reset memory for this bag
	{
		checkCudaErrors(cudaMemset(bagManager.m_pGDEachFvalueEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMemset(bagManager.m_pHessEachFvalueEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMemset(bagManager.m_pDenseFValueEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));

		checkCudaErrors(cudaMemset(bagManager.m_pGDPrefixSumEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMemset(bagManager.m_pHessPrefixSumEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));
		checkCudaErrors(cudaMemset(bagManager.m_pGainEachFvalueEachBag + bagManager.m_numFeaValue * bagId, 0, sizeof(float_point) * bagManager.m_numFeaValue));
	}
	cudaDeviceSynchronize();

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
		manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pBuffIdVec_h, sizeof(int) * numofSNode);
		manager.MemcpyDeviceToHost(manager.m_pSNIdToBuffId, pSNIdToBuffId_h, sizeof(int) * maxNumofSplittable);
		manager.MemcpyDeviceToHost(manager.m_pInsIdToNodeId, indexComp.m_insIdToNodeId_dh, sizeof(int) * manager.m_numofIns);
		//compute indices
		indexComp.ComputeIndex(numofSNode, pSNIdToBuffId_h, maxNumofSplittable, pBuffIdVec_h);
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
		//copy index info to device memory
		manager.MemcpyHostToDevice(indexComp.m_pIndices_dh, ffManager.m_pIndices_d, sizeof(int) * ffManager.m_totalNumFeaValue);
		manager.MemcpyHostToDevice(indexComp.m_pNumFeaValueEachNode_dh, ffManager.m_pNumFeaValueEachNode_d, sizeof(long long) * maxNumofSplittable);
		manager.MemcpyHostToDevice(indexComp.m_pFeaValueStartPosEachNode_dh, ffManager.m_pFeaValueStartPosEachNode_d, sizeof(long long) * maxNumofSplittable);
		manager.MemcpyHostToDevice(indexComp.m_pEachFeaStartPosEachNode_dh, ffManager.m_pEachFeaStartPosEachNode_d, sizeof(long long) * maxNumofSplittable * nNumofFeature);
		manager.MemcpyHostToDevice(indexComp.m_pEachFeaLenEachNode_dh, ffManager.m_pEachFeaLenEachNode_d, sizeof(int) * maxNumofSplittable * nNumofFeature);

		clock_t start_gd = clock();
		LoadGDHessFvalue<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(manager.m_pGrad, manager.m_pHess, manager.m_numofIns,
															   manager.m_pDInsId, manager.m_pdDFeaValue, ffManager.m_pIndices_d, indexComp.m_totalFeaValue,
															   ffManager.pGDEachFeaValue, ffManager.pHessEachFeaValue, ffManager.pDenseFeaValue);
		cudaDeviceSynchronize();
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);
	}
	else
	{
		clock_t start_gd = clock();
		LoadGDHessFvalueRoot<<<dimNumofBlockToLoadGD, blockSizeLoadGD>>>(//manager.m_pGrad, manager.m_pHess, manager.m_numofIns,
															   bagManager.m_pInsGradEachBag + bagId * bagManager.m_numIns,
															   	   bagManager.m_pInsHessEachBag + bagId * bagManager.m_numIns, bagManager.m_numIns,
															   manager.m_pDInsId, manager.m_pdDFeaValue, indexComp.m_totalFeaValue,
															   //ffManager.pGDEachFeaValue, ffManager.pHessEachFeaValue, ffManager.pDenseFeaValue);
															   bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
															   	   bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
															   	   bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue);
		cudaDeviceSynchronize();
		clock_t end_gd = clock();
		total_fill_gd_t += (end_gd - start_gd);

		clock_t comIdx_start = clock();
		//manager.MemcpyHostToDevice(&manager.m_totalNumofValues, ffManager.m_pNumFeaValueEachNode_d, sizeof(long long));
		manager.MemcpyHostToDevice(&manager.m_totalNumofValues, bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable, sizeof(long long));
		//manager.MemcpyDeviceToDevice(manager.m_pFeaStartPos, ffManager.m_pFeaValueStartPosEachNode_d, sizeof(long long));//only one node
		manager.MemcpyDeviceToDevice(manager.m_pFeaStartPos, bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable, sizeof(long long));//only one node
		//manager.MemcpyDeviceToDevice(manager.m_pFeaStartPos, ffManager.m_pEachFeaStartPosEachNode_d, sizeof(long long) * nNumofFeature);
		manager.MemcpyDeviceToDevice(manager.m_pFeaStartPos,
										bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
										sizeof(long long) * nNumofFeature);
		//manager.MemcpyDeviceToDevice(manager.m_pDNumofKeyValue, ffManager.m_pEachFeaLenEachNode_d, sizeof(int) * nNumofFeature);
		manager.MemcpyDeviceToDevice(manager.m_pDNumofKeyValue,
									    bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									    sizeof(int) * nNumofFeature);

		//initialise indexComp
		//manager.MemcpyDeviceToHost(ffManager.m_pEachFeaLenEachNode_d, indexComp.m_pEachFeaLenEachNode_dh, sizeof(int) * nNumofFeature);
		manager.MemcpyDeviceToHost(bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									indexComp.m_pEachFeaLenEachNode_dh, sizeof(int) * nNumofFeature);
		indexComp.m_pFeaValueStartPosEachNode_dh[0] = 0;
		indexComp.m_pNumFeaValueEachNode_dh[0] = manager.m_totalNumofValues;
		clock_t comIdx_end = clock();
		total_com_idx_t += (comIdx_end - comIdx_start);
	}

	//load gd and hessian to a dense array in device memory
//	cout << "load gd" << endl;

	//manager.MemcpyDeviceToHost(ffManager.pGDEachFeaValue, ffManager.pGDPrefixSum, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(bagManager.m_pGDEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
									bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(bagManager.m_pHessEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
									bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue, sizeof(float_point) * manager.m_totalNumofValues);

#if testing
	float_point deltaTest = 0.01;
	float_point *pfGDEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	float_point *pfHessEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	manager.MemcpyDeviceToHost(ffManager.pGDEachFeaValue, pfGDEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(ffManager.pHessEachFeaValue, pfHessEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues);
#endif

//	cout << "prefix sum" << endl;
	clock_t start_scan = clock();
	//compute the feature with the maximum number of values
	int totalNumArray = indexComp.m_numFea * numofSNode;
	ComputeMaxNumValuePerFea(indexComp.m_pEachFeaLenEachNode_dh, totalNumArray);
	//compute prefix sum for gd and hess
	PrefixSumForEachNode(//totalNumArray, ffManager.pGDPrefixSum, ffManager.pHessPrefixSum,
						 totalNumArray, bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
						 	 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
						 //ffManager.m_pEachFeaStartPosEachNode_d, indexComp.m_pEachFeaLenEachNode_dh,
						 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
						 	 indexComp.m_pEachFeaLenEachNode_dh,
						 m_maxNumValuePerFea);//last parameter is a host variable
	cudaDeviceSynchronize();
	clock_t end_scan = clock();
	total_scan_t += (end_scan - start_scan);

#if testing
	float_point *pfGDScanEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	float_point *pfHessScanEachFeaValue_h = new float_point[manager.m_totalNumofValues];
	manager.MemcpyDeviceToHost(ffManager.pGDPrefixSum, pfGDScanEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(ffManager.pHessPrefixSum, pfHessScanEachFeaValue_h, sizeof(float_point) * manager.m_totalNumofValues);
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
	//compute gain
	int numofDenseValue = indexComp.m_pFeaValueStartPosEachNode_dh[numofSNode - 1] + indexComp.m_pNumFeaValueEachNode_dh[numofSNode - 1];
	int blockSizeComGain;
	dim3 dimNumofBlockToComGain;
	conf.ConfKernel(numofDenseValue, blockSizeComGain, dimNumofBlockToComGain);
	ComputeGainDense<<<dimNumofBlockToComGain, blockSizeComGain>>>(
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
	cudaDeviceSynchronize();

#if testing
	float_point *pGainDense = new float_point[manager.m_totalNumofValues];
	memset(pGainDense, 0, sizeof(float_point) * manager.m_totalNumofValues);
	manager.MemcpyDeviceToHost(ffManager.pGainEachFeaValue, pGainDense, sizeof(float_point) * manager.m_totalNumofValues);
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
	int blockSizeFirstGain;
	dim3 dimNumofBlockFirstGain;
	conf.ConfKernel(numFeaStartPos, blockSizeFirstGain, dimNumofBlockFirstGain);
	FirstFeaGain<<<dimNumofBlockFirstGain, blockSizeFirstGain>>>(//ffManager.m_pEachFeaStartPosEachNode_d, numFeaStartPos, ffManager.pGainEachFeaValue);
																bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
																	numFeaStartPos,
																	bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue);
	cudaDeviceSynchronize();
	clock_t end_comp_gain = clock();
	total_com_gain_t += (end_comp_gain - start_comp_gain);
#if testing
	manager.MemcpyDeviceToHost(ffManager.pGainEachFeaValue, pGainDense, sizeof(float_point) * manager.m_totalNumofValues);
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
	//find the block level best gain for each node
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
	PickLocalBestSplitEachNode<<<dimNumofBlockLocalBestGain, blockSizeLocalBestGain>>>(
								//ffManager.m_pNumFeaValueEachNode_d, ffManager.m_pFeaValueStartPosEachNode_d,
								bagManager.m_pNumFvalueEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
									bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable,
								//ffManager.pGainEachFeaValue, ffManager.pfLocalBestGain_d, ffManager.pnLocalBestGainKey_d);
								bagManager.m_pGainEachFvalueEachBag + bagId * bagManager.m_numFeaValue,
									bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode);
	cudaDeviceSynchronize();

	//find the global best gain for each node
	if(numBlockPerNode > 1)
	{
		int blockSizeBestGain;
		dim3 dimNumofBlockDummy;
		conf.ConfKernel(numBlockPerNode, blockSizeBestGain, dimNumofBlockDummy);
		PickGlobalBestSplitEachNode<<<numofSNode, blockSizeBestGain>>>(
									//ffManager.pfLocalBestGain_d, ffManager.pnLocalBestGainKey_d,
									bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
										bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_maxNumofBlockPerNode,
									//ffManager.pfGlobalBestGain_d, ffManager.pnGlobalBestGainKey_d,
									bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
								    numBlockPerNode, numofSNode);
		cudaDeviceSynchronize();
	}
	else
	{//local best fea is the global best fea
		//manager.MemcpyDeviceToDevice(ffManager.pfLocalBestGain_d, ffManager.pfGlobalBestGain_d, sizeof(float_point) * numofSNode);
		manager.MemcpyDeviceToDevice(bagManager.m_pfLocalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
										bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable, sizeof(float_point) * numofSNode);
		//manager.MemcpyDeviceToDevice(ffManager.pnLocalBestGainKey_d, ffManager.pnGlobalBestGainKey_d, sizeof(int) * numofSNode);
		manager.MemcpyDeviceToDevice(bagManager.m_pnLocalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
										bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable, sizeof(int) * numofSNode);
	}

	cudaDeviceSynchronize();
	clock_t end_search = clock();
	total_search_t += end_search - start_search;

//	cout << "construct split point" << endl;
	//construct split points; memset for split points
	//manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable, sizeof(SplitPoint) * maxNumofSplittable);
	FindSplitInfo<<<1, numofSNode>>>(//ffManager.m_pEachFeaStartPosEachNode_d, ffManager.m_pEachFeaLenEachNode_d,
									 bagManager.m_pEachFeaStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
									 	 bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea,
				  	  	  	  	  	 //ffManager.pDenseFeaValue, ffManager.pfGlobalBestGain_d, ffManager.pnGlobalBestGainKey_d,
									 bagManager.m_pDenseFValueEachBag + bagId * bagManager.m_numFeaValue,
									 	 bagManager.m_pfGlobalBestGainEachBag_d + bagId * bagManager.m_maxNumSplittable,
									 	 bagManager.m_pnGlobalBestGainKeyEachBag_d + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 //manager.m_pBuffIdVec, nNumofFeature,
				  	  	  	  	  	 bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable, nNumofFeature,
				  	  	  	  	  	 //manager.m_pSNodeStat, ffManager.pGDPrefixSum, ffManager.pHessPrefixSum,
				  	  	  	  	  	 bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 	 bagManager.m_pGDPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 	 bagManager.m_pHessPrefixSumEachBag + bagId * bagManager.m_numFeaValue,
				  	  	  	  	  	 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 	 bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
				  	  	  	  	  	 	 bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable);
				  	  	  	  	  	 //manager.m_pBestSplitPoint, manager.m_pRChildStat, manager.m_pLChildStat);
	cudaDeviceSynchronize();
//	cout << "Done find split" << endl;

#if testing
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint3, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pRChildStat, testpRChildStat3, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pLChildStat, testpLChildStat3, sizeof(nodeStat) * maxNumofSplittable);

#endif
	//end using dense array
}


