/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include <math.h>
#include <time.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "Host/DataReader/LibsvmReaderSparse.h"
#include "Host/HostTrainer.h"
#include "Host/Evaluation/RMSE.h"
#include "Host/PureHostGBDTMain.h"
#include "DeviceHost/MyAssert.h"
#include "DeviceHost/svm-shared/fileOps.h"

#include "Device/Memory/gbdtGPUMemManager.h"
#include "Device/Memory/SNMemManager.h"
#include "Device/Memory/dtMemManager.h"
#include "Device/Memory/findFeaMemManager.h"
#include "Device/Bagging/BagManager.h"
#include "Device/GPUTask/GBDTTask.h"
#include "Device/DeviceTrainer.h"
#include "Device/DevicePredictor.h"
#include "Device/initCuda.h"
#include "Device/FindSplit/IndexComputer.h"

#include "Device/prefix-sum/prefixSum.h"
#include "Device/FileBuffer/FileBuffer.h"

int main(int argc, char *argv[])
{

//	TestPrefixSum(argc, argv);
//	return 1;

	string strFileName;
	int fileOption = 0;
	if(argc == 2)
	{
		string str = argv[1];
		fileOption = atoi(str.c_str());
	}
	switch(fileOption)
	{
	case 0:
		strFileName = "data/abalone.txt";//8 features and 4177 instances
		break;
	case 1:
		strFileName = "data/normalized_amz.txt";//19,998 features and 30,000 instances
		break;
	case 2:
		strFileName = "data/slice_loc.txt";//385 features and 53,499 instances
		break;
	case 3:
		strFileName = "data/cadata.txt";//8 features and 20640 instances
		break;
	case 4:
		strFileName = "data/YearPredictionMSD";//90 features and 463,715 instances
		break;
	case 5:
		strFileName = "data/mnist-large/mnist0.5m01.scale";
		break;
	case 6:
		strFileName = "data/epsilon_normalized_50k";
		break;
	case 7:
		strFileName = "data/SUSY";//Bug: two many instances for the prefix sum
		break;
	default:
		cerr << fileOption << " is an unknown file name option" << endl;
		return -1;
	}
	cout << "processing this file: " << strFileName << endl;

//	mainPureHost(strFileName);
//	return 1;

	CUcontext context;
	if(!InitCUDA('T', context))
	{
		cerr << "cannot initialise GPU" << endl;
		return 0;
	}

	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	int maxNumofUsedFeature = 1000;
	int maxNumofDenseIns = 1;//###### is later set to the number of instances
	int maxUsedFeaInTrees = 1000;
	int numBag = 1;//number of bags for bagging

	//for training
	int nNumofTree = 5;
	int nMaxDepth = 1;
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = 1;//minimum loss
	int maxNumofNodePerTree = pow(2, nMaxDepth + 1) - 1;
	int maxNumofSplittableNode = pow(2, nMaxDepth);

	DevicePredictor pred;

	DeviceSplitter splitter;
	splitter.InitDeviceSplitter(maxNumofNodePerTree);
	DeviceTrainer trainer(&splitter);

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	int numFea;
	int numIns;
	long long numFeaValue;

	string strFolder = strFileName + "-folder/";
	string bufferFileName = strFolder + "data-info.buffer";
	bool bBufferFileExist = CFileOps::IsFileExist(bufferFileName);
	bool bUsedBuffer = bBufferFileExist;

	vector<float_point> v_fLabel;
	vector<vector<KeyValue> > v_vInsSparse;
	clock_t start_init, end_init;
	//read the file the first time
	if(bBufferFileExist == false)
	{
		dataReader.GetDataInfo(strFileName, numFea, numIns, numFeaValue);
		dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel, strFileName, numFea, numIns);
		trainer.m_vvInsSparse = v_vInsSparse;//later used in sorting values for each feature
		trainer.m_vTrueValue = v_fLabel;
	}
	else
	{
		//read data set info
		FileBuffer::ReadDataInfo(strFolder, numFea, numIns, numFeaValue);
	}
	cout << "data has " << numFea << " features, " << numIns << " instances, and " << numFeaValue << " fea values" << endl;
	if(maxNumofUsedFeature < numFea)//maximum number of used features of a tree
		maxNumofUsedFeature = numFea;

	//fill the bags
	BagManager bagManager;
	bagManager.InitBagManager(numIns, numFea, nNumofTree, numBag, maxNumofSplittableNode,
							  maxNumofNodePerTree, numFeaValue, maxNumofUsedFeature);

	start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, numFea, bUsedBuffer);
	end_init = clock();

	//store feature key-value into array
	int *pInsId = new int[numFeaValue];
	float_point *pdValue = new float_point[numFeaValue];
	int *pNumofKeyValue = new int[numFea];
	long long *plFeaStartPos = new long long[numFea];//get start position of each feature

	//instances for prediction
	int *pFeaId = new int[numFeaValue];
	float_point *pfFeaValue = new float_point[numFeaValue];
	int *pNumofFea = new int[numIns];
	long long *plInsStartPos = new long long[numIns];

	float_point *pTrueLabel = new float_point[numIns];

	if(bBufferFileExist == false)
	{
		KeyValue::VecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue, plFeaStartPos);
//		KeyValue::TestVecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);
		//store sparse instances to GPU memory for prediction
		KeyValue::VecToArray(trainer.m_vvInsSparse, pFeaId, pfFeaValue, pNumofFea, plInsStartPos);
	//	KeyValue::TestVecToArray(trainer.m_vvInsSparse, pFeaId, pdFeaValue, pNumofFea);

		for(int i = 0; i < numIns; i++)
		{
			pTrueLabel[i] = v_fLabel[i];
		}

		//saved to buffer file
		FileBuffer::SetMembers(pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
							   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
							   pTrueLabel,
							   numFea, numIns, numFeaValue);
		FileBuffer::WriteBufferFile(strFolder);
	}
	else//read the arrays from buffer
	{
		cout << "read from buffer file: " << bufferFileName << endl;
		FileBuffer::ReadBufferFile(strFolder, pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
								   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
								   pTrueLabel,
								   numFea, numIns, numFeaValue);

#if false
		KeyValue::TestVecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);
		KeyValue::TestVecToArray(trainer.m_vvInsSparse, pFeaId, pfFeaValue, pNumofFea);
		for(int i = 0; i < numIns; i++)
		{
			PROCESS_ERROR(pTrueLabel[i] == v_fLabel[i]);
		}
#endif
	}

	//allocate memory for trees
	DTGPUMemManager treeMemManager;
	treeMemManager.allocMemForTrees(nNumofTree, maxNumofNodePerTree, nMaxDepth);

	//initialise gpu memory allocator
	GBDTGPUMemManager memAllocator;
	PROCESS_ERROR(numFeaValue > 0);
	memAllocator.m_totalNumofValues = numFeaValue;
	memAllocator.maxNumofDenseIns = numIns;
	memAllocator.m_maxUsedFeaInTrees = maxUsedFeaInTrees;

	//allocate memory for instances
	memAllocator.allocMemForIns(numFeaValue, numIns, numFea);
	memAllocator.allocMemForSplittableNode(maxNumofSplittableNode);//use in find features (i.e. best split points) process
	memAllocator.allocHostMemory();//allocate reusable host memory
	//allocate numofFeature*numofSplittabeNode

	SNGPUManager snManger;
	snManger.allocMemForTree(maxNumofNodePerTree);//reserve memory for the tree
	snManger.allocMemForParenChildIdMapping(maxNumofSplittableNode);
	snManger.allocMemForNewNode(maxNumofSplittableNode);
	snManger.allocMemForUsedFea(maxNumofUsedFeature);//use in splitting all nodes process

	FFMemManager ffManager;
	ffManager.m_totalNumFeaValue = numFeaValue;
	ffManager.getMaxNumofSN(numFeaValue, maxNumofSplittableNode);
	ffManager.allocMemForFindFea(numFeaValue, numIns, numFea, maxNumofSplittableNode);

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/
	Pruner::min_loss = fGamma;

	//initialise index computer object
	IndexComputer indexCom;
	indexCom.m_pInsId = pInsId;
	indexCom.m_totalFeaValue = numFeaValue;
	indexCom.m_pFeaStartPos = plFeaStartPos;
	indexCom.m_total_copy = 0;
	indexCom.AllocMem(numIns, numFea, maxNumofSplittableNode);

	//copy feature key-value to device memory
	memAllocator.MemcpyHostToDevice(pInsId, memAllocator.m_pDInsId, numFeaValue * sizeof(int));
	memAllocator.MemcpyHostToDevice(pdValue, memAllocator.m_pdDFeaValue, numFeaValue * sizeof(float_point));
	memAllocator.MemcpyHostToDevice(pNumofKeyValue, memAllocator.m_pDNumofKeyValue, numFea * sizeof(int));
	memAllocator.MemcpyHostToDevice(plFeaStartPos, memAllocator.m_pFeaStartPos, numFea * sizeof(long long));

	//copy instance key-value to device memory for prediction
	memAllocator.MemcpyHostToDevice(pFeaId, memAllocator.m_pDFeaId, numFeaValue * sizeof(int));
	memAllocator.MemcpyHostToDevice(pfFeaValue, memAllocator.m_pdDInsValue, numFeaValue * sizeof(float_point));
	memAllocator.MemcpyHostToDevice(pNumofFea, memAllocator.m_pDNumofFea, numIns * sizeof(int));
	memAllocator.MemcpyHostToDevice(plInsStartPos, memAllocator.m_pInsStartPos, numIns * sizeof(long long));

	//free host memory
	delete []pdValue;
	delete []pNumofKeyValue;
	delete []pFeaId;
	delete []pfFeaValue;
	delete []pNumofFea;
	delete []plInsStartPos;

	//copy true labels to gpu memory
	memAllocator.MemcpyHostToDevice(pTrueLabel, memAllocator.m_pdTrueTargetValue, numIns * sizeof(float_point));

	//training trees
	vector<RegTree> v_Tree;
	clock_t start_train_time = clock();

	//using multiple CPU threads
	GBDTTask gbdtTask;
	ThreadParam *thdInput = new ThreadParam[numBag];
	TaskParam *taskParam = new TaskParam[numBag];
	cudaStream_t *pStream = new cudaStream_t[numBag];
	int *pThreadStatus = new int[numBag];
	vector<pthread_t> vTid;
	for(int bag = 0; bag < numBag; bag++)
	{
		pThreadStatus[bag] = 1;//it is not used
		checkCudaErrors(cudaStreamCreate(&pStream[bag]));
		taskParam[bag].pCudaStream = &pStream[bag];
		taskParam[bag].thread_status = &pThreadStatus[bag];

		taskParam[bag].pDataPack = NULL;
		taskParam[bag].pCudaContext = &context;

		taskParam[bag].pResult = NULL;
		thdInput[bag].pObj = &gbdtTask;
		thdInput[bag].pThdParam = &taskParam[bag];

		pthread_attr_t attr;
		pthread_t tid = -1;//cpu thread id
		pthread_attr_init(&attr);
//		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);//PTHREAD_CREATE_DETACHED
		int nCreateTd = pthread_create(&tid, &attr, gbdtTask.Process, (void*) &thdInput[bag]);
		pthread_attr_destroy(&attr);
		cout << "thread id " << tid << endl;
		if(nCreateTd != 0)
			cerr << "creating thread failed" << endl;
		vTid.push_back(tid);
//		trainer.TrainGBDT(v_Tree);
	}
	//synchronise threads
	for(int i = 0; i < numBag; i++)
	{
		cout << "join thread id " << vTid[i] << endl;
		pthread_join(vTid[i], NULL);
	}
	clock_t end_train_time = clock();

	//save the trees to a file
	end_whole = clock();
	cout << "saved to file" << endl;
	trainer.SaveModel("tree.txt", v_Tree);

	double total_copy_indexCom = (double(indexCom.m_total_copy)) / CLOCKS_PER_SEC;
	cout << "total copy time = " << total_copy_indexCom << endl;
	double total_init = (double(end_init - start_init) / CLOCKS_PER_SEC);
	cout << "total init time = " << total_init << endl;
	double total_train = (double(end_train_time - start_train_time) / CLOCKS_PER_SEC);
	cout << "total training time (-extra idx comp) = " << total_train - total_copy_indexCom << endl;
	double total_all = (double(end_whole - begin_whole) / CLOCKS_PER_SEC);
	cout << "all sec = " << total_all << endl;

	//read testing instances from a file


	//run the GBDT prediction process
	clock_t begin_pre, end_pre;
	vector<float_point> v_fPredValue;

	begin_pre = clock();
	pred.PredictSparseIns(v_vInsSparse, v_Tree, v_fPredValue);
	end_pre = clock();
	double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
	cout << "prediction sec = " << prediction_time << endl;

	EvalRMSE rmse;
	float fRMSE = rmse.Eval(v_fPredValue, pTrueLabel, numIns);
	delete[] pTrueLabel;
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);
	memAllocator.releaseHostMemory();
	ffManager.freeMemForFindFea();

	ReleaseCuda(context);
	//free host memory
	delete []pInsId;
	delete []plFeaStartPos;
	delete []indexCom.m_pIndexCounterEachNode;
//	cudaFreeHost(indexCom.m_pIndices_dh);
//	cudaFreeHost(indexCom.m_insIdToNodeId_dh);

	return 0;
}


