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

#include "Device/Memory/gbdtGPUMemManager.h"
#include "Device/Memory/SNMemManager.h"
#include "Device/Memory/dtMemManager.h"
#include "Device/Memory/findFeaMemManager.h"
#include "Device/DeviceTrainer.h"
#include "Device/DevicePredictor.h"
#include "Device/initCuda.h"
#include "Device/FindSplit/IndexComputer.h"

#include "Device/prefix-sum/prefixSum.h"
#include "DeviceHost/svm-shared/fileOps.h"
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

	//for training
	int nNumofTree = 2;
	int nMaxDepth = 5;
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = 1;//minimum loss

	DevicePredictor pred;

	DeviceSplitter splitter;
	DeviceTrainer trainer(&splitter);

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	int nNumofFeatures;
	int nNumofExamples;
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
		dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples, numFeaValue);
		dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);
		trainer.m_vvInsSparse = v_vInsSparse;//later used in sorting values for each feature
		trainer.m_vTrueValue = v_fLabel;
	}
	else
	{
		//read data set info
		FileBuffer::ReadDataInfo(strFolder, nNumofFeatures, nNumofExamples, numFeaValue);
	}
	cout << "data has " << nNumofFeatures << " features, " << nNumofExamples << " instances and, " << numFeaValue << " fea values" << endl;

	start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, nNumofFeatures, bUsedBuffer);
	end_init = clock();

	//store feature key-value into array
	int *pInsId = new int[numFeaValue];
	float_point *pdValue = new float_point[numFeaValue];
	int *pNumofKeyValue = new int[nNumofFeatures];
	long long *plFeaStartPos = new long long[nNumofFeatures];//get start position of each feature

	//instances for prediction
	int *pFeaId = new int[numFeaValue];
	float_point *pfFeaValue = new float_point[numFeaValue];
	int *pNumofFea = new int[nNumofExamples];
	long long *plInsStartPos = new long long[nNumofExamples];

	float_point *pTrueLabel = new float_point[nNumofExamples];

	if(bBufferFileExist == false)
	{
		KeyValue::VecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue, plFeaStartPos);
//		KeyValue::TestVecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);
		//store sparse instances to GPU memory for prediction
		KeyValue::VecToArray(trainer.m_vvInsSparse, pFeaId, pfFeaValue, pNumofFea, plInsStartPos);
	//	KeyValue::TestVecToArray(trainer.m_vvInsSparse, pFeaId, pdFeaValue, pNumofFea);

		for(int i = 0; i < nNumofExamples; i++)
		{
			pTrueLabel[i] = v_fLabel[i];
		}

		//saved to buffer file
		FileBuffer::SetMembers(pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
							   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
							   pTrueLabel,
							   nNumofFeatures, nNumofExamples, numFeaValue);
		FileBuffer::WriteBufferFile(strFolder);
	}
	else//read the arrays from buffer
	{
		cout << "read from buffer file: " << bufferFileName << endl;
		FileBuffer::ReadBufferFile(strFolder, pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
								   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
								   pTrueLabel,
								   nNumofFeatures, nNumofExamples, numFeaValue);

#if false
		KeyValue::TestVecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);
		KeyValue::TestVecToArray(trainer.m_vvInsSparse, pFeaId, pfFeaValue, pNumofFea);
		for(int i = 0; i < nNumofExamples; i++)
		{
			PROCESS_ERROR(pTrueLabel[i] == v_fLabel[i]);
		}
#endif
	}

	//allocate memory for trees
	DTGPUMemManager treeMemManager;
	int maxNumofNodePerTree = pow(2, nMaxDepth + 1) - 1;
	int maxNumofSplittableNode = pow(2, nMaxDepth);
	treeMemManager.allocMemForTrees(nNumofTree, maxNumofNodePerTree, nMaxDepth);

	//initialise gpu memory allocator
	GBDTGPUMemManager memAllocator;
	PROCESS_ERROR(numFeaValue > 0);
	memAllocator.m_totalNumofValues = numFeaValue;
	memAllocator.maxNumofDenseIns = nNumofExamples;
	memAllocator.m_maxUsedFeaInTrees = maxUsedFeaInTrees;

	//allocate memory for instances
	memAllocator.allocMemForIns(numFeaValue, nNumofExamples, nNumofFeatures);
	memAllocator.allocMemForSplittableNode(maxNumofSplittableNode);//use in find features (i.e. best split points) process
	memAllocator.allocHostMemory();//allocate reusable host memory
	//allocate numofFeature*numofSplittabeNode
	memAllocator.allocMemForSNForEachThread(nNumofFeatures, maxNumofSplittableNode);

	SNGPUManager snManger;
	snManger.allocMemForTree(maxNumofNodePerTree);//reserve memory for the tree
	snManger.allocMemForParenChildIdMapping(maxNumofSplittableNode);
	snManger.allocMemForNewNode(maxNumofSplittableNode);
	snManger.allocMemForUsedFea(maxNumofUsedFeature);//use in splitting all nodes process

	FFMemManager ffManager;
	ffManager.m_totalNumFeaValue = numFeaValue;
	ffManager.getMaxNumofSN(numFeaValue, maxNumofSplittableNode);
	ffManager.allocMemForFindFea(numFeaValue, nNumofExamples, nNumofFeatures, maxNumofSplittableNode);

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
	indexCom.AllocMem(nNumofExamples, nNumofFeatures, maxNumofSplittableNode);

	//copy feature key-value to device memory
	memAllocator.MemcpyHostToDevice(pInsId, memAllocator.m_pDInsId, numFeaValue * sizeof(int));
	memAllocator.MemcpyHostToDevice(pdValue, memAllocator.m_pdDFeaValue, numFeaValue * sizeof(float_point));
	memAllocator.MemcpyHostToDevice(pNumofKeyValue, memAllocator.m_pDNumofKeyValue, nNumofFeatures * sizeof(int));
	memAllocator.MemcpyHostToDevice(plFeaStartPos, memAllocator.m_pFeaStartPos, nNumofFeatures * sizeof(long long));

//	memAllocator.TestMemcpyDeviceToHost();
//	memAllocator.TestMemcpyDeviceToDevice();
//	memAllocator.TestMemcpyHostToDevice(pInsId, memAllocator.m_pDInsId, numFeaValue * sizeof(int));
//	memAllocator.TestMemcpyHostToDevice(pdValue, memAllocator.m_pdDFeaValue, numFeaValue * sizeof(float_point));
//	memAllocator.TestMemcpyHostToDevice(pNumofKeyValue, memAllocator.m_pDNumofKeyValue, nNumofFeatures * sizeof(int));

	//copy instance key-value to device memory for prediction
	memAllocator.MemcpyHostToDevice(pFeaId, memAllocator.m_pDFeaId, numFeaValue * sizeof(int));
	memAllocator.MemcpyHostToDevice(pfFeaValue, memAllocator.m_pdDInsValue, numFeaValue * sizeof(float_point));
	memAllocator.MemcpyHostToDevice(pNumofFea, memAllocator.m_pDNumofFea, nNumofExamples * sizeof(int));
	memAllocator.MemcpyHostToDevice(plInsStartPos, memAllocator.m_pInsStartPos, nNumofExamples * sizeof(long long));

	//free host memory
	delete []pdValue;
	delete []pNumofKeyValue;
	delete []pFeaId;
	delete []pfFeaValue;
	delete []pNumofFea;
	delete []plInsStartPos;

	//copy true labels to gpu memory
	memAllocator.MemcpyHostToDevice(pTrueLabel, memAllocator.m_pdTrueTargetValue, nNumofExamples * sizeof(float_point));

	//training trees
	vector<RegTree> v_Tree;
	clock_t start_train_time = clock();
	trainer.TrainGBDT(v_Tree);
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
	float fRMSE = rmse.Eval(v_fPredValue, pTrueLabel, nNumofExamples);
	delete[] pTrueLabel;
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);
	memAllocator.releaseHostMemory();
	memAllocator.freeMemForSNForEachThread();
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


