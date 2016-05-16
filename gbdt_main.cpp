/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include <math.h>
#include <time.h>
#include "pureHost/DataReader/LibsvmReaderSparse.h"
#include "pureHost/HostTrainer.h"
#include "pureHost/Predictor.h"
#include "pureHost/Evaluation/RMSE.h"
#include "pureHost/MyAssert.h"

#include "gpu/Memory/gbdtGPUMemManager.h"
#include "gpu/DeviceTrainer.h"
#include "gpu/initCuda.h"
#include "pureHost/PureHostGBDTMain.h"

int main()
{
	//mainPureHost();
	//return 1;
	if(!InitCUDA('G'))
	{
		cerr << "cannot initialise GPU" << endl;
		return 0;
	}

	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	string strFileName = "data/abalone.txt";
	int maxNumofSplittableNode = 100;
	int maxNumofUsedFeature = 1000;
	int maxNumofNode = 10000;
	DeviceSplitter splitter;
	DeviceTrainer trainer(&splitter);

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	int nNumofFeatures;
	int nNumofExamples;
	long long nNumofValue;
	dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples, nNumofValue);

	vector<double> v_fLabel;
	vector<vector<KeyValue> > v_vInsSparse;
	dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	//initialise gpu memory allocator
	GBDTGPUMemManager memAllocator;
	PROCESS_ERROR(nNumofValue > 0);
	memAllocator.m_totalNumofValues = nNumofValue;
	//allocate memory for instances
	memAllocator.allocMemForIns(nNumofValue, nNumofExamples, nNumofFeatures);
	memAllocator.allocMemForSplittableNode(maxNumofSplittableNode);//use in find features (i.e. best split points) process
	memAllocator.allocMemForSplitting(maxNumofUsedFeature);//use in splitting all nodes process
	memAllocator.allocMemForTree(maxNumofNode);//reserve memory for the tree

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/

	trainer.m_vvInsSparse = v_vInsSparse;
	trainer.m_vTrueValue = v_fLabel;

	int nNumofTree = 2;
	int nMaxDepth = 5;
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = 1;//minimum loss
	Pruner::min_loss = fGamma;

	clock_t start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, nNumofFeatures);
	clock_t end_init = clock();

	//store feature key-value into array
	int *pInsId = new int[memAllocator.m_totalNumofValues];
	double *pdValue = new double[memAllocator.m_totalNumofValues];
	int *pNumofKeyValue = new int[nNumofFeatures];
	KeyValue::VecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);
	KeyValue::TestVecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue);

	//copy feature key-value to device memory
	memAllocator.MemcpyHostToDevice(pInsId, memAllocator.m_pDInsId, nNumofValue * sizeof(int));
	memAllocator.MemcpyHostToDevice(pdValue, memAllocator.m_pdDFeaValue, nNumofValue * sizeof(double));
	memAllocator.MemcpyHostToDevice(pNumofKeyValue, memAllocator.m_pDNumofKeyValue, nNumofFeatures * sizeof(int));

	memAllocator.TestMemcpyDeviceToHost();
	memAllocator.TestMemcpyHostToDevice(pInsId, memAllocator.m_pDInsId, nNumofValue * sizeof(int));
	memAllocator.TestMemcpyHostToDevice(pdValue, memAllocator.m_pdDFeaValue, nNumofValue * sizeof(double));
	memAllocator.TestMemcpyHostToDevice(pNumofKeyValue, memAllocator.m_pDNumofKeyValue, nNumofFeatures * sizeof(int));

	//free host memory
	delete []pInsId;
	delete []pdValue;
	delete []pNumofKeyValue;

	//training trees
	vector<RegTree> v_Tree;
	clock_t start_train_time = clock();
	trainer.TrainGBDT(v_Tree);
	clock_t end_train_time = clock();

	//save the trees to a file
	end_whole = clock();
	cout << "saved to file" << endl;
	trainer.SaveModel("tree.txt", v_Tree);

	double total_init = (double(end_init - start_init) / CLOCKS_PER_SEC);
	cout << "total init time = " << total_init << endl;
	double total_train = (double(end_train_time - start_train_time) / CLOCKS_PER_SEC);
	cout << "total training time = " << total_train << endl;
	double total_all = (double(end_whole - begin_whole) / CLOCKS_PER_SEC);
	cout << "all sec = " << total_all << endl;

	//read testing instances from a file


	//run the GBDT prediction process
	clock_t begin_pre, end_pre;
	Predictor pred;
	vector<double> v_fPredValue;

	begin_pre = clock();
	pred.PredictSparseIns(v_vInsSparse, v_Tree, v_fPredValue);
	end_pre = clock();
	double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
	cout << "prediction sec = " << prediction_time << endl;

	EvalRMSE rmse;
	float fRMSE = rmse.Eval(v_fPredValue, v_fLabel);
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);

	return 0;
}


