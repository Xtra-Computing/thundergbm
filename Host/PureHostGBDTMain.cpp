/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include <math.h>
#include "DataReader/LibsvmReaderSparse.h"
#include "HostTrainer.h"
#include "HostPredictor.h"
#include "Evaluation/RMSE.h"
#include "../DeviceHost/MyAssert.h"
#include "UpdateOps/Pruner.h"

#include "PureHostGBDTMain.h"

int mainPureHost(string strFileName)
{
	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	vector<vector<double> > v_vInstance;
	//string strFileName = "data/abalone.txt";
	HostSplitter splitter;
	HostTrainer trainer(&splitter);

	int nNumofFeatures;
	int nNumofExamples;
	long long nNumofValue;

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples, nNumofValue);
//	dataReader.ReadLibSVMDataFormat(v_vInstance, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	vector<float_point> v_fLabel;
	vector<vector<KeyValue> > v_vInsSparse;
	dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/
	vector<RegTree> v_Tree;

	trainer.m_vvInsSparse = v_vInsSparse;
//	trainer.m_vvInstance_fixedPos = v_vInstance;
	trainer.m_vTrueValue = v_fLabel;

	int nNumofTree = 2;
	int nMaxDepth = 5;
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = 1;//minimum loss
	Pruner::min_loss = fGamma;

	clock_t start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, nNumofFeatures, false);
	clock_t end_init = clock();

	clock_t start_train_time = clock();
	trainer.TrainGBDT(v_Tree, NULL, 0);
	clock_t end_train_time = clock();

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
	HostPredictor pred;
	vector<float_point> v_fPredValue;

	begin_pre = clock();
	pred.PredictSparseIns(v_vInsSparse, v_Tree, v_fPredValue);
	end_pre = clock();
	double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
	cout << "prediction sec = " << prediction_time << endl;

	EvalRMSE rmse;
	float_point fRMSE = rmse.Eval(v_fPredValue, &v_fLabel[0], v_fLabel.size());
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);

	return 0;
}


