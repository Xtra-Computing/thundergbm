/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include <math.h>
#include "DataReader/LibSVMDataReader.h"
#include "Trainer.h"
#include "Predictor.h"
#include "Evaluation/RMSE.h"
#include "MyAssert.h"

int main()
{
	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	vector<vector<double> > v_vInstance;
	string strFileName = "data/YearPredictionMSD";
	int nNumofFeatures;
	int nNumofExamples;

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples);
//	dataReader.ReadLibSVMDataFormat(v_vInstance, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	vector<double> v_fLabel;
	vector<vector<key_value> > v_vInsSparse;
	dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/
	vector<RegTree> v_Tree;
	Trainer trainer;

	trainer.m_vvInsSparse = v_vInsSparse;
//	trainer.m_vvInstance_fixedPos = v_vInstance;
	trainer.m_vTrueValue_fixedPos = v_fLabel;

	int nNumofTree = 2;
	int nMaxDepth = 20;
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = 1;//minimum loss

	clock_t start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, nNumofFeatures);
	clock_t end_init = clock();

	clock_t start_train_time = clock();
	trainer.TrainGBDT(v_Tree);
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
	Predictor pred;
	vector<double> v_fPredValue_fixed;

	begin_pre = clock();
	pred.PredictSparseIns(v_vInsSparse, v_Tree, v_fPredValue_fixed);
	end_pre = clock();
	double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
	cout << "prediction sec = " << prediction_time << endl;

	EvalRMSE rmse;
	float fRMSE = rmse.Eval(v_fPredValue_fixed, v_fLabel);
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);

	return 0;
}


