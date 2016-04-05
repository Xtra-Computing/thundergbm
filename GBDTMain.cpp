/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include "DataReader/LibSVMDataReader.h"
#include "Trainer.h"
#include "Predictor.h"
#include "Evaluation/RMSE.h"

int main()
{
	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	vector<vector<double> > v_vInstance;
	vector<double> v_fLabel;
	string strFileName = "data/kdd98.txt";
	int nNumofFeatures;
	int nNumofExamples;

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples);
//	dataReader.ReadLibSVMDataFormat(v_vInstance, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	vector<double> v_fLabel_non;
	vector<vector<key_value> > v_vInsSparse;
	dataReader.ReadLibSVMFormatSparse(v_vInsSparse, v_fLabel_non, strFileName, nNumofFeatures, nNumofExamples);

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/
	vector<RegTree> v_Tree;
	Trainer trainer;
//	trainer.m_vvInstance = v_vInstance;
	trainer.m_vTrueValue = v_fLabel;

	trainer.m_vvInsSparse = v_vInsSparse;
//	trainer.m_vvInstance_fixedPos = v_vInstance;
	trainer.m_vTrueValue_fixedPos = v_fLabel_non;

	int nNumofTree = 8;
	int nMaxDepth = 4;
	float fLabda = 1;
	float fGamma = 1;
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, nNumofFeatures);
	trainer.TrainGBDT(v_Tree);
	end_whole = clock();
	cout << "saved to file" << endl;
	trainer.SaveModel("tree.txt", v_Tree);


	double total_all = (double(end_whole - begin_whole) / CLOCKS_PER_SEC);
	cout << "all sec = " << total_all << endl;

	//read testing instances from a file


	//run the GBDT prediction process
	Predictor pred;
	vector<double> v_fPredValue_fixed;
	vector<double> v_fPreValue_buffer;
	for(int i = 0; i < v_vInsSparse.size(); i++)
		v_fPreValue_buffer.push_back(0);
	pred.PredictSparseIns(v_vInsSparse, v_Tree, v_fPredValue_fixed);

	EvalRMSE rmse;
	float fRMSE = rmse.Eval(v_fPredValue_fixed, v_fLabel_non);
	cout << "rmse=" << fRMSE << endl;

	return 0;
}


