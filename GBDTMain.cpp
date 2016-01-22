/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include "DataReader/LibSVMDataReader.h"
#include "Trainer.h"

int main()
{
	/********* read training instances from a file **************/
	vector<vector<double> > v_vInstance;
	vector<double> v_fLabel;
	string strFileName = "data/abalone.txt";
	int nNumofFeatures;
	int nNumofExamples;

	LibSVMDataReader dataReader;
	dataReader.GetDataInfo(strFileName, nNumofFeatures, nNumofExamples);
	dataReader.ReadLibSVMDataFormat(v_vInstance, v_fLabel, strFileName, nNumofFeatures, nNumofExamples);

	/********* run the GBDT learning process ******************/
	vector<RegTree> v_Tree;
	Trainer trainer;
	trainer.m_vvInstance = v_vInstance;
	trainer.m_vTrueValue = v_fLabel;
	int nNumofTree = 2;
	int nMaxDepth = 2;
	float fLabda = 1;
	float fGamma = 1;
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma);
	trainer.TrainGBDT(v_vInstance, v_fLabel, v_Tree);
	trainer.SaveModel("tree.txt", v_Tree);

	//read testing instances from a file


	//run the GBDT prediction process


	return 0;
}


