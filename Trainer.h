/*
 * Trainer.h
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *      @brief: GBDT trainer
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include <iostream>
#include <vector>

#include "RegTree.h"
#include "DatasetInfo.h"
#include "keyValue.h"
#include "SplitPoint.h"
#include "GDPair.h"
#include "Splitter.h"

using std::string;
using std::vector;



class Trainer
{
public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;
	int m_nNumofSplittableNode;//can be removed
	vector<vector<double> > m_vvInstance;
	vector<double> m_vTrueValue;
	vector<double> m_vPredBuffer;
	vector<gdpair> m_vGDPair;

	vector<vector<double> > m_vvInstance_fixedPos;
	vector<double> m_vTrueValue_fixedPos;
	vector<double> m_vPredBuffer_fixedPos;

	DataInfo data;

	/*** for more efficient on finding the best split value of a feature ***/
	vector<vector<key_value> > m_vvInsSparse;
	Splitter splitter;

private:
	int m_nNumofNode;

public:
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea);
	void TrainGBDT(vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

	void ReleaseTree(vector<RegTree> &v_Tree);

protected:
	void SortFeaValue(int nNumofDim);
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);

private:

	void ComputeGD(vector<double> &v_fPredValue);

//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);

	double total_find_fea_t;
	double total_split_t;
};



#endif /* TRAINER_H_ */
