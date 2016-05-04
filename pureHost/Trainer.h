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

#include "Tree/RegTree.h"
#include "KeyValue.h"
#include "GDPair.h"
#include "UpdateOps/Splitter.h"
#include "UpdateOps/Pruner.h"

using std::string;
using std::vector;

class Trainer
{
public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;

	vector<vector<double> > m_vvInstance;
	vector<double> m_vTrueValue;

	/*** for more efficient on finding the best split value of a feature ***/
	vector<vector<KeyValue> > m_vvInsSparse;
	Splitter splitter;

private:
	int m_nNumofNode;
	Pruner pruner;
	vector<double> m_vPredBuffer;

public:
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea);
	void TrainGBDT(vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

	void ReleaseTree(vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);

private:
//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);

	double total_find_fea_t;
	double total_split_t;
	double total_prune_t;
};



#endif /* TRAINER_H_ */
