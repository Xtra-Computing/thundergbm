/*
 * BaseTrainer.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: A general class of trainer
 */

#ifndef BASETRAINER_H_
#define BASETRAINER_H_

#include <iostream>
#include <vector>

#include "../Tree/RegTree.h"
#include "../KeyValue.h"
#include "../GDPair.h"
#include "BaseSplitter.h"
#include "../UpdateOps/Pruner.h"

using std::string;
using std::vector;

class BaseTrainer
{
public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;

	vector<vector<double> > m_vvInstance;
	vector<double> m_vTrueValue;

	/*** for more efficient on finding the best split value of a feature ***/
	vector<vector<KeyValue> > m_vvInsSparse;
	BaseSplitter *splitter;

private:
public:
	int m_nNumofNode;
	Pruner pruner;
	vector<double> m_vPredBuffer;

public:
	BaseTrainer(BaseSplitter *pSplitter){splitter = pSplitter;}
	virtual ~BaseTrainer(){}
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea);
	void TrainGBDT(vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

	void ReleaseTree(vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	virtual void GrowTree(RegTree &tree) = 0;

private:
//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);

public:
	double total_find_fea_t;
	double total_split_t;
	double total_prune_t;
};



#endif /* BASETRAINER_H_ */
