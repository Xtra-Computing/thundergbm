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

#include "../../Host/Tree/RegTree.h"
#include "../../Host/KeyValue.h"
#include "../../Host/GDPair.h"
#include "BaseSplitter.h"
#include "../../Host/UpdateOps/Pruner.h"

using std::string;
using std::vector;

class BaseTrainer
{
public:
	static int m_nMaxNumofTree;
	static int m_nMaxDepth;

//	vector<vector<double> > m_vvInstance;
	vector<float_point> m_vTrueValue;

	/*** for more efficient on finding the best split value of a feature ***/
	vector<vector<KeyValue> > m_vvInsSparse;
	BaseSplitter *splitter;

private:
public:
	int m_nNumofNode;
	Pruner pruner;
	vector<float_point> m_vPredBuffer;

public:
	BaseTrainer(BaseSplitter *pSplitter){splitter = pSplitter;}
	virtual ~BaseTrainer(){}
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea, bool usedBuffer);
	void TrainGBDT(vector<RegTree> &v_Tree, void *pStream, int bagId);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);


protected:
	virtual void InitTree(RegTree &tree) = 0;
	virtual void GrowTree(RegTree &tree, void *pStream, int bagId) = 0;
	virtual void ReleaseTree(vector<RegTree> &v_Tree) = 0;

private:
//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<float_point> &vPred);

public:
	double total_init_t;
	double total_find_fea_t;
	double total_split_t;
	double total_prune_t;
};



#endif /* BASETRAINER_H_ */
