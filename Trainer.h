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
#include <fstream>
#include <vector>
#include <ctime>
#include <assert.h>
#include "RegTree.h"
#include "DatasetInfo.h"
#include "TreeNode.h"
#include "keyValue.h"
#include "NodeStat.h"
#include "SplitPoint.h"
#include "GDPair.h"
#include "Splitter.h"

using std::string;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;


class Trainer
{
public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;
	int m_nNumofSplittableNode;
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
	void SortFeaValue(int nNumofDim);
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea);
	void TrainGBDT(vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

	void ReleaseTree(vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);
	void GrowTree2(RegTree &tree);

private:
	void CreateNode();

	void ComputeGD(vector<double> &v_fPredValue);

	void ComputeWeight(TreeNode &node);
	double ComputeWeightSparseData(nodeStat & nStat);


	int Partition(const SplitPoint &sp, int startId, int endId);


//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);
	void CheckPartition(int startId, int endId, int middle, const SplitPoint &sp);

	template <class T> void Swap(T& x, T& y) { T t=x; x=y; y=t; }

	double total_find_fea_t;
	double total_split_t;
};



#endif /* TRAINER_H_ */
