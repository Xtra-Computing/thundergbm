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
#include "RegTree.h"
#include "DatasetInfo.h"
#include "TreeNode.h"
#include "keyValue.h"

using std::string;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;

/**
 * @brief: a structure to store split points
 */
struct SplitPoint{
	double m_fGain;
	double m_fSplitValue;
	int m_nFeatureId;

	SplitPoint()
	{
		m_fGain = 0;
		m_fSplitValue = 0;
		m_nFeatureId = -1;
	}

	void UpdateSplitPoint(double fGain, double fSplitValue, int nFeatureId)
	{
		if(fGain > m_fGain || (fGain == m_fGain && nFeatureId == m_nFeatureId))//second condition is for updating to a new split value
		{
			m_fGain = fGain;
			m_fSplitValue = fSplitValue;
			m_nFeatureId = nFeatureId;
		}
	}
};

class nodeStat{
public:
	double sum_gd;
	double sum_hess;

	nodeStat()
	{
		sum_gd = 0.0;
		sum_hess = 0.0;
	}

	void Subtract(const nodeStat &parent, const nodeStat &r_child)
	{
		sum_gd = parent.sum_gd - r_child.sum_gd;
		sum_hess = parent.sum_hess - r_child.sum_hess;
	}
	void Add(double gd, double hess)
	{
		sum_gd += gd;
		sum_hess += hess;
	}
};

class Trainer
{
private:
	struct gdpair {
	  /*! \brief gradient statistics */
	  double grad;
	  /*! \brief second order gradient statistics */
	  double hess;
	  gdpair(void) {grad = 0; hess = 0;}
	  gdpair(double grad, double hess) : grad(grad), hess(hess) {}
	};

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
	vector<gdpair> m_vGDPair_fixedPos;

	DataInfo data;
	double m_labda;//the weight of the cost of complexity of a tree
	double m_gamma;//the weight of the cost of the number of trees

	/*** for more efficient on finding the best split value of a feature ***/
	vector<vector<key_value> > m_vvInsSparse;
	vector<vector<key_value> > m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id is instance id
	vector<int> m_nodeIds; //instance id to node id
	vector<nodeStat> m_nodeStat; //all the constructed tree nodes

private:
	int m_nNumofNode;


public:
	void SortFeaValue(int nNumofDim);
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea);
	void TrainGBDT(vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);

private:
	void ComputeGD(vector<double> &v_fPredValue);
	void ComputeGDSparse(vector<double> &v_fPredValue);
	void CreateNode();
	double ComputeGain(double fSplitValue, int featureId, int dataStartId, int dataEndId);
	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);
	void ComputeWeight(TreeNode &node);
	double ComputeWeightSparseData(nodeStat & nStat);

	void SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree, vector<nodeStat> &v_nodeStat);
	void SplitNodeSparseData(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree, vector<nodeStat> &v_nodeStat);

	int Partition(const SplitPoint &sp, int startId, int endId);

	void ComputeNodeStat(int nId, nodeStat &nodeStat);
	void UpdateNodeIdForSparseData(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);
	void UpdateNodeId(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);

	//two different functions for computing the gain of a feature
	void EfficientFeaFinder(SplitPoint &bestSplit, const nodeStat &parent, int nodeId);
	void NaiveFeaFinder(SplitPoint &bestSplit, int startId, int endId);

	//for sorting on each feature
	void BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent, int nodeId);

//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);
	void CheckPartition(int startId, int endId, int middle, const SplitPoint &sp);

	template <class T> void Swap(T& x, T& y) { T t=x; x=y; y=t; }
};



#endif /* TRAINER_H_ */
