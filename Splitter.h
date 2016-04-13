/*
 * Splitter.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: split a node
 */

#ifndef SPLITTER_H_
#define SPLITTER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <assert.h>
#include "RegTree.h"
#include "DatasetInfo.h"
#include "TreeNode.h"
#include "keyValue.h"
#include "SplitPoint.h"
#include "NodeStat.h"
#include "GDPair.h"

using std::string;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;


class Splitter
{
public:
	vector<vector<key_value> > m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id is instance id
	map<int, int> mapNodeIdToBufferPos;
	vector<int> m_nodeIds; //instance id to node id
	vector<gdpair> m_vGDPair_fixedPos;
	vector<nodeStat> m_nodeStat; //all the constructed tree nodes
	double m_labda;//the weight of the cost of complexity of a tree
	double m_gamma;//the weight of the cost of the number of trees

public:
	void SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp,
				   RegTree &tree, vector<nodeStat> &v_nodeStat, int &m_nNumofNode);
	void SplitNodeSparseData(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp,
							 RegTree &tree, vector<nodeStat> &v_nodeStat, int &m_nNumofNode);

	void ComputeNodeStat(int nId, nodeStat &nodeStat);
	void UpdateNodeIdForSparseData(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);
	void UpdateNodeId(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);

	void ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos);

	//two different functions for computing the gain of a feature
	void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat, vector<double> &vLastValue);
	void EfficientFeaFinder(SplitPoint &bestSplit, const nodeStat &parent, int nodeId);
	void NaiveFeaFinder(SplitPoint &bestSplit, int startId, int endId);

	//for sorting on each feature
	void BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent, int nodeId);

	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);
	double ComputeGain(double fSplitValue, int featureId, int dataStartId, int dataEndId);
};



#endif /* SPLITTER_H_ */
