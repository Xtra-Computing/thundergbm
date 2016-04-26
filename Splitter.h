/*
 * Splitter.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: a class for splitting a node
 */

#ifndef SPLITTER_H_
#define SPLITTER_H_

#include <vector>
#include <unordered_map>

#include "RegTree.h"
#include "TreeNode.h"
#include "keyValue.h"
#include "SplitPoint.h"
#include "NodeStat.h"
#include "GDPair.h"

using std::vector;
using std::unordered_map;


class Splitter
{
public:
	vector<vector<key_value> > m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id is instance id
	unordered_map<int, int> mapNodeIdToBufferPos;
	vector<int> m_nodeIds; //instance id to node id
	vector<gdpair> m_vGDPair_fixedPos;
	vector<nodeStat> m_nodeStat; //all the constructed tree nodes
	double m_labda;//the weight of the cost of complexity of a tree
	double m_gamma;//the weight of the cost of the number of trees

public:
	void SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
				  const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel);
	void SplitNodeSparseData(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp,
							 RegTree &tree, int &m_nNumofNode);
	double ComputeWeightSparseData(int bufferPos);
	void ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos);

	//two different functions for computing the gain of a feature
	void EfficientFeaFinder(SplitPoint &bestSplit, const nodeStat &parent, int nodeId);
	void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat);

	//mark as processed
	void MarkProcessed(int nodeId);
	void UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat);

private:
	void UpdateNodeIdForSparseData(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);
	void UpdateNodeId(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId);

	void NaiveFeaFinder(SplitPoint &bestSplit, int startId, int endId);

	//for sorting on each feature
	void BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent, int nodeId);

	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);
	double ComputeGain(double fSplitValue, int featureId, int dataStartId, int dataEndId);

	const static int LEAFNODE = -2;

	//for debugging
	template<class T>
	void PrintVec(vector<T> &vec);
};



#endif /* SPLITTER_H_ */
