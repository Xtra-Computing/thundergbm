/*
 * BaseSplitter.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef BASESPLITTER_H_
#define BASESPLITTER_H_

#include <vector>
#include <map>

#include "../Tree/RegTree.h"
#include "../Tree/TreeNode.h"
#include "../KeyValue.h"
#include "SplitPoint.h"
#include "NodeStat.h"
#include "../GDPair.h"

using std::vector;
using std::map;


class BaseSplitter
{
public:
	vector<vector<KeyValue> > m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id is instance id
	map<int, int> mapNodeIdToBufferPos;
	vector<int> m_nodeIds; //instance id to node id
	vector<gdpair> m_vGDPair_fixedPos;
	vector<nodeStat> m_nodeStat; //all the constructed tree nodes
	double m_labda;//the weight of the cost of complexity of a tree
	double m_gamma;//the weight of the cost of the number of trees

public:
	void SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
				  const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel);

	double ComputeWeightSparseData(int bufferPos);
	void ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos);

	//a function for computing the gain of a feature
	void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat);

	void UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat);

public:
	//for sorting on each feature
	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);

	const static int LEAFNODE = -2;

	const static float rt_eps = 1e-5;
	const static double min_child_weight = 1.0;//follow xgboost

public:
	int m_nRound;
	int m_nCurDept;
};




#endif /* BASESPLITTER_H_ */
