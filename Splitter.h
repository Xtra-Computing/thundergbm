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
#include "SplitPoint.h"

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

	double ComputeWeightSparseData(int bufferPos);
	void ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos);

	//a function for computing the gain of a feature
	void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat);

	void UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat);

private:
	//for sorting on each feature
	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);

	const static int LEAFNODE = -2;

public:
	//for debugging
	template<class T>
	void PrintVec(vector<T> &vec)
	{
		int nNumofEle = vec.size();
		for(int i = 0; i < nNumofEle; i++)
		{
			cout << vec[i] << "\t";
		}
		cout << endl;
	}

	int m_nRound;
	int m_nCurDept;
};



#endif /* SPLITTER_H_ */
