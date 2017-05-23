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
#include <unordered_map>
#include <string>

#include "../../Host/Tree/RegTree.h"
#include "../TreeNode.h"
#include "../../Host/KeyValue.h"
#include "../../Host/UpdateOps/SplitPoint.h"
#include "../NodeStat.h"
#include "../../Host/GDPair.h"

using std::vector;
using std::unordered_map;
using std::string;

class BaseSplitter
{
public:
	static vector<vector<KeyValue> > m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id is instance id
	static unordered_map<int, int> mapNodeIdToBufferPos;
	static vector<int> m_nodeIds; //instance id to node id
	static vector<gdpair> m_vGDPair_fixedPos;
	static vector<nodeStat> m_nodeStat; //all the constructed tree nodes
	static real m_lambda;//the weight of the cost of complexity of a tree
	static real m_gamma;//the weight of the cost of the number of trees

public:
	virtual ~BaseSplitter(){}

	virtual string SpliterType() = 0;

	virtual void SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
				  	  	  const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel, void *pStream, int bagId) = 0;
	//a function for computing the gain of a feature
	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat, void *pStream, int bagId) = 0;
	//predict the value for each instance and compute their gradient
	virtual void ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > & vvInsSparse, void *stream, int bagId) = 0;

	double ComputeWeightSparseData(int bufferPos);

	void UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat);

public:
	//for sorting on each feature
	double CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child);

	const static int LEAFNODE = -2;

	static constexpr float rt_eps = 1e-5;
	static constexpr double min_child_weight = 1.0;//follow xgboost

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

public:
	int m_nRound;
	int m_nCurDept;
};




#endif /* BASESPLITTER_H_ */
