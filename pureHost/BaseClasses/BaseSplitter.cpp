/*
 * BaseSplitter.cpp
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */


#include <algorithm>
#include <math.h>
#include <map>
#include <iostream>

#include "../UpdateOps/SplitPoint.h"
#include "../MyAssert.h"
#include "BaseSplitter.h"

using std::map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

vector<vector<KeyValue> > BaseSplitter::m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id (or key) is instance id
map<int, int> BaseSplitter::mapNodeIdToBufferPos;
vector<int> BaseSplitter::m_nodeIds; //instance id to node id
vector<gdpair> BaseSplitter::m_vGDPair_fixedPos;
vector<nodeStat> BaseSplitter::m_nodeStat; //all the constructed tree nodes
double BaseSplitter::m_labda;//the weight of the cost of complexity of a tree
double BaseSplitter::m_gamma;//the weight of the cost of the number of trees

/**
 * @brief: update the node statistics and buffer positions.
 */
void BaseSplitter::UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat)
{
	PROCESS_ERROR(mapNodeIdToBufferPos.empty() == true);
	PROCESS_ERROR(newSplittableNode.size() == v_nodeStat.size());
	m_nodeStat.clear();
	for(int i = 0; i < newSplittableNode.size(); i++)
	{
		int nid = newSplittableNode[i]->nodeId;
		PROCESS_ERROR(nid >= 0);
		mapNodeIdToBufferPos.insert(make_pair(nid, i));
		m_nodeStat.push_back(v_nodeStat[i]);
	}
}


/**
 * @brief: compute the weight of a leaf node
 */
double BaseSplitter::ComputeWeightSparseData(int bufferPos)
{
	double nodeWeight = (-m_nodeStat[bufferPos].sum_gd / (m_nodeStat[bufferPos].sum_hess + m_labda));
//	printf("sum gd=%f, sum hess=%f, pv=%f\n", m_nodeStat[bufferPos].sum_gd, m_nodeStat[bufferPos].sum_hess, predValue);
	return nodeWeight;
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void BaseSplitter::ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos)
{
	nodeStat rootStat;
	int nTotal = m_vTrueValue_fixedPos.size();
	for(int i = 0; i < nTotal; i++)
	{
		m_vGDPair_fixedPos[i].grad = v_fPredValue[i] - m_vTrueValue_fixedPos[i];
		m_vGDPair_fixedPos[i].hess = 1;
		rootStat.sum_gd += m_vGDPair_fixedPos[i].grad;
		rootStat.sum_hess += m_vGDPair_fixedPos[i].hess;
//		if(i < 20)
//		{
//			cout.precision(6);
//			printf("pred and gd of %d is %f and %f\n", i, v_fPredValue[i], m_vGDPair_fixedPos[i].grad);
//		}
	}

	m_nodeStat.clear();
	m_nodeStat.push_back(rootStat);
	mapNodeIdToBufferPos.insert(make_pair(0,0));//node0 in pos0 of buffer
}

/**
 * @brief: compute gain for a split
 */
double BaseSplitter::CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child)
{
	PROCESS_ERROR(abs(parent.sum_gd - l_child.sum_gd - r_child.sum_gd) < 0.0001);
	PROCESS_ERROR(parent.sum_hess == l_child.sum_hess + r_child.sum_hess);

	//compute the gain
	double fGain = (l_child.sum_gd * l_child.sum_gd)/(l_child.sum_hess + m_labda) +
				   (r_child.sum_gd * r_child.sum_gd)/(r_child.sum_hess + m_labda) -
				   (parent.sum_gd * parent.sum_gd)/(parent.sum_hess + m_labda);

	//This is different from the documentation of xgboost on readthedocs.com (i.e. fGain = 0.5 * fGain - m_gamma)
	//This is also different from the xgboost source code (i.e. fGain = fGain), since xgboost first splits all nodes and
	//then prune nodes with gain less than m_gamma.
	//fGain = fGain - m_gamma;

	return fGain;
}


