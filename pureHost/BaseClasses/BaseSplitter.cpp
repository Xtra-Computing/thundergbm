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
 * @brief: split all splittable nodes of the current level
 */
void BaseSplitter::SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel)
{
	int preMaxNodeId = m_nNumofNode - 1;

	int nNumofSplittableNode = splittableNode.size();
	PROCESS_ERROR(nNumofSplittableNode > 0);
	PROCESS_ERROR(splittableNode.size() == vBest.size());
	PROCESS_ERROR(vBest.size() == rchildStat.size());
	PROCESS_ERROR(vBest.size() == lchildStat.size());

	//for each splittable node, assign lchild and rchild ids
	map<int, pair<int, int> > mapPidCid;//(parent id, (lchildId, rchildId)).
	vector<TreeNode*> newSplittableNode;
	vector<nodeStat> newNodeStat;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = splittableNode[n]->nodeId;
//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = mapNodeIdToBufferPos[nid];
		PROCESS_ERROR(bufferPos < vBest.size());
		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		tree.nodes[nid]->loss = vBest[bufferPos].m_fGain;
		tree.nodes[nid]->base_weight = ComputeWeightSparseData(bufferPos);

		if(vBest[bufferPos].m_fGain <= rt_eps || bLastLevel == true)
		{
			//weight of a leaf node
			tree.nodes[nid]->predValue = tree.nodes[nid]->base_weight;
			tree.nodes[nid]->rightChildId = LEAFNODE;
		}
		else
		{
			int lchildId = m_nNumofNode;
			int rchildId = m_nNumofNode + 1;

			mapPidCid.insert(make_pair(nid, make_pair(lchildId, rchildId)));

			//push left and right child statistics into a vector
			PROCESS_ERROR(lchildStat[bufferPos].sum_hess > 0);
			PROCESS_ERROR(rchildStat[bufferPos].sum_hess > 0);
			newNodeStat.push_back(lchildStat[bufferPos]);
			newNodeStat.push_back(rchildStat[bufferPos]);

			//split into two nodes
			TreeNode *leftChild = new TreeNode[1];
			TreeNode *rightChild = new TreeNode[1];

			leftChild->nodeId = lchildId;
			leftChild->parentId = nid;
			leftChild->level = tree.nodes[nid]->level + 1;
			rightChild->nodeId = rchildId;
			rightChild->parentId = nid;
			rightChild->level = tree.nodes[nid]->level + 1;

			newSplittableNode.push_back(leftChild);
			newSplittableNode.push_back(rightChild);

			tree.nodes.push_back(leftChild);
			tree.nodes.push_back(rightChild);

			tree.nodes[nid]->leftChildId = leftChild->nodeId;
			tree.nodes[nid]->rightChildId = rightChild->nodeId;
			PROCESS_ERROR(vBest[bufferPos].m_nFeatureId >= 0);
			tree.nodes[nid]->featureId = vBest[bufferPos].m_nFeatureId;
			tree.nodes[nid]->fSplitValue = vBest[bufferPos].m_fSplitValue;


			m_nNumofNode += 2;
		}
	}

	//get all the used feature indices
	vector<int> vFid;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int fid = splittableNode[n]->featureId;
		int nid = splittableNode[n]->nodeId;
		if(fid == -1 && tree.nodes[nid]->rightChildId == LEAFNODE)
		{//leaf node should satisfy two conditions at this step
			continue;
		}
		PROCESS_ERROR(fid >= 0);
		vFid.push_back(fid);
	}
//	PrintVec(vFid);
	if(vFid.size() == 0)
		PROCESS_ERROR(nNumofSplittableNode == 1 || bLastLevel == true);
	PROCESS_ERROR(vFid.size() <= nNumofSplittableNode);
	sort(vFid.begin(), vFid.end());
	vFid.resize(std::unique(vFid.begin(), vFid.end()) - vFid.begin());
	PROCESS_ERROR(vFid.size() <= nNumofSplittableNode);
//	PrintVec(vFid);

	//for each used feature to make decision
	for(int u = 0; u < vFid.size(); u++)
	{
		int ufid = vFid[u];
		PROCESS_ERROR(ufid < m_vvFeaInxPair.size() && ufid >= 0);

		//for each instance that has value on the feature
		vector<KeyValue> &featureKeyValues = m_vvFeaInxPair[ufid];
		int nNumofPair = featureKeyValues.size();
		for(int i = 0; i < nNumofPair; i++)
		{
			int insId = featureKeyValues[i].id;
			PROCESS_ERROR(insId < m_nodeIds.size());
			int nid = m_nodeIds[insId];

			if(nid < 0)//leaf node
				continue;

			PROCESS_ERROR(nid >= 0);
			int bufferPos = mapNodeIdToBufferPos[nid];
			int fid = vBest[bufferPos].m_nFeatureId;
			if(fid != ufid)//this feature is not the splitting feature for the instance.
				continue;

			map<int, pair<int, int> >::iterator it = mapPidCid.find(nid);

			if(it == mapPidCid.end())//node doesn't need to split (leaf node or new node)
			{
				if(tree.nodes[nid]->rightChildId != LEAFNODE)
				{
					PROCESS_ERROR(nid > preMaxNodeId);
					continue;
				}
				PROCESS_ERROR(tree.nodes[nid]->rightChildId == LEAFNODE);
				continue;
			}

			if(it != mapPidCid.end())
			{//internal node (needs to split)
				PROCESS_ERROR(it->second.second == it->second.first + 1);//right child id > than left child id

				double fPivot = vBest[bufferPos].m_fSplitValue;
				double fvalue = featureKeyValues[i].featureValue;
				if(fvalue >= fPivot)
				{
					m_nodeIds[insId] = it->second.second;//right child id
				}
				else
					m_nodeIds[insId] = it->second.first;//left child id
			}
		}
	}

	//for those instances of unknown feature values.
	for(int i = 0; i < m_nodeIds.size(); i++)
	{
		int nid = m_nodeIds[i];
		if(nid == -1 || nid > preMaxNodeId)//processed node (i.e. leaf node or new node)
			continue;
		//newly constructed leaf node
		if(tree.nodes[nid]->rightChildId == LEAFNODE)
		{
			m_nodeIds[i] = -1;
		}
		else
		{
			map<int, pair<int, int> >::iterator it = mapPidCid.find(nid);
			m_nodeIds[i] = it->second.first;//by default the instance with unknown feature value going to left child

			PROCESS_ERROR(it != mapPidCid.end());
		}
	}

	mapNodeIdToBufferPos.clear();

	UpdateNodeStat(newSplittableNode, newNodeStat);

	splittableNode.clear();
	splittableNode = newSplittableNode;
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


