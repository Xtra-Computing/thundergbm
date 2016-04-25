/*
 * Splitter.cpp
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: split a node of a tree
 */

#include <algorithm>
#include <assert.h>
#include <math.h>
#include <map>
#include <iostream>

#include "Splitter.h"

using std::map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

/**
 * @brief: efficient best feature finder
 */
void Splitter::EfficientFeaFinder(SplitPoint &bestSplit, const nodeStat &parent, int nodeId)
{
	int nNumofFeature = m_vvFeaInxPair.size();
	for(int f = 0; f < nNumofFeature; f++)
	{
		double fBestSplitValue = -1;
		double fGain = 0.0;
		BestSplitValue(fBestSplitValue, fGain, f, parent, nodeId);

//		cout << "fid=" << f << "; gain=" << fGain << "; split=" << fBestSplitValue << endl;

		bestSplit.UpdateSplitPoint(fGain, fBestSplitValue, f);
	}
}

/**
 * @brief: mark as process for a node id
 */
void Splitter::MarkProcessed(int nodeId)
{
	//erase the split node or leaf node
	mapNodeIdToBufferPos.erase(nodeId);
	//mapNodeIdToBufferPos.clear(); can used this

	for(int i = 0; i < m_nodeIds.size(); i++)
	{
		if(m_nodeIds[i] == nodeId)
		{
//			assert(false);
			m_nodeIds[i] = -1;
		}
	}
}

/**
 * @brief: update the node statistics and buffer positions.
 */
void Splitter::UpdateNodeStat(vector<TreeNode*> &newSplittableNode, vector<nodeStat> &v_nodeStat)
{
	assert(mapNodeIdToBufferPos.empty() == true);
	assert(newSplittableNode.size() == v_nodeStat.size());
	m_nodeStat.clear();
	for(int i = 0; i < newSplittableNode.size(); i++)
	{
		int nid = newSplittableNode[i]->nodeId;
		assert(nid >= 0);
//		cout << "node " << nid << " maps to " << i << endl;
		mapNodeIdToBufferPos.insert(make_pair(nid, i));
		m_nodeStat.push_back(v_nodeStat[i]);
//		cout << "nodeStat[" << nid <<"].sum_hess=" << v_nodeStat[i].sum_hess << endl;
	}

	//compute node counts
/*	vector<int> counter;
	counter.resize(newSplittableNode.size());
	for(int i = 0; i < m_nodeIds.size(); i++)
	{
		int nid = m_nodeIds[i];
		map<int, int>::iterator it = mapNodeIdToBufferPos.find(nid);
		assert(it != mapNodeIdToBufferPos.end() || nid == -1);
		if(nid != -1)
			counter[it->second]++;
	}
	for(int j = 0; j < newSplittableNode.size(); j++)
	{
		cout << "bufferPos " << j << " numof ins=" << counter[j] << endl;
	}
	*/
}

/**
 * @brief: efficient best feature finder
 */
void Splitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	int nNumofFeature = m_vvFeaInxPair.size();
	for(int f = 0; f < nNumofFeature; f++)
	{
		vector<key_value> &featureKeyValues = m_vvFeaInxPair[f];

		int nNumofKeyValues = featureKeyValues.size();
		vector<nodeStat> tempStat;
		vector<double> vLastValue;

		int bufferSize = mapNodeIdToBufferPos.size();

		tempStat.clear();
		vLastValue.clear();
		tempStat.resize(bufferSize);
		vLastValue.resize(bufferSize);

	    for(int i = 0; i < nNumofKeyValues; i++)
	    {
	    	int insId = featureKeyValues[i].id;
			int nid = m_nodeIds[insId];
			assert(nid >= -1);
			if(nid == -1)
				continue;

			// start working
			double fvalue = featureKeyValues[i].featureValue;

			// get the statistics of nid node
			// test if first hit, this is fine, because we set 0 during init
			map<int, int>::iterator it = mapNodeIdToBufferPos.find(nid);
			assert(it != mapNodeIdToBufferPos.end());
			int bufferPos = it->second;
			if(abs(tempStat[bufferPos].sum_hess) < 0.0001)
			{
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
			else
			{
				// try to find a split
				double min_child_weight = 1.0;//follow xgboost
				if(fabs(fvalue - vLastValue[bufferPos]) > 0.000002 &&
				   tempStat[bufferPos].sum_hess >= min_child_weight)
				{
					nodeStat lTempStat;
					assert(m_nodeStat.size() > bufferPos);
					lTempStat.Subtract(m_nodeStat[bufferPos], tempStat[bufferPos]);
					if(lTempStat.sum_hess >= min_child_weight)
					{
						double loss_chg = CalGain(m_nodeStat[bufferPos], tempStat[bufferPos], lTempStat);
						bool bUpdated = vBest[bufferPos].UpdateSplitPoint(loss_chg, (fvalue + vLastValue[bufferPos]) * 0.5f, f);
						if(bUpdated == true)
						{
							lchildStat[bufferPos] = lTempStat;
							rchildStat[bufferPos] = tempStat[bufferPos];
						}
					}
				}
				//update the statistics
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
		}
	}
}


/**
 * @brief: compute the best split value for a feature
 */
void Splitter::BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent, int nodeId)
{
	vector<key_value> &featureKeyValues = m_vvFeaInxPair[nFeatureId];

	double last_fvalue;
	SplitPoint bestSplit;
	nodeStat r_child, l_child;
	bool bFirst = true;

	int nCounter = 0;

	int nNumofKeyValues = featureKeyValues.size();

    for(int i = 0; i < nNumofKeyValues; i++)
    {
    	int originalInsId = featureKeyValues[i].id;
		int nid = m_nodeIds[originalInsId];
		if(nid != nodeId)
			continue;

		nCounter++;

		// start working
		double fvalue = featureKeyValues[i].featureValue;

		// get the statistics of nid node
		// test if first hit, this is fine, because we set 0 during init
		if(bFirst == true)
		{
			bFirst = false;
			r_child.Add(m_vGDPair_fixedPos[originalInsId].grad, m_vGDPair_fixedPos[originalInsId].hess);
			last_fvalue = fvalue;
		}
		else
		{
			// try to find a split
			double min_child_weight = 1.0;//follow xgboost
			if(fabs(fvalue - last_fvalue) > 0.000002 &&
			   r_child.sum_hess >= min_child_weight)
			{
				l_child.Subtract(parent, r_child);
				if(l_child.sum_hess >= min_child_weight)
				{
					double loss_chg = CalGain(parent, r_child, l_child);
					bestSplit.UpdateSplitPoint(loss_chg, (fvalue + last_fvalue) * 0.5f, nFeatureId);
				}
			}
			//update the statistics
			r_child.Add(m_vGDPair_fixedPos[originalInsId].grad, m_vGDPair_fixedPos[originalInsId].hess);
			last_fvalue = fvalue;
		}
	}

    fBestSplitValue = bestSplit.m_fSplitValue;
    fGain = bestSplit.m_fGain;
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void Splitter::ComputeGDSparse(vector<double> &v_fPredValue, vector<double> &m_vTrueValue_fixedPos)
{
	nodeStat rootStat;
	int nTotal = m_vTrueValue_fixedPos.size();
	for(int i = 0; i < nTotal; i++)
	{
		m_vGDPair_fixedPos[i].grad = v_fPredValue[i] - m_vTrueValue_fixedPos[i];
		m_vGDPair_fixedPos[i].hess = 1;
		rootStat.sum_gd += m_vGDPair_fixedPos[i].grad;
		rootStat.sum_hess += m_vGDPair_fixedPos[i].hess;
	}
//	cout << rootStat.sum_gd << " v.s. " << rootStat.sum_hess << endl;
	m_nodeStat.clear();
	m_nodeStat.push_back(rootStat);
	mapNodeIdToBufferPos.insert(make_pair(0,0));//node0 in pos0 of buffer
}

/**
 * @brief: compute gain for a split
 */
double Splitter::CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child)
{
	assert(abs(parent.sum_gd - l_child.sum_gd - r_child.sum_gd) < 0.0001);
	assert(parent.sum_hess == l_child.sum_hess + r_child.sum_hess);

	//compute the gain
	double fGain = (l_child.sum_gd * l_child.sum_gd)/(l_child.sum_hess + m_labda) +
				   (r_child.sum_gd * r_child.sum_gd)/(r_child.sum_hess + m_labda) -
				   (parent.sum_gd * parent.sum_gd)/(parent.sum_hess + m_labda);

	//This is different from the documentation of xgboost on readthedocs.com (i.e. fGain = 0.5 * fGain - m_gamma)
	//This is also different from the xgboost source code (i.e. fGain = fGain), since xgboost first splits all nodes and
	//then prune nodes with gain less than m_gamma.
	fGain = fGain - m_gamma;

	return fGain;
}

/**
 * @brief: split all splittable nodes of the current level
 */
void Splitter::SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel)
{
	int preMaxNodeId = m_nNumofNode - 1;

	int nNumofSplittableNode = splittableNode.size();
	assert(nNumofSplittableNode > 0);
	assert(splittableNode.size() == vBest.size());
	assert(vBest.size() == rchildStat.size());
	assert(vBest.size() == lchildStat.size());

	//for each splittable node, assign lchild and rchild ids
	map<int, pair<int, int> > mapPidCid;//(parent id, (lchildId, rchildId)).
	vector<TreeNode*> newSplittableNode;
	vector<nodeStat> newNodeStat;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = splittableNode[n]->nodeId;
//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = mapNodeIdToBufferPos[nid];
		assert(bufferPos < vBest.size());
		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		if(vBest[bufferPos].m_fGain <= 0 || bLastLevel == true)
		{
			//compute weight of leaf nodes
			splittableNode[n]->predValue = ComputeWeightSparseData(bufferPos);
			tree.nodes[nid]->rightChildId = LEAFNODE;
		}
		else
		{
			int lchildId = m_nNumofNode;
			int rchildId = m_nNumofNode + 1;

			mapPidCid.insert(make_pair(nid, make_pair(lchildId, rchildId)));

			//push left and right child statistics into a vector
			assert(lchildStat[bufferPos].sum_hess > 0);
			assert(rchildStat[bufferPos].sum_hess > 0);
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
		assert(fid >= 0);
		vFid.push_back(fid);
	}
//	PrintVec(vFid);
	assert((vFid.size() == 0 && (nNumofSplittableNode == 1 || bLastLevel == true)) || vFid.size() == nNumofSplittableNode);
	sort(vFid.begin(), vFid.end());
	vFid.resize(std::unique(vFid.begin(), vFid.end()) - vFid.begin());
	assert(vFid.size() <= nNumofSplittableNode);
//	PrintVec(vFid);

	//for each used feature to make decision
	for(int u = 0; u < vFid.size(); u++)
	{
		int ufid = vFid[u];
		assert(ufid < m_vvFeaInxPair.size() && ufid >= 0);

		//for each instance that has value on the feature
		vector<key_value> &featureKeyValues = m_vvFeaInxPair[ufid];
		int nNumofPair = featureKeyValues.size();
		for(int i = 0; i < nNumofPair; i++)
		{
			int insId = featureKeyValues[i].id;
			assert(insId < m_nodeIds.size());
			int nid = m_nodeIds[insId];

			int bufferPos = mapNodeIdToBufferPos[nid];

			int fid = vBest[bufferPos].m_nFeatureId;
			if(fid != ufid)//this feature is not the splitting feature for the instance.
				continue;

			double fvalue = featureKeyValues[i].featureValue;
			assert(nid >= 0);
			map<int, pair<int, int> >::iterator it = mapPidCid.find(nid);

			if(it == mapPidCid.end())//node doesn't need to split (leaf node or new node)
			{
				if(tree.nodes[nid]->rightChildId != LEAFNODE)
				{
					assert(nid > preMaxNodeId);
					continue;
				}
				assert(tree.nodes[nid]->rightChildId == LEAFNODE);
				continue;
			}

			if(it != mapPidCid.end())
			{//internal node (needs to split)
				double fPivot = vBest[bufferPos].m_fSplitValue;

				assert(it->second.second == it->second.first + 1);//right child id > than left child id
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
			if(it != mapPidCid.end())//node id is in the set of splittable nodes
				m_nodeIds[i] = it->second.first;//by default the instance with unknown feature value going to left child
			else
				cout << nid << endl;
			assert(it != mapPidCid.end());
		}
	}

	mapNodeIdToBufferPos.clear();

	UpdateNodeStat(newSplittableNode, newNodeStat);

	splittableNode.clear();
	splittableNode = newSplittableNode;
}

/**
 * @brief: split a node
 */
void Splitter::SplitNodeSparseData(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree, int &m_nNumofNode)
{
	TreeNode *leftChild = new TreeNode[1];
	TreeNode *rightChild = new TreeNode[1];

	leftChild->nodeId = m_nNumofNode;
	leftChild->parentId = node->nodeId;
	rightChild->nodeId = m_nNumofNode + 1;
	rightChild->parentId = node->nodeId;

	newSplittableNode.push_back(leftChild);
	newSplittableNode.push_back(rightChild);

	tree.nodes.push_back(leftChild);
	tree.nodes.push_back(rightChild);

	//node IDs. CAUTION: This part must be written here, because "union" is used for variables in nodes.
	node->leftChildId = leftChild->nodeId;
	node->rightChildId = rightChild->nodeId;
	node->featureId = sp.m_nFeatureId;
	node->fSplitValue = sp.m_fSplitValue;

	UpdateNodeIdForSparseData(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);

	m_nNumofNode += 2;

	leftChild->parentId = node->nodeId;
	rightChild->parentId = node->nodeId;
	leftChild->level = node->level + 1;
	rightChild->level = node->level + 1;
}



/**
 * @brief: update the node ids for the newly constructed nodes
 */
void Splitter::UpdateNodeIdForSparseData(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId)
{
	int nNumofIns = m_nodeIds.size();
	int fid = sp.m_nFeatureId;
	double fPivot = sp.m_fSplitValue;

	//create a mark
	vector<int> vMark;
	for(int i = 0; i < nNumofIns; i++)
		vMark.push_back(0);

	//for each instance that has value on the feature
	int nNumofPair = m_vvFeaInxPair[fid].size();
	for(int j = 0; j < nNumofPair; j++)
	{
		int insId = m_vvFeaInxPair[fid][j].id;
		double fvalue = m_vvFeaInxPair[fid][j].featureValue;
		if(m_nodeIds[insId] != parentNodeId)
		{
			vMark[insId] = -1;//this instance can be skipped.
			continue;
		}
		else
		{
			vMark[insId] = 1;//this instance has been considered.
			if(fvalue >= fPivot)
			{
				m_nodeIds[insId] = rightNodeId;
			}
			else
				m_nodeIds[insId] = leftNodeId;
		}
	}

	for(int i = 0; i < nNumofIns; i++)
	{
		if(vMark[i] != 0)
			continue;
		if(parentNodeId == m_nodeIds[i])
			m_nodeIds[i] = leftNodeId;
	}
}

/**
 * @brief: compute the weight of a leaf node
 */
double Splitter::ComputeWeightSparseData(int bufferPos)
{
	double predValue = -m_nodeStat[bufferPos].sum_gd / (m_nodeStat[bufferPos].sum_hess + m_labda);
	return predValue;
}

template<class T>
void Splitter::PrintVec(vector<T> &vec)
{
	int nNumofEle = vec.size();
	for(int i = 0; i < nNumofEle; i++)
	{
		cout << vec[i] << "\t";
	}
	cout << endl;
}

