/*
 * Splitter.cpp
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: split a node of a tree
 */

#include <math.h>
#include <algorithm>
#include <map>

#include "Splitter.h"

using std::map;
using std::make_pair;

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
 * @brief: efficient best feature finder
 */
void Splitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat, vector<double> &vLastValue)
{
	int nNumofFeature = m_vvFeaInxPair.size();
	for(int f = 0; f < nNumofFeature; f++)
	{
		vector<key_value> &featureKeyValues = m_vvFeaInxPair[f];

		int nNumofKeyValues = featureKeyValues.size();
		tempStat.clear();
		lchildStat.clear();
		vLastValue.clear();

		int bufferSize = mapNodeIdToBufferPos.size();

		tempStat.resize(bufferSize);
		lchildStat.resize(bufferSize);
		vLastValue.resize(bufferSize);

	    for(int i = 0; i < nNumofKeyValues; i++)
	    {
	    	int insId = featureKeyValues[i].id;
			int nid = m_nodeIds[insId];
			if(nid == -1)
				continue;

			// start working
			double fvalue = featureKeyValues[i].featureValue;

			// get the statistics of nid node
			// test if first hit, this is fine, because we set 0 during init
			int bufferPos = mapNodeIdToBufferPos[nid];
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
					lchildStat[bufferPos].Subtract(m_nodeStat[bufferPos], tempStat[bufferPos]);
					if(lchildStat[bufferPos].sum_hess >= min_child_weight)
					{
						double loss_chg = CalGain(m_nodeStat[bufferPos], tempStat[bufferPos], lchildStat[bufferPos]);
						vBest[bufferPos].UpdateSplitPoint(loss_chg, (fvalue + vLastValue[bufferPos]) * 0.5f, f);
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
 * @brief: split a node
 */
void Splitter::SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp,
						 RegTree &tree, vector<nodeStat> &v_nodeStat, int &m_nNumofNode)
{
	TreeNode *leftChild = new TreeNode[1];
	TreeNode *rightChild = new TreeNode[1];

	//startId and endId in the node will be changed later in this function
	int nNodeStart = node->startId;
	int nNodeEnd = node->endId;

	//re-organise gd vector

	leftChild->startId = node->startId;


	rightChild->endId = node->endId;

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

//	UpdateNodeId(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);
	UpdateNodeIdForSparseData(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);

	nodeStat leftNodeStat;
	nodeStat rightNodeStat;

	ComputeNodeStat(m_nNumofNode, leftNodeStat);
	ComputeNodeStat(m_nNumofNode + 1, rightNodeStat);

	v_nodeStat.push_back(leftNodeStat);
	v_nodeStat.push_back(rightNodeStat);

	m_nNumofNode += 2;

	leftChild->parentId = node->nodeId;
	rightChild->parentId = node->nodeId;
	leftChild->level = node->level + 1;
	rightChild->level = node->level + 1;
}

/**
 * @brief: split a node
 */
void Splitter::SplitNodeSparseData(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp,
								   RegTree &tree, vector<nodeStat> &v_nodeStat, int &m_nNumofNode)
{
	TreeNode *leftChild = new TreeNode[1];
	TreeNode *rightChild = new TreeNode[1];

	//startId and endId in the node will be changed later in this function
	int nNodeStart = node->startId;
	int nNodeEnd = node->endId;

	//re-organise gd vector

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

//	UpdateNodeId(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);
	UpdateNodeIdForSparseData(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);

	nodeStat leftNodeStat;
	nodeStat rightNodeStat;

	ComputeNodeStat(m_nNumofNode, leftNodeStat);
	ComputeNodeStat(m_nNumofNode + 1, rightNodeStat);

	v_nodeStat.push_back(leftNodeStat);
	v_nodeStat.push_back(rightNodeStat);

//	m_nodeStat[leftChild->nodeId] = leftNodeStat;
//	m_nodeStat[rightChild->nodeId] = rightNodeStat;

	m_nNumofNode += 2;

	leftChild->parentId = node->nodeId;
	rightChild->parentId = node->nodeId;
	leftChild->level = node->level + 1;
	rightChild->level = node->level + 1;
}


/**
 * @brief: compute the node statistics
 */
void Splitter::ComputeNodeStat(int nId, nodeStat &nodeStat)
{
	int nNumofIns = m_nodeIds.size();
	for(int i = 0; i < nNumofIns; i++)
	{
		if(m_nodeIds[i] != nId)
			continue;
		nodeStat.Add(m_vGDPair_fixedPos[i].grad, m_vGDPair_fixedPos[i].hess);
	}
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
