/*
 * Splitter.cpp
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: split a node of a tree
 */

#include <algorithm>
#include <math.h>
#include <unordered_map>
#include <iostream>

#include "../Evaluation/RMSE.h"
#include "../HostPredictor.h"
#include "HostSplitter.h"
#include "SplitPoint.h"
#include "../../DeviceHost/MyAssert.h"

using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

/**
 * @brief: efficient best feature finder
 */
void HostSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat, void *pStream, int bagId)
{
	const float rt_2eps = 2.0 * rt_eps;
	double min_child_weight = 1.0;//follow xgboost

	int nNumofFeature = m_vvFeaInxPair.size();
	vector<nodeStat> tempStat;
	vector<double> vLastValue;
	vector<SplitPoint> vBest16;
	int bufferSize = mapNodeIdToBufferPos.size();

	for(int f = 0; f < nNumofFeature; f++)
	{
		vector<KeyValue> &featureKeyValues = m_vvFeaInxPair[f];
		if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
		{
			vBest16.clear();
			vBest16.resize(bufferSize);
		}

		int nNumofKeyValues = featureKeyValues.size();

		tempStat.clear();
		vLastValue.clear();
		tempStat.resize(bufferSize);
		vLastValue.resize(bufferSize);

	    for(int i = 0; i < nNumofKeyValues; i++)
	    {
	    	int insId = featureKeyValues[i].id;
			int nid = m_nodeIds[insId];
			PROCESS_ERROR(nid >= -1);
			if(nid == -1)
				continue;

			// start working
			double fvalue = featureKeyValues[i].featureValue;

			// get the statistics of nid node
			// test if first hit, this is fine, because we set 0 during init
			unordered_map<int, int>::iterator it = mapNodeIdToBufferPos.find(nid);
			PROCESS_ERROR(it != mapNodeIdToBufferPos.end());
			int bufferPos = it->second;
			if(tempStat[bufferPos].IsEmpty())
			{
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
			else
			{
				// try to find a split
				if(fabs(fvalue - vLastValue[bufferPos]) > rt_2eps &&
				   tempStat[bufferPos].sum_hess >= min_child_weight)
				{
					nodeStat lTempStat;
					PROCESS_ERROR(m_nodeStat.size() > bufferPos);
					lTempStat.Subtract(m_nodeStat[bufferPos], tempStat[bufferPos]);
					if(lTempStat.sum_hess >= min_child_weight)
					{
						double loss_chg = CalGain(m_nodeStat[bufferPos], tempStat[bufferPos], lTempStat);
						double sv = static_cast<float>((fvalue + vLastValue[bufferPos]) * 0.5f);
						bool bUpdated = vBest[bufferPos].UpdateSplitPoint(loss_chg, sv, f);
						if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
						{
							vBest16[bufferPos].UpdateSplitPoint(loss_chg, sv, f);
						}
						if(bUpdated == true)
						{
							lchildStat[bufferPos] = lTempStat;
							rchildStat[bufferPos] = tempStat[bufferPos];
							//if(f == 12 && nid == 262)
							//	printf("fid=%d; node id=%d; fvalue=%f; last_fvalue=%f; sv=%f\n", f, nid, fvalue, vLastValue[bufferPos], sv);
						}
					}
				}
				//update the statistics
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
		}

	    // finish updating all statistics, check if it is possible to include all sum statistics
	    for(unordered_map<int, int>::iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
	    {
	    	const int nid = it->first;
            nodeStat lTempStat;
	        lTempStat.Subtract(m_nodeStat[it->second], tempStat[it->second]);
	        if(lTempStat.sum_hess >= min_child_weight && tempStat[it->second].sum_hess >= min_child_weight)
	        {
//	        	cout << "good" << endl;
	        	double loss_chg = CalGain(m_nodeStat[it->second], tempStat[it->second], lTempStat);
	            const float gap = fabs(vLastValue[it->second]) + rt_eps;
	            const float delta = gap;
	            vBest[it->second].UpdateSplitPoint(loss_chg, vLastValue[it->second] + delta, f);
	        }
	    }


		if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
		{
			PrintVec(vBest16);
		}
	}
}

/**
 * @brief: split all splittable nodes of the current level
 */
void HostSplitter::SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel, void *pStream, int bagId)
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
		unordered_map<int, int>::iterator itBufferPos = mapNodeIdToBufferPos.find(nid);
//		cout << itBufferPos->first << " v.s. " << itBufferPos->second << endl;
		PROCESS_ERROR(itBufferPos != mapNodeIdToBufferPos.end() && bufferPos == itBufferPos->second);
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
			unordered_map<int, int>::iterator itBufferPos = mapNodeIdToBufferPos.find(nid);
			PROCESS_ERROR(itBufferPos != mapNodeIdToBufferPos.end() && bufferPos == itBufferPos->second);
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
 * @brief: predict the value for each instance and compute the gradient for each instance
 */
void HostSplitter::ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > & vvInsSparse, void *non, int bagIdZero)
{
	vector<real> v_fPredValue;

	HostPredictor pred;
	pred.PredictSparseInsByLastTree(vvInsSparse, vTree, v_fPredValue, m_vPredBuffer);

	if(vTree.size() > 0)
	{
		//run the GBDT prediction process
		EvalRMSE rmse;
		real fRMSE = rmse.Eval(v_fPredValue, &m_vTrueValue[0], m_vTrueValue.size());
		cout << "rmse=" << fRMSE << endl;
	}

	ComputeGDSparse(v_fPredValue, m_vTrueValue);
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void HostSplitter::ComputeGDSparse(vector<real> &v_fPredValue, vector<real> &m_vTrueValue_fixedPos)
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
