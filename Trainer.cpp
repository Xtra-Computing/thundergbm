/*
 * Trainer.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: GBDT trainer implementation
 */

#include <assert.h>

#include "Trainer.h"
#include "Predictor.h"
#include "TreeNode.h"

/*
 * @brief: initialise constants of a trainer
 */
void Trainer::InitTrainer(int nNumofTree, int nMaxDepth)
{
	m_nMaxNumofTree = nNumofTree;
	m_nMaxDepth = nMaxDepth;
	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInstance.size(); i++)
	{
		m_vPredBuffer[i] = 0;
	}
}

void Trainer::TrainGBDT(vector<vector<float> > &v_vInstance, vector<float> &v_fLabel, vector<RegTree> & vTree)
{
	assert(v_vInstance.size() > 0);
	assert(v_vInstance[0].size() > 0);
	data.nNumofInstance = v_vInstance.size();
	data.nNumofFeature = v_vInstance[0].size();

	Predictor pred;
	for(int i = 0; i < m_nMaxNumofTree; i++)
	{
		//initialise a tree
		RegTree tree;
		InitTree(tree);

		//predict the data by the existing trees
		vector<vector<float> > v_vInstance;
		vector<float> v_fPredValue;
		pred.Predict(v_vInstance, vTree, v_fPredValue, m_vPredBuffer);

		//compute gradient
		ComputeGD(v_fPredValue, m_vGDPair);

		//grow the tree
		GrowTree(tree);

		//save the tree
		vTree.push_back(tree);
	}
}

/**
 * @brief: initialise tree
 */
void Trainer::InitTree(RegTree &tree)
{
	TreeNode root;
	m_nNumofNode = 1;
	root.nodeId = 0;

	//initialise the range of instances that are covered by this node
	root.startId = 0;
	root.endId = m_vvInstance.size() - 1;

	tree.nodes.push_back(root);
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void Trainer::ComputeGD(vector<float> &v_fPredValue, vector<gdpair> &v_gdpair)
{
	int nTotal = m_vTrueValue.size();
	for(int i = 0; i < nTotal; i++)
	{
		v_gdpair[i].grad = m_vTrueValue[i] - v_fPredValue[i];
		v_gdpair[i].hess = 1;
	}
}

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void Trainer::GrowTree(RegTree &tree)
{
	m_nNumofSplittableNode = 0;
	//start splitting this tree from the root node
	vector<TreeNode*> splittableNode;
	for(int i = 0; i < int(tree.nodes.size()); i++)
	{
		splittableNode.push_back(&tree.nodes[i]);
		m_nNumofSplittableNode++;
	}

	vector<TreeNode*> newSplittableNode;

	int nCurDepth = 0;
	while(m_nNumofSplittableNode > 0)
	{
		//for each splittable node
		for(int n = 0; n < m_nNumofSplittableNode; n++)
		{
			//find the best feature to split the node
			SplitPoint bestSplit;
			for(int f = 0; f < data.nNumofFeature; f++)
			{
				//find the best split point of each feature

				for(int i = splittableNode[n]->startId; i <= splittableNode[n]->endId; i++)
				{//for each value in the node
					float fSplitValue = m_vvInstance[i][f];
					float fGain = ComputeGain(fSplitValue, f, splittableNode[n]->startId, splittableNode[n]->endId);

					//update the split point (only better gain has effect on the update)
					bestSplit.UpdateSplitPoint(fGain, fSplitValue, f);
				}
			}

			//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
			if(bestSplit.m_fGain < 0 || m_nMaxDepth == nCurDepth)
			{
				//compute weight of leaf nodes
				ComputeWeight(*splittableNode[n]);
			}
			else
			{
				//split the current node
				SplitNode(*splittableNode[n], newSplittableNode, bestSplit, tree);
			}
		}

		nCurDepth++;

		//assign new splittable nodes to the container
		splittableNode.clear();
		splittableNode = newSplittableNode;
		newSplittableNode.clear();
		m_nNumofSplittableNode = splittableNode.size();
	}
}

/**
 * @brief: compute the gain of a split
 */
float Trainer::ComputeGain(float fSplitValue, int featureId, int dataStartId, int dataEndId)
{
	//compute total gradient
	float firstGD_sum = 0;
	float secondGD_sum = 0;
	for(int i = dataStartId; i <= dataEndId; i++)
	{
		firstGD_sum += m_vGDPair[i].grad;
		secondGD_sum += m_vGDPair[i].hess;
	}

	//compute the gradient of children
	float firstGD_sum_l = 0;
	float firstGD_sum_r = 0;
	float secondGD_sum_l = 0;
	float secondGD_sum_r = 0;
	for(int i = dataStartId; i <= dataEndId; i++)
	{
		if(m_vvInstance[i][featureId] < fSplitValue)
		{//go to left
			firstGD_sum_l += m_vGDPair[i].grad;
			secondGD_sum_l += m_vGDPair[i].hess;
		}
		else
		{
			firstGD_sum_r += m_vGDPair[i].grad;
			secondGD_sum_r += m_vGDPair[i].hess;
		}
	}

	assert(firstGD_sum == firstGD_sum_l + firstGD_sum_r);
	assert(secondGD_sum == secondGD_sum_l + secondGD_sum_r);

	//compute the gain
	float fGain = (firstGD_sum_l * firstGD_sum_l)/(secondGD_sum_l + m_labda) +
				  (firstGD_sum_r * firstGD_sum_r)/(secondGD_sum_r + m_labda) -
				  (firstGD_sum * firstGD_sum)/(secondGD_sum + m_labda);
	fGain = fGain * 0.5 - m_gamma;

	return fGain;
}

/**
 * @brief: compute the weight of a leaf node
 */
void Trainer::ComputeWeight(TreeNode &node)
{
	int startId = node.startId, endId = node.endId;
	float sum_gd = 0, sum_hess = 0;

	for(int i = startId; i <= endId; i++)
	{
		sum_gd += m_vGDPair[i].grad;
		sum_hess += m_vGDPair[i].hess;
	}

	node.predValue = -sum_gd / (sum_hess + m_labda);
}

/**
 * @brief: split a node
 */
void Trainer::SplitNode(TreeNode &node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree)
{
	TreeNode *leftChild = new TreeNode, *rightChild = new TreeNode;

	//node IDs
	node.leftChildId = m_nNumofNode;
	node.rightChildId = m_nNumofNode + 1;
	m_nNumofNode += 2;

	leftChild->parentId = node.nodeId;
	rightChild->parentId = node.nodeId;

	//re-organise gd vector
	int leftChildEndId = Partition(sp, node.startId, node.endId);

	leftChild->endId = leftChildEndId;
	leftChild->startId = node.startId;

	rightChild->endId = node.endId;
	rightChild->startId = leftChildEndId + 1;

	newSplittableNode.push_back(leftChild);
	newSplittableNode.push_back(rightChild);

	tree.nodes.push_back(*leftChild);
	tree.nodes.push_back(*rightChild);
}

/**
 * @brief: partition the data under the split node
 * @return: index of the last element of left child
 */
int Trainer::Partition(SplitPoint &sp, int startId, int endId)
{
	int end = endId;
	float fPivot = sp.m_fSplitValue;
	int fId = sp.m_nFeatureId;
	for(int i = startId; i < end; i++)
	{
		while(m_vvInstance[end][fId] >= fPivot)
			end--;

		if(i >= end)
			break;

		if(m_vvInstance[i][fId] > fPivot)
		{
			vector<float> vSmall = m_vvInstance[end];
			gdpair small = m_vGDPair[end];

			m_vvInstance[end] = m_vvInstance[i];
			m_vGDPair[end] = m_vGDPair[i];

			m_vvInstance[i] = vSmall;
			m_vGDPair[i] = small;

			end--;
		}
	}

	return end;
}

