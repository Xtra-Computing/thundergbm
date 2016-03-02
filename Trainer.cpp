/*
 * Trainer.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: GBDT trainer implementation
 */

#include <assert.h>
#include <iostream>
#include <ctime>
#include <stdlib.h>

#include "Trainer.h"
#include "Predictor.h"
#include "TreeNode.h"
#include "PrintTree.h"

using std::cout;
using std::endl;

/*
 * @brief: initialise constants of a trainer
 */
void Trainer::InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma)
{
	m_nMaxNumofTree = nNumofTree;
	m_nMaxDepth = nMaxDepth;
	m_labda = fLabda;
	m_gamma = fGamma;

	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInstance.size(); i++)
	{
		m_vPredBuffer.push_back(0);
		gdpair gd;
		m_vGDPair.push_back(gd);
	}
}

/**
 * @brief: training GBDTs
 */
void Trainer::TrainGBDT(vector<vector<double> > &v_vInstance, vector<double> &v_fLabel, vector<RegTree> & vTree)
{
	assert(v_vInstance.size() > 0);
	assert(v_vInstance[0].size() > 0);
	data.nNumofInstance = v_vInstance.size();
	data.nNumofFeature = v_vInstance[0].size();

	clock_t begin_pred, begin_gd, begin_grow;
	clock_t end_pred, end_gd, end_grow;
	double total_pred = 0, total_gd = 0, total_grow = 0;

	Predictor pred;
	for(int i = 0; i < m_nMaxNumofTree; i++)
	{
		cout << "start round " << i << endl;
		//initialise a tree
		RegTree tree;
		InitTree(tree);

		//predict the data by the existing trees
		vector<double> v_fPredValue;
		begin_pred = clock();
		pred.Predict(m_vvInstance, vTree, v_fPredValue, m_vPredBuffer);
		end_pred = clock();
		total_pred += (double(end_pred - begin_pred) / CLOCKS_PER_SEC);

//		PrintPrediction(v_fPredValue);

		//compute gradient
		begin_gd = clock();
		ComputeGD(v_fPredValue);
		end_gd = clock();
		total_gd += (double(end_gd - begin_gd) / CLOCKS_PER_SEC);

		//grow the tree
		begin_grow = clock();
		GrowTree(tree);
		end_grow = clock();
		total_grow += (double(end_grow - begin_grow) / CLOCKS_PER_SEC);

		//save the tree
		vTree.push_back(tree);
//		PrintTree(tree);
	}

	cout << "pred sec = " << total_pred << "; gd sec = " << total_gd << "; grow sec = " << total_grow << endl;

}

void Trainer::PrintTree(const RegTree &tree)
{
	int nNumofNode = tree.nodes.size();
	for(int i = 0; i < nNumofNode; i++)
	{
		cout << "node id " << tree.nodes[i]->nodeId << "\n";
	}
}

/**
 * @brief: save the trained model to a file
 */
void Trainer::SaveModel(string fileName, const vector<RegTree> &v_Tree)
{
	TreePrinter printer;
	printer.m_writeOut.open(fileName.c_str());

	int nNumofTree = v_Tree.size();
	for(int i = 0; i < nNumofTree; i++)
	{
		printer.m_writeOut << "booster[" << i << "]:\n";
		printer.PrintTree(v_Tree[i]);
	}

}

/**
 * @brief: initialise tree
 */
void Trainer::InitTree(RegTree &tree)
{
	TreeNode *root = new TreeNode;
	m_nNumofNode = 1;
	root->nodeId = 0;
	root->level = 0;

	//initialise the range of instances that are covered by this node
	root->startId = 0;
	root->endId = m_vvInstance.size() - 1;

	tree.nodes.push_back(root);
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void Trainer::ComputeGD(vector<double> &v_fPredValue)
{
	int nTotal = m_vTrueValue.size();
	for(int i = 0; i < nTotal; i++)
	{
		m_vGDPair[i].grad = v_fPredValue[i] - m_vTrueValue[i];
		m_vGDPair[i].hess = 1;
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
		splittableNode.push_back(tree.nodes[i]);
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

				int nodeId = -1;//#####

				double fBestSplitValue;
				double fGain;
				BestSplitValue(fBestSplitValue, fGain, f, m_nodeStat[nodeId]);

				bestSplit.UpdateSplitPoint(fGain, fBestSplitValue, f);

				//find the best split point of each feature
				for(int i = splittableNode[n]->startId; i <= splittableNode[n]->endId; i++)
				{//for each value in the node
					double fSplitValue = m_vvInstance[i][f];
					double fGain = ComputeGain(fSplitValue, f, splittableNode[n]->startId, splittableNode[n]->endId);

					//update the split point (only better gain has effect on the update)
					bestSplit.UpdateSplitPoint(fGain, fSplitValue, f);
				}
			}

			//find the value just smaller than the best split value
			int bestFeaId = bestSplit.m_nFeatureId;
			double bestFeaValue = bestSplit.m_fSplitValue;
			double fCurNextToBest = 0;
			for(int i = splittableNode[n]->startId; i <= splittableNode[n]->endId; i++)
			{//for each value in the node
				double fSplitValue = m_vvInstance[i][bestFeaId];
				if(fSplitValue < bestFeaValue && fSplitValue > fCurNextToBest)
				{
					fCurNextToBest = fSplitValue;
				}
			}
			double dNewSplitValue = (bestSplit.m_fSplitValue + fCurNextToBest) * 0.5;
			bestSplit.UpdateSplitPoint(bestSplit.m_fGain, dNewSplitValue, bestSplit.m_nFeatureId);

			//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
			if(bestSplit.m_fGain <= 0 || m_nMaxDepth == nCurDepth)
			{
				//compute weight of leaf nodes
				ComputeWeight(*splittableNode[n]);
			}
			else
			{
				//split the current node
				SplitNode(splittableNode[n], newSplittableNode, bestSplit, tree);
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
 * @brief: compute the best split value for a feature
 */
void Trainer::BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent)
{

	vector<double> &featureValues = m_vvTransIns[nFeatureId];
	vector<int> &InsIds = m_vvInsId[nFeatureId];

	double last_fvalue;
	SplitPoint bestSplit;
	nodeStat r_child, l_child;

	int nNumofDim = featureValues.size();
    for(int i = 0; i < nNumofDim; i++)
    {
    	int ridx = InsIds[i];
		int nid = m_nodePos[ridx];
		// start working
		double fvalue = featureValues[i];
		// get the statistics of nid node
		// test if first hit, this is fine, because we set 0 during init
		if(i == 0)
		{
			r_child.sum_gd = m_vGDPair[ridx].grad;
			r_child.sum_hess = m_vGDPair[ridx].hess;
			last_fvalue = fvalue;
		}
		else
		{
			// try to find a split
			double min_child_weight = 1;//follow xgboost
			if(abs(fvalue - last_fvalue) > 1e-5f * 2.0 &&
			   r_child.sum_hess >= min_child_weight)
			{
				l_child.Subtract(parent, r_child);
				if (l_child.sum_hess >= min_child_weight)
				{
					double loss_chg = CalGain(parent, r_child, l_child);
					bestSplit.UpdateSplitPoint(loss_chg, (fvalue + last_fvalue) * 0.5f, nFeatureId);
				}
			}
			// update the statistics
			r_child.sum_gd += m_vGDPair[ridx].grad;
			r_child.sum_hess += m_vGDPair[ridx].hess;
			last_fvalue = fvalue;
		}
	}

}

/**
 * @brief: compute the gain of a split
 */
double Trainer::ComputeGain(double fSplitValue, int featureId, int dataStartId, int dataEndId)
{
	//compute total gradient
	double firstGD_sum = 0;
	double secondGD_sum = 0;
	for(int i = dataStartId; i <= dataEndId; i++)
	{
		firstGD_sum += m_vGDPair[i].grad;
		secondGD_sum += m_vGDPair[i].hess;
	}

	//compute the gradient of children
	double firstGD_sum_l = 0.0;
	double firstGD_sum_r = 0.0;
	double secondGD_sum_l = 0;
	double secondGD_sum_r = 0;
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
	double fGain = (firstGD_sum_l * firstGD_sum_l)/(secondGD_sum_l + m_labda) +
				  (firstGD_sum_r * firstGD_sum_r)/(secondGD_sum_r + m_labda) -
				  (firstGD_sum * firstGD_sum)/(secondGD_sum + m_labda);

	//This is different from the documentation of xgboost on readthedocs.com (i.e. fGain = 0.5 * fGain - m_gamma)
	//This is also different from the xgboost source code (i.e. fGain = fGain), since xgboost first splits all nodes and
	//then prune nodes with gain less than m_gamma.
	fGain = fGain - m_gamma;

	return fGain;
}

/**
 * @brief: compute gain for a split
 */
double Trainer::CalGain(const nodeStat &parent, const nodeStat &r_child, const nodeStat &l_child)
{
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
 * @brief: compute the weight of a leaf node
 */
void Trainer::ComputeWeight(TreeNode &node)
{
	int startId = node.startId, endId = node.endId;
	double sum_gd = 0, sum_hess = 0;

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
void Trainer::SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree)
{
	TreeNode *leftChild = new TreeNode[1];
	TreeNode *rightChild = new TreeNode[1];


	//re-organise gd vector
	int leftChildEndId = Partition(sp, node->startId, node->endId);
	if(node->startId == 0 && node->endId == 9 && leftChildEndId == 7)
	{
		cout << "partition " << leftChildEndId << endl;
		for(int i = leftChildEndId; i <= 9; i++)
			cout << m_vGDPair[i].grad << endl;
	}

	leftChild->startId = node->startId;
	leftChild->endId = leftChildEndId;

	rightChild->startId = leftChildEndId + 1;
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

	m_nNumofNode += 2;

	leftChild->parentId = node->nodeId;
	rightChild->parentId = node->nodeId;
	leftChild->level = node->level + 1;
	rightChild->level = node->level + 1;

//	cout << "node " << node->nodeId << " split to " << leftChild->nodeId << " and " << rightChild->nodeId << endl;
}

/**
 * @brief: partition the data under the split node
 * @return: index of the last element of left child
 */
int Trainer::Partition(SplitPoint &sp, int startId, int endId)
{
	bool bPrint = false;
	int middle = endId;
	double fPivot = sp.m_fSplitValue;
	int fId = sp.m_nFeatureId;
	for(int i = startId; i <= middle; i++)
	{
		while(m_vvInstance[middle][fId] >= fPivot)
		{
			if(middle == 0)
			{
				break;
			}
			middle--;
		}

		if(i > middle)
		{
			bPrint = true;
//			cout << i << " v.s. " << startId << " : " << endId << "; " << sp.m_fSplitValue << " v.s. " << m_vvInstance[i][fId] << " : " << m_vvInstance[endId][fId] << endl;
			break;
		}


		if(m_vvInstance[i][fId] >= fPivot)
		{
			Swap(m_vvInstance[middle], m_vvInstance[i]);
			Swap(m_vGDPair[middle], m_vGDPair[i]);
			Swap(m_vTrueValue[middle], m_vTrueValue[i]);
			Swap(m_vPredBuffer[middle], m_vPredBuffer[i]);

			middle--;
		}
	}

	if(bPrint == true)
		CheckPartition(startId, endId, middle, sp);

	return middle;
}

void Trainer::CheckPartition(int startId, int endId, int middle, SplitPoint &sp)
{
	double fPivot = sp.m_fSplitValue;
	int fId = sp.m_nFeatureId;

	for(int i = startId; i <= endId; i++)
	{
		if(i <= middle && m_vvInstance[i][fId] >= fPivot)
		{
			cout << i << " v.s. " << middle << endl;
			cout << "split value is " << fPivot << endl;
			cout << "shit " << m_vvInstance[i][fId] << "\t";
			cout << endl;
		}
		if(i > middle && m_vvInstance[i][fId] < fPivot)
		{
			cout << i << " v.s. " << middle << endl;
			cout << "split value is " << fPivot << endl;
			cout << "oh shit " << m_vvInstance[i][fId] << "\t";
			cout << endl;
		}
	}
}

void Trainer::PrintPrediction(const vector<double> &vPred)
{
	int n = vPred.size();
	for(int i = 0; i < n; i++)
	{
		if(vPred[i] > 0)
			cout << vPred[i] << "\t";
	}
	cout << endl;
}

