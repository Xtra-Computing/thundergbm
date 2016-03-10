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
#include <algorithm>
#include <math.h>

#include "Trainer.h"
#include "Predictor.h"
#include "TreeNode.h"
#include "PrintTree.h"



using std::cout;
using std::endl;
using std::sort;
using std::ofstream;

/**
 * @brief: sort a vector in a descendant order
 */
bool CmpValue(const key_value &a, const key_value &b) {
  return a.featureValue > b.featureValue;
}

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

	//for debugging
	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInstance.size(); i++)
	{
		m_vPredBuffer_fixedPos.push_back(0);
		gdpair gd;
		m_vGDPair_fixedPos.push_back(gd);
	}

	int nNumofDim = m_vvInstance[0].size();
	SortFeaValue(nNumofDim);
}

/**
 * @brief:
 */
void Trainer::SortFeaValue(int nNumofDim)
{
	//sort the feature values for each feature
	vector<int> vCurParsePos;
	int nNumofIns = m_vvInsSparse.size();
	assert(m_vvInsSparse.size() == m_vvInstance.size());
	for(int i = 0; i < nNumofIns; i++)
	{
		vCurParsePos.push_back(0);
	}

	for(int j = 0; j < nNumofDim; j++)
	{
		vector<key_value> featurePair;
		for(int i = 0; i < nNumofIns; i++)
		{
			int curTop = vCurParsePos[i];
			if(m_vvInsSparse[i].size() == curTop)
				continue;

			int curFeaId = m_vvInsSparse[i][curTop].id;
			if(curFeaId == j)
			{
				key_value kv;
				kv.id = i;
				kv.featureValue = m_vvInsSparse[i][curTop].featureValue;
				featurePair.push_back(kv);
				vCurParsePos[i] = vCurParsePos[i] + 1;
			}
		}

		sort(featurePair.begin(), featurePair.end(), CmpValue);

		m_vvFeaInxPair.push_back(featurePair);
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

		vector<double> v_fPredValue_fixed;
		pred.Predict(m_vvInstance_fixedPos, vTree, v_fPredValue_fixed, m_vPredBuffer_fixedPos);
		ComputeGD2(v_fPredValue_fixed);

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

	//all instances are under node 0
	m_nodeIds.clear();
	for(int i = 0; i < m_vvInstance.size(); i++)
	{
		m_nodeIds.push_back(0);
	}
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void Trainer::ComputeGD(vector<double> &v_fPredValue)
{
	nodeStat rootStat;
	int nTotal = m_vTrueValue.size();
	for(int i = 0; i < nTotal; i++)
	{
		m_vGDPair[i].grad = v_fPredValue[i] - m_vTrueValue[i];
		m_vGDPair[i].hess = 1;
		rootStat.sum_gd += m_vGDPair[i].grad;
		rootStat.sum_hess += m_vGDPair[i].hess;
	}
//	cout << rootStat.sum_gd << " v.s. " << rootStat.sum_hess << endl;
	m_nodeStat.push_back(rootStat);
}

/**
 * @brief: compute the first order gradient and the second order gradient
 */
void Trainer::ComputeGD2(vector<double> &v_fPredValue)
{
	nodeStat rootStat;
	int nTotal = m_vTrueValue_fixedPos.size();
	for(int i = 0; i < nTotal; i++)
	{
		m_vGDPair_fixedPos[i].grad = v_fPredValue[i] - m_vTrueValue_fixedPos[i];
		m_vGDPair_fixedPos[i].hess = 1;
	}
//	cout << rootStat.sum_gd << " v.s. " << rootStat.sum_hess << endl;
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
	vector<nodeStat> newNodeStat;

	int nCurDepth = 0;
	while(m_nNumofSplittableNode > 0)
	{
		//for each splittable node
		for(int n = 0; n < m_nNumofSplittableNode; n++)
		{
			int nodeId = splittableNode[n]->nodeId;
			cout << "node=" << nodeId << "\t" << "numof ins=" << splittableNode[n]->endId - splittableNode[n]->startId + 1 << endl;;

			//find the best feature to split the node
			SplitPoint bestSplit;

			/**** two approaches to find the best feature ****/
			//efficient way to find the best split
			EfficientFeaFinder(bestSplit, m_nodeStat[n], nodeId);
			//naive way to find the best split
			//NaiveFeaFinder(bestSplit, splittableNode[n]->startId, splittableNode[n]->endId);

			//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
			if(bestSplit.m_fGain <= 0 || m_nMaxDepth == nCurDepth)
			{
				//compute weight of leaf nodes
				ComputeWeight(*splittableNode[n]);
			}
			else
			{
				//split the current node
//				cout << "start splitting..." << endl;
				SplitNode(splittableNode[n], newSplittableNode, bestSplit, tree, newNodeStat);
				assert(abs(m_nodeStat[n].sum_gd - newNodeStat[newNodeStat.size() - 1].sum_gd - newNodeStat[newNodeStat.size() - 2].sum_gd) < 0.0001);
//				cout << "end splitting." << endl;
			}
		}

		nCurDepth++;

		//assign new splittable nodes to the container
		splittableNode.clear();
		m_nodeStat.clear();
		splittableNode = newSplittableNode;
		m_nodeStat = newNodeStat;
		newSplittableNode.clear();
		newNodeStat.clear();
		m_nNumofSplittableNode = splittableNode.size();
	}
}

/**
 * @brief: efficient best feature finder
 */
void Trainer::EfficientFeaFinder(SplitPoint &bestSplit, const nodeStat &parent, int nodeId)
{
	for(int f = 0; f < data.nNumofFeature; f++)
	{
		double fBestSplitValue = -1;
		double fGain = 0.0;
		BestSplitValue(fBestSplitValue, fGain, f, parent, nodeId);

//		cout << "fid=" << f << "; gain=" << fGain << "; split=" << fBestSplitValue << endl;

		bestSplit.UpdateSplitPoint(fGain, fBestSplitValue, f);
	}
}

/**
 * @brief: naive best feature finder
 */
void Trainer::NaiveFeaFinder(SplitPoint &bestSplit, int startId, int endId)
{
	for (int f = 0; f < data.nNumofFeature; f++)
	{

		//find the best split point of each feature
		for (int i = startId; i <= endId; i++)
		{	//for each value in the node
			double fSplitValue = m_vvInstance[i][f];
			double fGain = ComputeGain(fSplitValue, f, startId, endId);

			//update the split point (only better gain has effect on the update)
			bestSplit.UpdateSplitPoint(fGain, fSplitValue, f);
		}
	}

	//find the value just smaller than the best split value
	int bestFeaId = bestSplit.m_nFeatureId;
	double bestFeaValue = bestSplit.m_fSplitValue;
	double fCurNextToBest = 0;
	for(int i = startId; i <= endId; i++)
	{	//for each value in the node
		double fSplitValue = m_vvInstance[i][bestFeaId];
		if(fSplitValue < bestFeaValue && fSplitValue > fCurNextToBest)
		{
			fCurNextToBest = fSplitValue;
		}
	}
	double dNewSplitValue = (bestSplit.m_fSplitValue + fCurNextToBest) * 0.5;
	bestSplit.UpdateSplitPoint(bestSplit.m_fGain, dNewSplitValue, bestSplit.m_nFeatureId);
}

/**
 * @brief: compute the best split value for a feature
 */
void Trainer::BestSplitValue(double &fBestSplitValue, double &fGain, int nFeatureId, const nodeStat &parent, int nodeId)
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
void Trainer::SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree, vector<nodeStat> &v_nodeStat)
{
	TreeNode *leftChild = new TreeNode[1];
	TreeNode *rightChild = new TreeNode[1];

	//startId and endId in the node will be changed later in this function
	int nNodeStart = node->startId;
	int nNodeEnd = node->endId;

	//re-organise gd vector
	int leftChildEndId = Partition(sp, nNodeStart, nNodeEnd);

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

	UpdateNodeId(sp, node->nodeId, m_nNumofNode, m_nNumofNode + 1);

	m_nNumofNode += 2;

	leftChild->parentId = node->nodeId;
	rightChild->parentId = node->nodeId;
	leftChild->level = node->level + 1;
	rightChild->level = node->level + 1;

	//get left and right node statistics
	nodeStat leftNodeStat;
	for(int i = nNodeStart; i <= leftChildEndId; i++)
		leftNodeStat.Add(m_vGDPair[i].grad, m_vGDPair[i].hess);
	nodeStat rightNodeStat;
	for(int i = leftChildEndId + 1; i <= nNodeEnd; i++)
		rightNodeStat.Add(m_vGDPair[i].grad, m_vGDPair[i].hess);

/*	cout << "after split l: " << leftNodeStat.sum_gd << " v.s. " << leftNodeStat.sum_hess << endl;
	cout << "after split r: " << rightNodeStat.sum_gd << " v.s. " << rightNodeStat.sum_hess << endl;
	exit(0);
*/

	v_nodeStat.push_back(leftNodeStat);
	v_nodeStat.push_back(rightNodeStat);
}

/**
 * @brief: update the node ids for the newly constructed nodes
 */
void Trainer::UpdateNodeId(const SplitPoint &sp, int parentNodeId, int leftNodeId, int rightNodeId)
{
	int nNumofIns = m_vvInstance_fixedPos.size();
	int fid = sp.m_nFeatureId;
	double fPivot = sp.m_fSplitValue;
	for(int i = 0; i < nNumofIns; i++)
	{
		if(m_nodeIds[i] != parentNodeId)
			continue;
		if(m_vvInstance_fixedPos[i][fid] >= fPivot)
		{
			m_nodeIds[i] = rightNodeId;
		}
		else
			m_nodeIds[i] = leftNodeId;
	}
}

/**
 * @brief: partition the data under the split node
 * @return: index of the last element of left child
 */
int Trainer::Partition(const SplitPoint &sp, int startId, int endId)
{
//	cout << "stardId=" << startId << "; endId=" << endId << endl;
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

void Trainer::CheckPartition(int startId, int endId, int middle, const SplitPoint &sp)
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

