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
using std::make_pair;
using std::cerr;

/**
 * @brief: sort a vector in a descendant order
 */
bool CmpValue(const key_value &a, const key_value &b)
{
  return a.featureValue > b.featureValue;
}

/*
 * @brief: initialise constants of a trainer
 */
void Trainer::InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea)
{
	m_nMaxNumofTree = nNumofTree;
	m_nMaxDepth = nMaxDepth;
	splitter.m_labda = fLabda;
	splitter.m_gamma = fGamma;

	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInsSparse.size(); i++)
	{
		m_vPredBuffer_fixedPos.push_back(0);
		gdpair gd;
		splitter.m_vGDPair_fixedPos.push_back(gd);
	}

	SortFeaValue(nNumofFea);
}

/**
 * @brief:
 */
void Trainer::SortFeaValue(int nNumofDim)
{
	//sort the feature values for each feature
	vector<int> vCurParsePos;
	int nNumofIns = m_vvInsSparse.size();
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

		splitter.m_vvFeaInxPair.push_back(featurePair);
	}
}

/**
 * @brief: training GBDTs
 */
void Trainer::TrainGBDT(vector<RegTree> & vTree)
{
	clock_t begin_pred, begin_gd, begin_grow;
	clock_t end_pred, end_gd, end_grow;
	double total_pred = 0, total_gd = 0, total_grow = 0;

	Predictor pred;
	for(int i = 0; i < m_nMaxNumofTree; i++)
	{
		cout << "start round " << i << endl;
		clock_t start_round = clock();
		//initialise a tree
		RegTree tree;
		InitTree(tree);

		//predict the data by the existing trees
/*		vector<double> v_fPredValue;
		pred.PredictDenseIns(m_vvInstance, vTree, v_fPredValue, m_vPredBuffer);
		//compute gradient
		ComputeGD(v_fPredValue);
*/
//		PrintPrediction(v_fPredValue);

		vector<double> v_fPredValue_fixed;
		begin_pred = clock();
		pred.PredictSparseIns(m_vvInsSparse, vTree, v_fPredValue_fixed, m_vPredBuffer_fixedPos);
		end_pred = clock();
		total_pred += (double(end_pred - begin_pred) / CLOCKS_PER_SEC);

//		if(i == 1)
//			PrintPrediction(v_fPredValue_fixed);

		begin_gd = clock();
		splitter.ComputeGDSparse(v_fPredValue_fixed, m_vTrueValue_fixedPos);
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

		clock_t end_round = clock();
		cout << "elapsed time of round " << i << " is " << (double(end_round - start_round) / CLOCKS_PER_SEC) << endl;
		cout << "split time = " << total_split_t << "; total find fea time = " << total_find_fea_t << endl;
	}

	cout << "pred sec = " << total_pred << "; gd sec = " << total_gd << "; grow sec = " << total_grow << endl;

}

/**
 * @brief: initialise tree
 */
void Trainer::InitTree(RegTree &tree)
{
	TreeNode *root = new TreeNode[1];
	m_nNumofNode = 1;
	root->nodeId = 0;
	root->level = 0;

	tree.nodes.push_back(root);

	//all instances are under node 0
	splitter.m_nodeIds.clear();
	for(int i = 0; i < m_vvInsSparse.size(); i++)
	{
		splitter.m_nodeIds.push_back(0);
	}

	total_find_fea_t = 0;
	total_split_t = 0;
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
	}
}

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void Trainer::GrowTree(RegTree &tree)
{
	int nNumofSplittableNode = 0;

	//start splitting this tree from the root node
	vector<TreeNode*> splittableNode;
	for(int i = 0; i < int(tree.nodes.size()); i++)
	{
		splittableNode.push_back(tree.nodes[i]);
		nNumofSplittableNode++;
	}

	//split node(s)
	int nCurDepth = 0;
	while(nNumofSplittableNode > 0)
	{
		//for each splittable node
		vector<SplitPoint> vBest;

		//these variables may be reused for optimisation
		vector<nodeStat> rchildStat, lchildStat;

		int bufferSize = splitter.mapNodeIdToBufferPos.size();//maps node id to buffer position
		vBest.resize(bufferSize);
		rchildStat.resize(bufferSize);
		lchildStat.resize(bufferSize);

		//efficient way to find the best split
		clock_t begin_find_fea = clock();
		splitter.FeaFinderAllNode(vBest, rchildStat, lchildStat);
		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

	vector<TreeNode*> newSplittableNode;
	vector<nodeStat> newNodeStat;

		//for each splittable node
		for(int n = 0; n < nNumofSplittableNode; n++)
		{
			int bufferPos = splitter.mapNodeIdToBufferPos[splittableNode[n]->nodeId];
			//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
			if(vBest[bufferPos].m_fGain <= 0 || m_nMaxDepth == nCurDepth)
			{
				//compute weight of leaf nodes
				splittableNode[n]->predValue = splitter.ComputeWeightSparseData(bufferPos);
			}
			else
			{
				clock_t start_split_t = clock();
				//split the current node
				splitter.SplitNodeSparseData(splittableNode[n], newSplittableNode, vBest[bufferPos], tree, m_nNumofNode);

				//push left and right child statistics into a vector
				newNodeStat.push_back(lchildStat[bufferPos]);
				newNodeStat.push_back(rchildStat[bufferPos]);
				clock_t end_split_t = clock();
				total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);
			}

			//marked the split node or the leaf node as "processed"
			splitter.MarkProcessed(splittableNode[n]->nodeId);
		}

		nCurDepth++;

		splitter.UpdateNodeStat(newSplittableNode, newNodeStat);

		//assign new splittable nodes to the container
		splittableNode.clear();
		splittableNode = newSplittableNode;
		nNumofSplittableNode = splittableNode.size();
	}
}

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void Trainer::GrowTreeGoodSplit(RegTree &tree)
{
	int nNumofSplittableNode = 0;

	//start splitting this tree from the root node
	vector<TreeNode*> splittableNode;
	for(int i = 0; i < int(tree.nodes.size()); i++)
	{
		splittableNode.push_back(tree.nodes[i]);
		nNumofSplittableNode++;
	}

	//split node(s)
	int nCurDepth = 0;
	while(splittableNode.size() > 0 && nCurDepth <= m_nMaxDepth)
	{
		cout << "splitting " << nCurDepth << " level..." << endl;
		//for each splittable node
		vector<SplitPoint> vBest;

		//these variables may be reused for optimisation
		vector<nodeStat> rchildStat, lchildStat;

		int bufferSize = splitter.mapNodeIdToBufferPos.size();//maps node id to buffer position
		vBest.resize(bufferSize);
		rchildStat.resize(bufferSize);
		lchildStat.resize(bufferSize);

		//efficient way to find the best split
		clock_t begin_find_fea = clock();
		splitter.FeaFinderAllNode(vBest, rchildStat, lchildStat);
		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

		//split all the splittable nodes
		clock_t start_split_t = clock();
		bool bLastLevel = false;
		if(nCurDepth == m_nMaxDepth)
			bLastLevel = true;
		splitter.SplitAll(splittableNode, vBest, tree, m_nNumofNode, rchildStat, lchildStat, bLastLevel);
		clock_t end_split_t = clock();
		total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);

		nCurDepth++;
	}
}

/**
 * @brief: print out a learned tree
 */
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
 * @brief: release memory used by trees
 */
void Trainer::ReleaseTree(vector<RegTree> &v_Tree)
{
	int nNumofTree = v_Tree.size();
	for(int i = 0; i < nNumofTree; i++)
	{
		int nNumofNodes = v_Tree[i].nodes.size();
		for(int j = 0; j < nNumofNodes; j++)
		{
			delete[] v_Tree[i].nodes[j];
		}
	}
}

/**
 * @brief: print the predicted values
 */
void Trainer::PrintPrediction(const vector<double> &vPred)
{
	int n = vPred.size();
	ofstream out("prediction.txt");
	out << "number of values is " << n << endl;
	for(int i = 0; i < n; i++)
	{
		out << vPred[i] << "\t";
		if(i != 0 && i % 50 == 0)
			out << endl;
	}
	out << endl;
}

