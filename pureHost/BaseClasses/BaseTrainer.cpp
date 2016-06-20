/*
 * BaseTrainer.cpp
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: definition of the base trainer
 */

#include "BaseTrainer.h"

#include <ctime>

#include "../UpdateOps/HostSplitter.h"
#include "../HostPredictor.h"
#include "../Tree/TreeNode.h"
#include "../Tree/PrintTree.h"
#include "../Evaluation/RMSE.h"

/*
 * @brief: initialise constants of a trainer
 */
void BaseTrainer::InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea)
{
	m_nMaxNumofTree = nNumofTree;
	m_nMaxDepth = nMaxDepth;
	splitter->m_labda = fLabda;
	splitter->m_gamma = fGamma;

	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInsSparse.size(); i++)
	{
		m_vPredBuffer.push_back(0.0);
		gdpair gd;
		splitter->m_vGDPair_fixedPos.push_back(gd);
	}

	KeyValue::SortFeaValue(nNumofFea, m_vvInsSparse, splitter->m_vvFeaInxPair);
}

/**
 * @brief: training GBDTs
 */
void BaseTrainer::TrainGBDT(vector<RegTree> & vTree)
{
	clock_t begin_pred, begin_gd, begin_grow;
	clock_t end_pred, end_gd, end_grow;
	double total_pred = 0, total_gd = 0, total_grow = 0;

	HostPredictor pred;
	HostSplitter hsplit;
	for(int i = 0; i < m_nMaxNumofTree; i++)
	{
		splitter->m_nRound = i;
		cout << "start round " << i << endl;
		clock_t start_round = clock();
		//initialise a tree
		RegTree tree;
		InitTree(tree);

		//predict the data by the existing trees
		hsplit.m_vvInsSparse = m_vvInsSparse;
		hsplit.m_vPredBuffer = m_vPredBuffer;
		hsplit.m_vTrueValue = m_vTrueValue;
		hsplit.ComputeGD(vTree);

		/*
		vector<double> v_fPredValue;
		begin_pred = clock();
		pred.PredictSparseIns(m_vvInsSparse, vTree, v_fPredValue, m_vPredBuffer);
		end_pred = clock();
		total_pred += (double(end_pred - begin_pred) / CLOCKS_PER_SEC);

		if(i > 0)
		{
			//run the GBDT prediction process
			EvalRMSE rmse;
			double fRMSE = rmse.Eval(v_fPredValue, m_vTrueValue);
			cout << "rmse=" << fRMSE << endl;
		}

		begin_gd = clock();
		splitter->ComputeGDSparse(v_fPredValue, m_vTrueValue);
		end_gd = clock();
		total_gd += (double(end_gd - begin_gd) / CLOCKS_PER_SEC);
		*/

		//grow the tree
		begin_grow = clock();
		GrowTree(tree);
		end_grow = clock();
		total_grow += (double(end_grow - begin_grow) / CLOCKS_PER_SEC);

		cout << "tree " << i << " has " << tree.nodes.size() << " node(s)" << endl;

		//save the tree
		vTree.push_back(tree);
//		PrintTree(tree);

		clock_t end_round = clock();
		cout << "elapsed time of round " << i << " is " << (double(end_round - start_round) / CLOCKS_PER_SEC) << endl;
		cout << "split time=" << total_split_t << "; total find fea time=" << total_find_fea_t << "; prune time=" << total_prune_t << endl;
	}

	cout << "pred sec = " << total_pred << "; gd sec = " << total_gd << "; grow sec = " << total_grow << endl;

}

/**
 * @brief: initialise tree
 */
void BaseTrainer::InitTree(RegTree &tree)
{
	TreeNode *root = new TreeNode[1];
	m_nNumofNode = 1;
	root->nodeId = 0;
	root->level = 0;

	tree.nodes.push_back(root);

	//all instances are under node 0
	splitter->m_nodeIds.clear();
	for(int i = 0; i < m_vvInsSparse.size(); i++)
	{
		splitter->m_nodeIds.push_back(0);
	}

	total_find_fea_t = 0;
	total_split_t = 0;
	total_prune_t = 0;
}

/**
 * @brief: print out a learned tree
 */
void BaseTrainer::PrintTree(const RegTree &tree)
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
void BaseTrainer::SaveModel(string fileName, const vector<RegTree> &v_Tree)
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
void BaseTrainer::ReleaseTree(vector<RegTree> &v_Tree)
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
void BaseTrainer::PrintPrediction(const vector<double> &vPred)
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


