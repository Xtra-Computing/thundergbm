/*
 * BaseTrainer.cpp
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: definition of the base trainer
 */

#include "BaseTrainer.h"

#include <ctime>

#include "../../Host/UpdateOps/HostSplitter.h"
#include "../../Host/HostPredictor.h"
#include "../../DeviceHost/TreeNode.h"
#include "../../Host/Tree/PrintTree.h"
#include "../../Host/Evaluation/RMSE.h"
#include "../../Device/Splitter/DeviceSplitter.h"

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

		if(splitter->SpliterType().compare("host") == 0)
		{
			((HostSplitter*)splitter)->m_vPredBuffer = m_vPredBuffer;
			((HostSplitter*)splitter)->m_vTrueValue = m_vTrueValue;
			splitter->ComputeGD(vTree, m_vvInsSparse);
			m_vPredBuffer = ((HostSplitter*)splitter)->m_vPredBuffer;
		}
		else
			splitter->ComputeGD(vTree, m_vvInsSparse);

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


