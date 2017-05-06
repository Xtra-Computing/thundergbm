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
#include "../../Device/DevicePredictor.h"
#include "../../Host/Evaluation/RMSE.h"
#include "../../Device/Bagging/BagManager.h"


int BaseTrainer::m_nMaxNumofTree = -1;
int BaseTrainer::m_nMaxDepth = -1;

/*
 * @brief: initialise constants of a trainer
 */
void BaseTrainer::InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma, int nNumofFea, bool usedBuffer)
{
	m_nMaxNumofTree = nNumofTree;
	m_nMaxDepth = nMaxDepth;
	splitter->m_lambda = fLabda;
	splitter->m_gamma = fGamma;

	//initialise the prediction buffer
	for(int i = 0; i < (int)m_vvInsSparse.size(); i++)
	{
		m_vPredBuffer.push_back(0.0);
		gdpair gd;
		splitter->m_vGDPair_fixedPos.push_back(gd);
	}

	if(usedBuffer == false)
		KeyValue::SortFeaValue(nNumofFea, m_vvInsSparse, splitter->m_vvFeaInxPair);
}

/**
 * @brief: training GBDTs
 */
void BaseTrainer::TrainGBDT(vector<RegTree> & vTree, void *pStream, int bagId)
{
	clock_t begin_gd, begin_grow;
	clock_t end_gd, end_grow;
	double total_gd = 0, total_grow = 0, total_find_fea = 0, total_split = 0;

	total_init_t = 0;

	HostPredictor pred;
	HostSplitter hsplit;
	for(int i = 0; i < m_nMaxNumofTree; i++)
	{
		splitter->m_nRound = i;
		cout << "start round " << i << endl;
		clock_t start_round = clock();
		//initialise a tree
		RegTree tree;
		InitTree(tree, pStream, bagId);

		//predict the data by the existing trees
		begin_gd = clock();
		splitter->ComputeGD(vTree, m_vvInsSparse, pStream, bagId);
		end_gd = clock();
		total_gd += (double(end_gd - begin_gd) / CLOCKS_PER_SEC);

		//grow the tree
		begin_grow = clock();
		GrowTree(tree, pStream, bagId);
		end_grow = clock();
		total_grow += (double(end_grow - begin_grow) / CLOCKS_PER_SEC);

		cout << "tree " << i << " has " << tree.nodes.size() << " node(s)" << endl;

		//save the tree
		vTree.push_back(tree);
//		PrintTree(tree);

		clock_t end_round = clock();
		cout << "split time=" << total_split_t << "; total find fea time=" << total_find_fea_t << "; prune time=" << total_prune_t << endl;
		cout << "elapsed time of round " << i << " is " << (double(end_round - start_round) / CLOCKS_PER_SEC) << endl;
		total_find_fea += total_find_fea_t;
		total_split += total_split_t;

		//run the GBDT prediction process
		DevicePredictor pred;
		clock_t begin_pre, end_pre;
		vector<float_point> v_fPredValue;

		begin_pre = clock();
		vector<vector<KeyValue> > dummy;
		pred.PredictSparseIns(dummy, vTree, v_fPredValue, pStream, bagId);
		end_pre = clock();
		double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
		cout << "prediction sec = " << prediction_time << endl;

		EvalRMSE rmse;
		float fRMSE = rmse.Eval(v_fPredValue, BagManager::m_pTrueLabel_h, v_fPredValue.size());
		cout << "rmse=" << fRMSE << endl;
	}

	cout << "total: comp gd = " << total_gd << "; grow = " << total_grow << "; find fea = " << total_find_fea
		 << "; split = " << total_split << endl;
	cout << "total init for grow tree = " << total_init_t/ CLOCKS_PER_SEC << endl;

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
void BaseTrainer::PrintPrediction(const vector<float_point> &vPred)
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


