/*
 * Predictor.cpp
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: predict value for each instance
 */

#include <assert.h>

#include "Predictor.h"
#include "SparsePred/DenseInstance.h"

/**
 * @brief: prediction function for dense instances
 */
void Predictor::PredictDenseIns(vector<vector<double> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue, vector<double> &v_predBuffer)
{
	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = v_vInstance.size();

	for(int i = 0; i < nNumofIns; i++)
	{
		double fValue = v_predBuffer[i];

		//prediction using the last tree
		for(int t = nNumofTree - 1; t >= 0 && t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIndex(v_vInstance[i]);
			fValue += vTree[t][nodeId]->predValue;
		}

		v_fPredValue.push_back(fValue);
		v_predBuffer[i] = fValue;
	}

	assert(v_fPredValue.size() == v_vInstance.size());
}

/**
 * @brief: prediction function for sparse instances
 */
void Predictor::PredictSparseIns(vector<vector<key_value> > &v_vInstance, vector<RegTree> &vTree,
								 vector<double> &v_fPredValue, vector<double> &v_predBuffer)
{
	DenseInsConverter denseInsConverter(vTree);
	//for each tree
	int nNumofIns = v_vInstance.size();

	for(int i = 0; i < nNumofIns; i++)
	{
		double fValue = v_predBuffer[i];

		//start prediction ###############
		int nNumofTree = vTree.size();

		vector<double> vDense;
		if(nNumofTree > 0)
			denseInsConverter.SparseToDense(v_vInstance[i], vDense);
		//prediction using the last tree
		for(int t = nNumofTree - 1; t >= 0 && t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIdSparseInstance(vDense, denseInsConverter.fidToDensePos);
			fValue += vTree[t][nodeId]->predValue;
		}

		//end prediction #################

		v_fPredValue.push_back(fValue);
		v_predBuffer[i] = fValue;
	}

	assert(v_fPredValue.size() == v_vInstance.size());
}


/**
 * @brief: prediction function for sparse instances
 */
void Predictor::PredictSparseIns(vector<vector<key_value> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue)
{
	DenseInsConverter denseInsConverter(vTree);
	//for each tree
	int nNumofIns = v_vInstance.size();

	for(int i = 0; i < nNumofIns; i++)
	{
		//start prediction ###############
		int nNumofTree = vTree.size();
		double fValue = 0;

		vector<double> vDense;
		if(nNumofTree > 0)
			denseInsConverter.SparseToDense(v_vInstance[i], vDense);
		//prediction using the last tree
		for(int t = 0; t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIdSparseInstance(vDense, denseInsConverter.fidToDensePos);
			fValue += vTree[t][nodeId]->predValue;
		}

		//end prediction #################

		v_fPredValue.push_back(fValue);
	}

	assert(v_fPredValue.size() == v_vInstance.size());
}
