/*
 * BasePredictor.cpp
 *
 *  Created on: 27/05/2016
 *      Author: zeyi
 */

#include "BasePredictor.h"
#include "../../pureHost/SparsePred/DenseInstance.h"
#include "../../DeviceHost/MyAssert.h"

/**
 * @brief: prediction function for sparse instances
 */
void BasePredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree,
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
		{
			denseInsConverter.SparseToDense(v_vInstance[i], vDense);
			//denseInsConverter.PrintDenseVec(vDense);
		}
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

	PROCESS_ERROR(v_fPredValue.size() == v_vInstance.size());
}


/**
 * @brief: prediction function for sparse instances
 */
void BasePredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue)
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



