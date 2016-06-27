/*
 * Predictor.cpp
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: predict value for each instance
 */

#include "HostPredictor.h"
#include "../DeviceHost/SparsePred/DenseInstance.h"
#include "../DeviceHost/MyAssert.h"

/**
 * @brief: prediction function for sparse instances
 */
void HostPredictor::PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue)
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
