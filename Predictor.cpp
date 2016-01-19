/*
 * Predictor.cpp
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: predict value for each instance
 */

#include <assert.h>

#include "Predictor.h"


void Predictor::Predict(vector<vector<float> > &v_vInstance, vector<RegTree> &vTree, vector<float> &v_fPredValue, vector<float> &v_predBuffer)
{
	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = v_vInstance.size();

	for(int i = 0; i < nNumofIns; i++)
	{
		float fValue = v_predBuffer[i];

		//prediction using the last tree
		for(int t = nNumofTree - 1; t < nNumofTree; t++)
		{
			int nodeId = vTree[t].GetLeafIndex(v_vInstance[i]);
			fValue += vTree[t][nodeId].predValue;
		}

		v_fPredValue.push_back(fValue);
		v_predBuffer[i] = fValue;
	}

	assert(v_fPredValue.size() == v_vInstance.size());
}

