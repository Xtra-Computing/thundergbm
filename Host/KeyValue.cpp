/*
 * KeyValue.cpp
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <algorithm>
#include <iostream>

#include "KeyValue.h"

using std::sort;
using std::cout;
using std::endl;

/**
 * @brief: sort a vector in a descendant order
 */
bool CmpValue(const KeyValue &a, const KeyValue &b)
{
  return a.featureValue > b.featureValue;
}

/**
 * @brief: (1) for each feature, sort the key value ordering based on the value; (2) store the sorted results to vectors
 */
void KeyValue::SortFeaValue(int nNumofDim, vector<vector<KeyValue> > &vvInsSparse, vector<vector<KeyValue> > &vvFeaInxPair)
{
	//sort the feature values for each feature
	vector<int> vCurParsePos;
	int nNumofIns = vvInsSparse.size();
	for(int i = 0; i < nNumofIns; i++)
	{
		vCurParsePos.push_back(0);
	}

#if analyse_data
	int maxParir = 50000;
	int count[maxParir];
	memset(count, 0, sizeof(int) * maxParir);
#endif

	for(int j = 0; j < nNumofDim; j++)
	{
		vector<KeyValue> featurePair;
		for(int i = 0; i < nNumofIns; i++)
		{
			int curTop = vCurParsePos[i];
			if(vvInsSparse[i].size() == curTop)
				continue;

			int curFeaId = vvInsSparse[i][curTop].id;
			if(curFeaId == j)
			{
				KeyValue kv;
				kv.id = i;
				kv.featureValue = vvInsSparse[i][curTop].featureValue;
				featurePair.push_back(kv);
				vCurParsePos[i] = vCurParsePos[i] + 1;
			}
		}

#if analyse_data
		count[featurePair.size()]++;
#endif
		sort(featurePair.begin(), featurePair.end(), CmpValue);

		vvFeaInxPair.push_back(featurePair);
	}
#if analyse_data
	for(int i = 0; i < maxParir; i++)
	{
		if(count[i] > 0)
			cout << count[i] << " feas have " << i << " values." << endl;
	}
#endif
}
