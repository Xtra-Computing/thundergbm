/*
 * keyValue.h
 *
 *  Created on: 9 Mar 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef KEYVALUE_H_
#define KEYVALUE_H_

#include <vector>
#include "../DeviceHost/DefineConst.h"
#include "../DeviceHost/MyAssert.h"


using std::vector;

class KeyValue
{
public:
	real featureValue;
	int id;//This id may be used as feature id or instance id

public:
	static void SortFeaValue(int nNumofDim, vector<vector<KeyValue> > &vvInsSparse, vector<vector<KeyValue> > &vvFeaInxPair);
	/**
	 * @brief: convert a vector to an array
	 * @vvFeaInxPair: each feature has a vector of key (which is instance id) and value
	 * @pId:  all the instance/feature ids corresponding pair
	 * @pdValue: all the feature values
	 * @pNumofKeyValue: number of key-value pairs of each feature
	 */
	template<class T>
	static void VecToArray(vector<vector<KeyValue> > &vvKeyValuePair, int *pId, real *pdValue, int *pNumofKeyValue, T *plStartPos)
	{
		PROCESS_ERROR(pId != NULL);
		PROCESS_ERROR(pdValue != NULL);
		PROCESS_ERROR(pNumofKeyValue != NULL);

		int nCur = 0;//the current processing key-value pair
		int nKeyValue = vvKeyValuePair.size();
		unsigned int totalPreValue = 0;
		for(int i = 0; i < nKeyValue; i++)
		{//for each feature
			vector<KeyValue> &vKV = vvKeyValuePair[i];
			unsigned int nNumofPair = vKV.size();
			pNumofKeyValue[i] = nNumofPair;

			//for each pair of key-value
			for(int p = 0; p < nNumofPair; p++)
			{
				pId[nCur] = vKV[p].id;
				pdValue[nCur] = vKV[p].featureValue;
				nCur++;//move to next key value
			}

			plStartPos[i] = totalPreValue;
			totalPreValue += nNumofPair;
		}
	}

	static void TestVecToArray(vector<vector<KeyValue> > &vvFeaInxPair, int *pInsId, real *pdValue, int *pNumofKeyValue);
};



#endif /* KEYVALUE_H_ */
