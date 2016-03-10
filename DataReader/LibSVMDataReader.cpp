/**
 * trainingDataIO.cu
 * @brief: this file includes the definition of functions for reading data
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#include "LibSVMDataReader.h"
#include <iostream>
#include <assert.h>
#include <sstream>

using std::istringstream;
using std::cout;
using std::endl;

/**
 * @brief: represent the data in a sparse form
 */
void LibSVMDataReader::ReadLibSVMFormatSparse(vector<vector<key_value> > &v_vInstance, vector<float_point> &v_fValue,
											  string strFileName, int nNumofFeatures, int nNumofInstance)
{
	ReaderHelper(v_vInstance, v_fValue, strFileName, nNumofFeatures, nNumofInstance, false);
}

/**
 * @brief: store the instances in a dense form
 */
void LibSVMDataReader::ReadLibSVMDataFormat(vector<vector<float_point> > &v_vInstance, vector<float_point> &v_fValue,
									  	    string strFileName, int nNumofFeatures, int nNumofExamples)
{
	vector<vector<key_value> > v_vInstanceKeyValue;
	ReaderHelper(v_vInstanceKeyValue, v_fValue, strFileName, nNumofFeatures, nNumofExamples, true);

	//convert key values to values only.
	for(int i = 0; i < nNumofExamples; i++)
	{
		vector<float_point> vIns;
		for(int j = 0; j < nNumofFeatures; j++)
		{
			vIns.push_back(v_vInstanceKeyValue[i][j].featureValue);
		}
		v_vInstance.push_back(vIns);
	}
}

/**
 * @brief: a function to read instances from libsvm format as either sparse or dense instances.
 */
void LibSVMDataReader::ReaderHelper(vector<vector<key_value> > &v_vInstance, vector<float_point> &v_fValue,
									string strFileName, int nNumofFeatures, int nNumofInstance, bool bUseDense)
{
	ifstream readIn;
	readIn.open(strFileName.c_str());
	assert(readIn.is_open());
	vector<key_value> vSample;

	//for storing character from file
	int j = 0;
	string str;
//	int nMissingCount = 0;

	//get a sample
	char cColon;
	do {
		j++;
		getline(readIn, str);

		istringstream in(str);
		int i = 0;
//		bool bMiss = false;
		float fValue = 0;
		in >> fValue;
		v_fValue.push_back(fValue);

		//get features of a sample
		int nFeature;
		float_point x;
		while (in >> nFeature >> cColon >> x)
		{
			//assert(x > 0 && x <= 1);
			//cout << nFeature << " " << cColon << endl;
			assert(cColon == ':');
			if(bUseDense == true)
			{
				while(int(vSample.size()) < nFeature - 1 && int(vSample.size()) < nNumofFeatures)
				{
					Push(i, 0, vSample);
					i++;
				}
			}

			if(nNumofFeatures == int(vSample.size()))
			{
				break;
			}
			assert(int(vSample.size()) <= nNumofFeatures);
			if(bUseDense == true)
				assert(i == nFeature - 1);

			Push(nFeature - 1, x, vSample);
			i++;
		}
		//fill the value of the rest of the features as 0
		if(bUseDense == true)
		{
			while(int(vSample.size()) < nNumofFeatures)
			{
				Push(i, 0, vSample);
				i++;
			}
		}

		v_vInstance.push_back(vSample);

		//clear vector
		vSample.clear();
	} while (readIn.eof() != true && j < nNumofInstance);

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		readIn.clear();
	}
}

/**
 * @brief:
 */
void LibSVMDataReader::Push(int feaId, float_point value, vector<key_value> &vIns)
{
	key_value pair;
	pair.id = feaId;
	pair.featureValue = value;
	vIns.push_back(pair);
}


/**
 * @brief: get the number of features and the number of instances of a dataset
 */
void LibSVMDataReader::GetDataInfo(string strFileName, int &nNumofFeatures, int &nNumofInstance)
{
	nNumofInstance = 0;
	nNumofFeatures = 0;

	ifstream readIn;
	readIn.open(strFileName.c_str());
	assert(readIn.is_open());

	//for storing character from file
	string str;

	//get a sample
	char cColon;
	while (readIn.eof() != true){
		getline(readIn, str);

		istringstream in(str);

		float fValue = 0;//label
		in >> fValue;

		//get features of a sample
		int nFeature;
		float_point x = -1;
		while (in >> nFeature >> cColon >> x)
		{
			assert(cColon == ':');
			if(nFeature > nNumofFeatures)
				nNumofFeatures = nFeature;
		}

		//skip an empty line (usually this case happens in the last line)
		if(x == -1)
			continue;

		nNumofInstance++;
	};

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		//cout << "end of file" << endl;
		readIn.clear();
	}

	readIn.close();
}
