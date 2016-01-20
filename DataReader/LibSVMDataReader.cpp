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

void LibSVMDataReader::ReadLibSVMDataFormat(vector<vector<float_point> > &v_vInstance, vector<float_point> &v_fValue,
									  	    string strFileName, int nNumofFeatures, int nNumofExamples)
{
	ifstream readIn;
	readIn.open(strFileName.c_str());
	assert(readIn.is_open());
	vector<float_point> vSample;

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
			i++;
			//assert(x > 0 && x <= 1);
			//cout << nFeature << " " << cColon << endl;
			assert(cColon == ':');
			while(int(vSample.size()) < nFeature - 1 && int(vSample.size()) < nNumofFeatures)
			{
				vSample.push_back(0);
			}
			if(nNumofFeatures == int(vSample.size()))
			{
				break;
			}
			assert(int(vSample.size()) <= nNumofFeatures);
			vSample.push_back(x);
		}
		//fill the value of the rest of the features as 0
		while(int(vSample.size()) < nNumofFeatures)
		{
			vSample.push_back(0);
		}
		v_vInstance.push_back(vSample);

		//clear vector
		vSample.clear();
	} while (readIn.eof() != true && j < nNumofExamples);///72309 is the number of samples

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		//cout << "end of file" << endl;
		readIn.clear();
	}
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
