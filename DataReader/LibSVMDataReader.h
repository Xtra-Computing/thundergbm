/**
 * trainingDataIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef TRAININGDATAIO_H_
#define TRAININGDATAIO_H_

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>

using std::string;
using std::vector;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

typedef float float_point;

class LibSVMDataReader
{
public:
	LibSVMDataReader(){}
	~LibSVMDataReader(){}

	void ReadLibSVMDataFormat(vector<vector<float_point> > &v_vSample, vector<float_point> &v_fValue,
										  string strFileName, int nNumofFeatures, int nNumofSamples);
};

#endif /* TRAININGDATAIO_H_ */
