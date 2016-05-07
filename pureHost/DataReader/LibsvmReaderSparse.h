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
#include "../KeyValue.h"
#include "BaseLibsvmReader.h"

using std::string;
using std::vector;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

typedef double float_point;

class LibSVMDataReader: public BaseLibSVMReader
{
public:
	LibSVMDataReader(){}
	~LibSVMDataReader(){}

	void ReadLibSVMDataFormat(vector<vector<float_point> > &v_vSample, vector<float_point> &v_fValue,
							  string strFileName, int nNumofFeatures, int nNumofInstance);



	void ReadLibSVMFormatSparse(vector<vector<KeyValue> > &v_vSample, vector<float_point> &v_fValue,
			  	  	  	  	  	string strFileName, int nNumofFeatures, int nNumofInstance);

private:
	void ReaderHelper(vector<vector<KeyValue> > &v_vSample, vector<float_point> &v_fValue,
	  	  	  		  string strFileName, int nNumofFeatures, int nNumofInstance, bool bUseDense);
	void Push(int feaId, float_point value, vector<KeyValue> &vIns);
};

#endif /* TRAININGDATAIO_H_ */
