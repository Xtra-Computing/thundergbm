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
#include "../../SharedUtility/DataType.h"

using std::string;
using std::vector;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;


class LibSVMDataReader: public BaseLibSVMReader
{
public:
	LibSVMDataReader(){}
	~LibSVMDataReader(){}

	void ReadLibSVMDataFormat(vector<vector<real> > &v_vSample, vector<real> &v_fValue,
							  string strFileName, int nNumofFeatures, int nNumofInstance);



	void ReadLibSVMFormatSparse(vector<vector<KeyValue> > &v_vSample, vector<real> &v_fValue,
			  	  	  	  	  	string strFileName, int nNumofFeatures, int nNumofInstance);

private:
	void ReaderHelper(vector<vector<KeyValue> > &v_vSample, vector<real> &v_fValue,
	  	  	  		  string strFileName, int nNumofFeatures, int nNumofInstance, bool bUseDense);
	void Push(int feaId, real value, vector<KeyValue> &vIns);
};

#endif /* TRAININGDATAIO_H_ */
