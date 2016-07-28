/*
 * FileBuffer.h
 *
 *  Created on: 28 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef FILEBUFFER_H_
#define FILEBUFFER_H_

#include "../../DeviceHost/DefineConst.h"

class FileBuffer
{
private:
	static int *m_pInsId;
	static float_point *m_pfValue;
	static int *m_pNumofKeyValue;
	static long long *m_plFeaStartPos;//get start position of each feature

	//instances for prediction
	static int *m_pFeaId;
	static float_point *m_pfFeaValue;
	static int *m_pNumofFea;
	static long long *m_plInsStartPos;

	//labels of training instances
	static float_point *m_pfTrueLabel;

	//some lengths
	static int m_nNumofFeatures;
	static int m_nNumofExamples;
	static long long m_numFeaValue;

	static string strInsId;
	static string strFeaValueInsId;
	static string strNumofValueEachFea;
	static string strFeaStartPos;
	static string strFeaId;
	static string strFeaValueFeaId;
	static string strNumofFeaEachIns;
	static string strInsStartPos;
	static string strInsLabel;
	static string strDataSetInfo;
public:
	static void SetMembers(int *pInsId, float_point *pdValue, int *pNumofKeyValue, long long *plFeaStartPos,
						   int *pFeaId, float_point *pfFeaValue, int *pNumofFea, long long *plInsStartPos,
						   float_point *pfTrueLabel,
						   int nNumofFeatures, int nNumofExamples, long long numFeaValue);
	static void WriteBufferFile(string strFolder);

	static void ReadDataInfo(string strFolder, int &numFeature, int &numExample, long long &numFeaValue);
	static void ReadBufferFile(string strFolder, int *pInsId, float_point *pdValue, int *pNumofKeyValue, long long *plFeaStartPos,
							   int *pFeaId, float_point *pfFeaValue, int *pNumofFea, long long *plInsStartPos,
							   float_point *pfTrueLabel,
							   int nNumofFeatures, int nNumofExamples, long long numFeaValue);
};



#endif /* FILEBUFFER_H_ */
