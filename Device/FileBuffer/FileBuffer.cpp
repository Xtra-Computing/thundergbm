/*
 * FileBuffer.cpp
 *
 *  Created on: 28 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include "../../DeviceHost/svm-shared/fileOps.h"
#include "FileBuffer.h"
#include "../../SharedUtility/CudaMacro.h"

using std::cerr;
using std::endl;

int *FileBuffer::m_pInsId = NULL;
real *FileBuffer::m_pfValue = NULL;
int *FileBuffer::m_pNumofKeyValue = NULL;
unsigned int *FileBuffer::m_plFeaStartPos = NULL;//get start position of each feature

//instances for prediction
int *FileBuffer::m_pFeaId = NULL;
real *FileBuffer::m_pfFeaValue = NULL;
int *FileBuffer::m_pNumofFea = NULL;
uint *FileBuffer::m_plInsStartPos = NULL;
real *FileBuffer::m_pfTrueLabel = NULL;	//labels of training instances

//some lengths
int FileBuffer::m_nNumofFeatures = -1;
int FileBuffer::m_nNumofExamples = -1;
uint FileBuffer::m_numFeaValue = -1;

//file names
string FileBuffer::strInsId = "InsId.dat";
string FileBuffer::strFeaValueInsId = "FeaValueInsId.dat";
string FileBuffer::strNumofValueEachFea = "NumofValueEachFea.dat";
string FileBuffer::strFeaStartPos = "FeaStartPos.dat";
string FileBuffer::strFeaId = "FeaId.dat";
string FileBuffer::strFeaValueFeaId = "FeaValueFeaId.dat";
string FileBuffer::strNumofFeaEachIns = "NumofFeaEachIns.dat";
string FileBuffer::strInsStartPos = "InsStartPos.dat";
string FileBuffer::strInsLabel = "Label.dat";
string FileBuffer::strDataSetInfo = "data-info.buffer";

/**
 * @brief: set all the members of this class
 */
void FileBuffer::SetMembers(int *pInsId, real *pdValue, int *pNumofKeyValue, uint *plFeaStartPos,
							int *pFeaId, real *pfFeaValue, int *pNumofFea, uint *plInsStartPos,
							real *pfTrueLabel,
							int nNumofFeatures, int nNumofExamples, uint numFeaValue)
{
	m_pInsId = pInsId;
	m_pfValue = pdValue;
	m_pNumofKeyValue = pNumofKeyValue;
	m_plFeaStartPos = plFeaStartPos;//get start position of each feature

	//instances for prediction
	m_pFeaId = pFeaId;
	m_pfFeaValue = pfFeaValue;
	m_pNumofFea = pNumofFea;
	m_plInsStartPos = plInsStartPos;

	m_pfTrueLabel = pfTrueLabel;

	//some lengths
	m_nNumofFeatures = nNumofFeatures;
	m_nNumofExamples = nNumofExamples;
	m_numFeaValue = numFeaValue;
}

/**
 * @brief: write buffer files for usage in next training
 */
void FileBuffer::WriteBufferFile(string strFolder)
{
	CFileOps::CreateFolder(strFolder);//create folder
	ofstream writeFile;

	writeFile.open(strFolder + strInsId, ofstream::trunc | std::ios::binary);
	int res = CFileOps::WriteToFile(writeFile, m_pInsId, sizeof(int) * m_numFeaValue);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strFeaValueInsId);
	res = CFileOps::WriteToFile(writeFile, m_pfValue, sizeof(real) * m_numFeaValue);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strNumofValueEachFea);
	res = CFileOps::WriteToFile(writeFile, m_pNumofKeyValue, sizeof(int) * m_nNumofFeatures);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strFeaStartPos);
	res = CFileOps::WriteToFile(writeFile, m_plFeaStartPos, sizeof(uint) * m_nNumofFeatures);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	//files for prediction
	writeFile.open(strFolder + strFeaId);
	res = CFileOps::WriteToFile(writeFile, m_pFeaId, sizeof(int) * m_numFeaValue);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strFeaValueFeaId);
	res = CFileOps::WriteToFile(writeFile, m_pfFeaValue, sizeof(real) * m_numFeaValue);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strNumofFeaEachIns);
	res = CFileOps::WriteToFile(writeFile, m_pNumofFea, sizeof(int) * m_nNumofExamples);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strInsStartPos);
	res = CFileOps::WriteToFile(writeFile, m_plInsStartPos, sizeof(uint) * m_nNumofExamples);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	writeFile.open(strFolder + strInsLabel);
	res = CFileOps::WriteToFile(writeFile, m_pfTrueLabel, sizeof(real) * m_nNumofExamples);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();

	//write dataset info
	writeFile.open(strFolder + strDataSetInfo);
	int numofInfo = 3;
	long long *lDataInfo = new long long[numofInfo];
	lDataInfo[0] = m_nNumofFeatures;
	lDataInfo[1] = m_nNumofExamples;
	lDataInfo[2] = m_numFeaValue;
	res =CFileOps::WriteToFile(writeFile, lDataInfo, sizeof(long long) * numofInfo);
	PROCESS_ERROR(res == -1);
	writeFile.close();
	writeFile.clear();
	delete[] lDataInfo;
}

/**
 * @brief: read the dataset information
 */
void FileBuffer::ReadDataInfo(string strFolder, int &numFeature, int &numExample, uint &numFeaValue)
{
	FILE *readFile = fopen((strFolder + strDataSetInfo).c_str(), "r");
	if(readFile == NULL)
	{
		cerr << "open " << strFolder + strDataSetInfo << " failed" << endl;
		exit(0);
	}
	int numofInfo = 3;
	long long *lDataInfo = new long long[numofInfo];
	CFileOps::ReadPartOfRowFromFile(readFile, lDataInfo, numofInfo);
	fclose(readFile);
	numFeature = lDataInfo[0];
	numExample = lDataInfo[1];
	numFeaValue = lDataInfo[2];
	delete []lDataInfo;
}

/**
 * @brief: read buffer files
 */
void FileBuffer::ReadBufferFile(string strFolder, int *pInsId, real *pdValue, int *pNumofKeyValue, uint *plFeaStartPos,
								int *pFeaId, real *pfFeaValue, int *pNumofFea, uint *plInsStartPos,
								real *pfTrueLabel,
								int numFeature, int numExample, long long numFeaValue)
{
	FILE *readFile = fopen((strFolder + strInsId).c_str(), "rb");
	printf("reading %s...\n", strInsId.c_str());
	CFileOps::ReadPartOfRowFromFile<int>(readFile, pInsId, numFeaValue);
	fclose(readFile);

	readFile = fopen((strFolder + strFeaValueInsId).c_str(), "r");
	printf("reading %s...\n", strFeaValueFeaId.c_str());
	CFileOps::ReadPartOfRowFromFile(readFile, pdValue, numFeaValue);
	fclose(readFile);

	readFile = fopen((strFolder + strNumofValueEachFea).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, pNumofKeyValue, numFeature);
	fclose(readFile);

	readFile = fopen((strFolder + strFeaStartPos).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, plFeaStartPos, numFeature);
	fclose(readFile);

	//files for prediction
	readFile = fopen((strFolder + strFeaId).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, pFeaId, numFeaValue);
	fclose(readFile);

	readFile = fopen((strFolder + strFeaValueFeaId).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, pfFeaValue, numFeaValue);
	fclose(readFile);

	readFile = fopen((strFolder + strNumofFeaEachIns).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, pNumofFea, numExample);
	fclose(readFile);

	readFile = fopen((strFolder + strInsLabel).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, pfTrueLabel, numExample);
	fclose(readFile);

	readFile = fopen((strFolder + strInsStartPos).c_str(), "r");
	CFileOps::ReadPartOfRowFromFile(readFile, plInsStartPos, numExample);
	fclose(readFile);
}
