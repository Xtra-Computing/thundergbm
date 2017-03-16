/**
 * fileOps.h
 * Created on: May 22, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef FILEOPS_H_
#define FILEOPS_H_

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "../../DeviceHost/DefineConst.h"

using std::string;
using std::fstream;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
using std::setprecision;

class CFileOps
{
public:
	static bool WriteToFile(ofstream &writeOut, float_point *pContent, int nNumofRows, int nNumofColumns);

	static bool ReadRowsFromFile(FILE *&readIn, float_point *&pContent, const int &nNumofElementsPerRow,
						  int nNumofRowsToRead, const int &nIndexofRow);
	//static bool ReadPartOfRowFromFile(boost::interprocess::mapped_region*, float_point *pContent, int nFullRowSize, int nNumofElementsToRead, long long nIndexof1stElement);

	static bool IsFileExist(string strFileName);
	static void CreateFolder(string folderName);

	/*
	 * @brief: write one Hessian row to file
	 * @return: the starting position of this Hessian row
	 */
	template<class T>
	static int WriteToFile(ofstream &writeOut, T *pContent, int numBytes)
	{
		if(!writeOut.is_open() || pContent == NULL || numBytes <= 0)
		{
			cerr << "write content to file failed: input param invalid" << endl;
			return -1;
		}

		if(writeOut.bad()){
			printf("error: failed before writing\n");
		}
		writeOut.write((char*)pContent, numBytes);
		if(writeOut.bad()){
			printf("error: failed after writing\n");
		}

		writeOut.flush();
		return 0;
	}

	/*
	 * @brief: read a continuous part from a file
	 * @param: pContent: storing the read content
	 * @param: nFullRowSize: the size of a full row in hessian matrix because we store the whole hessian matrix
	 * @param: nNumofElementsToRead: the number of elements read by this function
	 * @param: nIndexof1stElement: the start point of this reading procedure.
	 */
	template<class T>
	static void ReadPartOfRowFromFile(FILE *&readIn, T *pContent, int nNumofElementsToRead)
	{
		assert(readIn != NULL && pContent != NULL && nNumofElementsToRead > 0);
		rewind(readIn);
		fread(pContent, sizeof(T), nNumofElementsToRead, readIn);
	}
};


#endif /* FILEOPS_H_ */
