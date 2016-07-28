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
#include <boost/interprocess/mapped_region.hpp>

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
	static bool ReadPartOfRowFromFile(boost::interprocess::mapped_region*, float_point *pContent, int nFullRowSize, int nNumofElementsToRead, long long nIndexof1stElement);

	static bool IsFileExist(string strFileName);
	static void CreateFolder(string folderName);

	/*
	 * @brief: write one Hessian row to file
	 * @return: the starting position of this Hessian row
	 */
	template<class T>
	static int WriteToFile(ofstream &writeOut, T *pContent, int nSizeofContent)
	{
		int nReturn = -1;
		if(!writeOut.is_open() || pContent == NULL || nSizeofContent <= 0)
		{
			cerr << "write content to file failed: input param invalid" << endl;
			return nReturn;
		}

		//return the starting postion of this Hessian row
		nReturn = writeOut.tellp();
		//for(int i = 0; i < nSizeofContent; i++)
		//{
			writeOut.write((char*)pContent, sizeof(T) * nSizeofContent);
			//writeOut /*<< setprecision(16)*/ << (float_point)*pContent;
			//if(i != nSizeofContent - 1)
			//	writeOut << " : ";
		//	pContent++;
		//}
		//writeOut << endl;

		return nReturn;
	}

	/*
	 * @brief: read a continuous part from a file
	 * @param: pContent: storing the read content
	 * @param: nFullRowSize: the size of a full row in hessian matrix because we store the whole hessian matrix
	 * @param: nNumofElementsToRead: the number of elements read by this function
	 * @param: nIndexof1stElement: the start point of this reading procedure.
	 */
	template<class T>
	static void ReadPartOfRowFromFile(FILE *&readIn, T *pContent, int nNumofElementsToRead, long long nIndexof1stElement)
	{
		//bool bReturn = false;

		assert(readIn != NULL && pContent != NULL && nNumofElementsToRead > 0 && nIndexof1stElement >= 0);
		//find the position of this Hessian row
		fseek(readIn, 0, SEEK_END);
		assert(ftell(readIn) != 0);

		long long nSeekPos = sizeof(T) * nIndexof1stElement;
		fseek(readIn, nSeekPos, SEEK_SET);
//		cout << ftell(readIn) << endl;
		assert(ftell(readIn) != -1);

		fread(pContent, sizeof(T), nNumofElementsToRead, readIn);

//		cout << ftell(readIn) << endl;
//		assert(nNumofRead > 0);
//		cout << "the number of kernel values read " << nNumofRead << endl;
		if(ferror(readIn) == true)
		{
			cout  << "read kernel values from file error" << endl;
		}

		//clean eof bit, when pointer reaches end of file
		if(feof(readIn))
		{
//			cout << "end of file is reached" << endl;
			rewind(readIn);
		}
	}
};


#endif /* FILEOPS_H_ */
