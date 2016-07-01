/*
 * KernelConf.cu
 *
 *  Created on: 29 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include "KernelConf.h"

using std::cout;
using std::endl;

int KernelConf::m_maxBlockSize = -1;

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Ceil(a, b) (a%b!=0)?((a/b)+1):(a/b)


KernelConf::KernelConf()
{
	m_maxBlockSize = 128;
}

void KernelConf::ComputeBlock(int numofThread, dim3 &dimGridThinThread)
{
	int nGridDimY = 0;
	nGridDimY = Ceil(numofThread, m_maxBlockSize);

	int nGridDimX = 0;
	if(numofThread > m_maxBlockSize)
		nGridDimX = m_maxBlockSize;
	else
		nGridDimX = numofThread;

	dim3 temp(nGridDimX, nGridDimY);
//	cout << temp.x << " v.s. " << nGridDimX << "; y " << temp.y << " v.s. " << nGridDimY << endl;
	dimGridThinThread = temp;
}
