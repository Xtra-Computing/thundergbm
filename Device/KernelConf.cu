/*
 * KernelConf.cu
 *
 *  Created on: 29 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include "KernelConf.h"
#include "KernelConst.h"

using std::cout;
using std::endl;

int KernelConf::m_maxBlockSize = -1;
int KernelConf::m_maxNumofBlockOneDim = -1;

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Ceil(a, b) (a%b!=0)?((a/b)+1):(a/b)


KernelConf::KernelConf()
{
	m_maxBlockSize = BLOCK_SIZE;
	m_maxNumofBlockOneDim = 65535;
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

void KernelConf::ConfKernel(int totalNumofThread, int &threadPerBlock, dim3 &dimNumofBlock)
{
	if(totalNumofThread > m_maxBlockSize)
		threadPerBlock = m_maxBlockSize;
	else
		threadPerBlock = totalNumofThread;

	int numofBlockX = 0;
	numofBlockX = Ceil(totalNumofThread, threadPerBlock);
	int numofBlockY = 0;
	numofBlockY = Ceil(numofBlockX, m_maxNumofBlockOneDim);

	if(numofBlockX > m_maxNumofBlockOneDim)
		numofBlockX = m_maxNumofBlockOneDim;

	dim3 dimTempNumofBlock(numofBlockX, numofBlockY);
//	cout << temp.x << " v.s. " << nGridDimX << "; y " << temp.y << " v.s. " << numofBlockY << endl;
	dimNumofBlock = dimTempNumofBlock;
}

