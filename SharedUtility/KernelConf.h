/*
 * KernelConf.h
 *
 *  Created on: 29 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef KERNELCONF_H_
#define KERNELCONF_H_

#include <vector_types.h>

class KernelConf
{
public:
	static int m_maxBlockSize;
	static int m_maxNumofBlockOneDim;
public:
	KernelConf();
	void ComputeBlock(int numofThread, dim3 &dimGridThinThread);
	void ConfKernel(int totalNumofThread, int &threadPerBlock, dim3 &dimGridThinThread);
};

#define testing 	1

#endif /* KERNELCONF_H_ */
