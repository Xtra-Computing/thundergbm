/*
 * KernelConf.h
 *
 *  Created on: 29 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef KERNELCONF_H_
#define KERNELCONF_H_

class KernelConf
{
public:
	static int m_maxBlockSize;
public:
	KernelConf();
	void ComputeBlock(int numofThread, dim3 &dimGridThinThread);
};



#endif /* KERNELCONF_H_ */
