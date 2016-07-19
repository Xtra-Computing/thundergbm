/*
 * MemInfo.h
 *
 *  Created on: 19 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: get the free memory of GPU
 */

#ifndef MEMINFO_H_
#define MEMINFO_H_

class MemInfo
{
public:

	static long long GetFreeGPUMem();
};



#endif /* MEMINFO_H_ */
