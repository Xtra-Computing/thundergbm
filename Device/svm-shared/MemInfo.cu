/*
 * MemInfo.cu
 *
 *  Created on: 19 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "MemInfo.h"
#include <sys/sysinfo.h>
#include <iostream>
#include <cuda.h>
#include <helper_cuda.h>

#include "../../DeviceHost/DefineConst.h"

using std::cout;
using std::endl;

/*
 * @brief: get the size of free memory in the form of float point representation
 */
long long MemInfo::GetFreeGPUMem()
{
	size_t nFreeMem, nTotalMem;
	checkCudaErrors(cuMemGetInfo_v2(&nFreeMem, &nTotalMem));
	cout << "GPU has " << float_point(nFreeMem)/(1024*1024*1024) << "GB(s) of free memory; ";
	if(nTotalMem > 0)
		cout << 100.0 * (double)nFreeMem/nTotalMem << "% of the total memory" << endl;
//	long long nMaxNumofFloatPoint = 0.9 * nFreeMem / sizeof(float_point);
	long long nMaxNumofFloatPoint = nFreeMem / sizeof(float_point);
	return nMaxNumofFloatPoint;
}


