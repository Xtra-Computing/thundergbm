/*
 * gpuMemAllocator.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: Allocate GPU memory for the whole project
 */

#ifndef GPUMEMALLOCATOR_H_
#define GPUMEMALLOCATOR_H_

#include <vector>

typedef double float_point;

using std::vector;

class GPUMemManager
{
public:
	void MemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	void MemcpyDeviceToHost(void *pDevSrc, void *pHostDst, int numofByte);

	void TestMemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	void TestMemcpyDeviceToHost();

	//convert a vector to an array
	template<class T>
	void VecToArray(vector<T> vec, T* arr)
	{
		int numofEle = vec.size();
		for(int i = 0; i < numofEle; i++)
		{
			arr[i] = vec[i];
		}
	}
};



#endif /* GPUMEMALLOCATOR_H_ */
