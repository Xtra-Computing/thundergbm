/*
 * GPUTask.h
 *
 *  Created on: 13/08/2014
 *      Author: Zeyi Wen
 */

#ifndef GPUTASK_H_
#define GPUTASK_H_

#include <iostream>
#include <vector>
#include <map>

#include <cuda.h>
#include <driver_types.h>
#include <assert.h>
#include "Communicator.h"

using std::vector;
using std::string;
using std::map;

class GPUTask;

/*
 * @brief: input data from the TaskHandler to GPUTask program
 */
class TaskParam
{
public:
	cudaStream_t *pCudaStream;
	CUcontext *pCudaContext;//for all threads to use the same context
	int *thread_status;
	DataPack *pDataPack;
	DataPack *pResult;
};

class ThreadParam
{
public:
	GPUTask *pObj;
	void *pThdParam;
};

class GPUTask
{
public:
	GPUTask(){}
	virtual ~GPUTask(){}

	static void* Process(void *pInputParam)
	{
		ThreadParam *pTemp = (ThreadParam*)pInputParam;
		void *context = pTemp->pObj;
		assert(context != NULL);
		void *pTaskParam = pTemp->pThdParam;
		assert(pTaskParam != NULL);
		GPUTask *pGPUTask = (GPUTask*)context;
		assert(pGPUTask != NULL);

		void *pFunc = pGPUTask->ProcessTask(pTaskParam);
		return pFunc;
	}

	virtual int AllocateReusableResources(){return 0;};
	virtual int ReleaseResources(){return 0;}
	virtual void* ProcessTask(void*) = 0;
};

extern map<string, GPUTask *(*)()> gClassFactory;
#define MakeInstanceGetter(clazz) \
GPUTask *Make_##clazz(){ return new clazz; } \
GPUTask *(*ignore_##clazz##_clutter)() = gClassFactory[#clazz] = &Make_##clazz

#endif /* GPUTASK_H_ */
