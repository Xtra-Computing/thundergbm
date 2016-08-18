/*
 * VenueRecTask.cpp
 *
 *  Created on: 14/08/2014
 *      Author: Zeyi Wen
 */

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>

#include "GBDTTask.h"
#include "../DeviceTrainer.h"

using std::cout;
using std::endl;

//MakeInstanceGetter(GBDTTask);

int GBDTTask::AllocateReusableResources()
{
	return 0;
}

int GBDTTask::ReleaseResources()
{
	return 0;
}

void UnpackData(DataPack *pDataPack, string &strFileNames)
{
	strFileNames = pDataPack->ypData[1];
}

void* GBDTTask::ProcessTask(void* pInputParam)
{
//	checkCudaErrors(cudaSetDevice(0));//use the same context
	TaskParam *pTaskParam = (TaskParam*)pInputParam;
	cudaStream_t *pStream_gbdt = pTaskParam->pCudaStream;
	CUcontext *context = pTaskParam->pCudaContext;
	cuCtxSetCurrent(*context);
	cudaStreamSynchronize(*pStream_gbdt);

	DeviceSplitter splitter;
	DeviceTrainer trainer(&splitter);
	vector<RegTree> v_Tree;
	trainer.TrainGBDT(v_Tree, pStream_gbdt);

	pTaskParam->pResult = NULL;
	cudaStreamSynchronize(*pStream_gbdt);

	*pTaskParam->thread_status = 0;

	return NULL;
}



