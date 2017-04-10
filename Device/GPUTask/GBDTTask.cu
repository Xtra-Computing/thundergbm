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
#include "../DevicePredictor.h"
#include "../../Host/Evaluation/RMSE.h"
#include "../Bagging/BagManager.h"

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

void UnpackData(DataPack *pDataPack, int &bagId)
{
	bagId = *((int*)pDataPack->ypData[0]);
}

void* GBDTTask::ProcessTask(void* pInputParam)
{
//	checkCudaErrors(cudaSetDevice(0));//use the same context
	TaskParam *pTaskParam = (TaskParam*)pInputParam;
	cudaStream_t *pStream_gbdt = pTaskParam->pCudaStream;
	CUcontext *context = pTaskParam->pCudaContext;
	int bagId;
	UnpackData(pTaskParam->pDataPack, bagId);
	cuCtxSetCurrent(*context);
	cudaStreamSynchronize(*pStream_gbdt);

	DeviceSplitter splitter;
	DeviceTrainer trainer(&splitter);

	vector<RegTree> v_Tree;
	trainer.TrainGBDT(v_Tree, pStream_gbdt, bagId);

	cout << "saved to file" << endl;
	trainer.SaveModel("tree.txt", v_Tree);

	//run the GBDT prediction process
	DevicePredictor pred;
	clock_t begin_pre, end_pre;
	vector<float_point> v_fPredValue;

	begin_pre = clock();
	vector<vector<KeyValue> > dummy;
	pred.PredictSparseIns(dummy, v_Tree, v_fPredValue, pStream_gbdt, bagId);
	end_pre = clock();
	double prediction_time = (double(end_pre - begin_pre) / CLOCKS_PER_SEC);
	cout << "prediction sec = " << prediction_time << endl;

	cudaStreamSynchronize(*pStream_gbdt);
	EvalRMSE rmse;
	float fRMSE = rmse.Eval(v_fPredValue, BagManager::m_pTrueLabel_h, v_fPredValue.size());
	cout << "rmse=" << fRMSE << endl;

	trainer.ReleaseTree(v_Tree);

	pTaskParam->pResult = NULL;
	cudaStreamSynchronize(*pStream_gbdt);

	*pTaskParam->thread_status = 0;

	return NULL;
}



