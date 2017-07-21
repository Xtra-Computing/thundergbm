/*
 * GBDTMain.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: project main function
 */

#include <math.h>
#include <time.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "Host/Evaluation/RMSE.h"
#include "DeviceHost/svm-shared/fileOps.h"

#include "Device/Memory/gbdtGPUMemManager.h"
#include "Device/Bagging/BagManager.h"
#include "Device/GPUTask/GBDTTask.h"
#include "Device/DeviceTrainer.h"
#include "Device/DevicePredictor.h"
#include "SharedUtility/initCuda.h"
#include "SharedUtility/CudaMacro.h"
#include "SharedUtility/DataReader/LibsvmReaderSparse.h"
#include "Device/FindSplit/IndexComputer.h"

#include "Device/FileBuffer/FileBuffer.h"

#include "Host/GBDTCmdLineParser.h"

int main(int argc, char *argv[])
{
	char fileName[1024];
	char savedFileName[1024];
	GBDTCmdLineParser parser;

	parser.ParseLine(argc, argv, fileName, savedFileName);

	string strFileName = fileName;
	
	cout << "processing this file: " << strFileName << endl;

	CUcontext context;
	if(!InitCUDA(context)){
		cerr << "cannot initialise GPU" << endl;
		return 0;
	}

	clock_t begin_whole, end_whole;
	/********* read training instances from a file **************/
	int numBag = parser.numBag;//number of bags for bagging

	//for training
	int nNumofTree = parser.numTree;
	int nMaxDepth = parser.depth;
	if(nMaxDepth > 8){//change unsigned char to uint for supporting deeper trees.
		cerr << "maximum supported depth is 8! Please use more trees. " << endl;
		exit(0);
	}
	double fLabda = 1;//this one is constant in xgboost
	double fGamma = parser.gamma;//minimum loss
	int maxNumofNodePerTree = pow(2, nMaxDepth + 1) - 1;
	int maxNumofSplittableNode = pow(2, nMaxDepth - 1);
	int numInternalNode = pow(2, nMaxDepth) - 1;
	int maxNumofUsedFeature = numInternalNode;

	DevicePredictor pred;

	DeviceSplitter splitter;
	DeviceTrainer trainer(&splitter);

	cout << "reading data..." << endl;
	LibSVMDataReader dataReader;
	int numFea;
	int numIns;
	uint numFeaValue;

	string strFolder = strFileName + "-folder/";
	string bufferFileName = strFolder + "data-info.buffer";
	bool bBufferFileExist = CFileOps::IsFileExist(bufferFileName);
	bool bUsedBuffer = bBufferFileExist;

	vector<real> v_fLabel;
	vector<vector<KeyValue> > v_vInsSparse;
	clock_t start_init, end_init;
	//read the file the first time
	if(bBufferFileExist == false){
		dataReader.GetDataInfo(strFileName, numFea, numIns, numFeaValue);
		dataReader.ReadLibSVMAsSparse(v_vInsSparse, v_fLabel, strFileName, numFea, numIns);
		trainer.m_vvInsSparse = v_vInsSparse;//later used in sorting values for each feature
		trainer.m_vTrueValue = v_fLabel;
	}
	else{
		//read data set info
		FileBuffer::ReadDataInfo(strFolder, numFea, numIns, numFeaValue);
	}
	cout << "data has " << numFea << " features, " << numIns << " instances, and " << numFeaValue << " fea values" << endl;
	if(maxNumofUsedFeature > numFea)//maximum number of used features of a tree
		maxNumofUsedFeature = numFea;

	//fill the bags
	BagManager bagManager;
	bagManager.InitBagManager(numIns, numFea, nNumofTree, numBag, maxNumofSplittableNode,
							  maxNumofNodePerTree, numFeaValue, maxNumofUsedFeature, nMaxDepth);

	start_init = clock();
	trainer.InitTrainer(nNumofTree, nMaxDepth, fLabda, fGamma, numFea, bUsedBuffer);
	end_init = clock();

	//store feature key-value into array
	int *pInsId = new int[numFeaValue];
	real *pdValue = new real[numFeaValue];
	int *pNumofKeyValue = new int[numFea];
	uint *plFeaStartPos = new uint[numFea];//get start position of each feature

	//instances for prediction
	int *pFeaId = new int[numFeaValue];//continuous elements for each instance
	real *pfFeaValue = new real[numFeaValue];
	int *pNumofFea = new int[numIns];
	uint *plInsStartPos = new uint[numIns];

	real *pTrueLabel = new real[numIns];

	if(bBufferFileExist == false)
	{
		KeyValue::VecToArray(trainer.splitter->m_vvFeaInxPair, pInsId, pdValue, pNumofKeyValue, plFeaStartPos);
		//store sparse instances to GPU memory for prediction
		KeyValue::VecToArray(trainer.m_vvInsSparse, pFeaId, pfFeaValue, pNumofFea, plInsStartPos);

		for(int i = 0; i < numIns; i++)
		{
			pTrueLabel[i] = v_fLabel[i];
		}

		//saved to buffer file
		FileBuffer::SetMembers(pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
							   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
							   pTrueLabel,
							   numFea, numIns, numFeaValue);
		FileBuffer::WriteBufferFile(strFolder);
	}
	else//read the arrays from buffer
	{
		cout << "read from buffer file: " << bufferFileName << endl;
		FileBuffer::ReadBufferFile(strFolder, pInsId, pdValue, pNumofKeyValue, plFeaStartPos,
								   pFeaId, pfFeaValue, pNumofFea, plInsStartPos,
								   pTrueLabel,
								   numFea, numIns, numFeaValue);
	}
	bagManager.m_pTrueLabel_h = pTrueLabel;

	//initialise gpu memory allocator
	GBDTGPUMemManager manager;
	PROCESS_ERROR(numFeaValue > 0);
	manager.m_numFeaValue = numFeaValue;
	manager.m_maxUsedFeaInATree = maxNumofUsedFeature;

	//allocate memory for instances
	manager.mallocForTrainingIns(numFeaValue, numIns, numFea);
	manager.mallocForTestingIns(numFeaValue, numIns, numFea, numBag, nNumofTree, maxNumofNodePerTree);

	begin_whole = clock();
	cout << "start training..." << endl;
	/********* run the GBDT learning process ******************/
	Pruner::min_loss = fGamma;

	//initialise index computer object
	IndexComputer indexCom;
	indexCom.m_totalFeaValue = numFeaValue;
	indexCom.m_total_copy = 0;

	//copy feature key-value to device memory
	cudaMemcpy(manager.m_pDInsId, pInsId, numFeaValue * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pdDFeaValue, pdValue, numFeaValue * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pDNumofKeyValue, pNumofKeyValue, numFea * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pFeaStartPos, plFeaStartPos, numFea * sizeof(unsigned int), cudaMemcpyHostToDevice);

	//copy instance key-value to device memory for prediction
	cudaMemcpy(manager.m_pDFeaId, pFeaId, numFeaValue * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pdDInsValue, pfFeaValue, numFeaValue * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pDNumofFea, pNumofFea, numIns * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(manager.m_pInsStartPos, plInsStartPos, numIns * sizeof(uint), cudaMemcpyHostToDevice);

	//free host memory
	delete []pdValue;
	delete []pNumofKeyValue;
	delete []pFeaId;
	delete []pfFeaValue;
	delete []pNumofFea;
	delete []plInsStartPos;

	//copy true labels to gpu memory
	for(int i = 0; i < numBag; i++)
		cudaMemcpy(bagManager.m_pdTrueTargetValueEachBag + i * bagManager.m_numIns, pTrueLabel, numIns * sizeof(real), cudaMemcpyHostToDevice);

	clock_t start_train_time = clock();

	//using multiple CPU threads
	GBDTTask gbdtTask;
	ThreadParam *thdInput = new ThreadParam[numBag];
	TaskParam *taskParam = new TaskParam[numBag];
	cudaStream_t *pStream = new cudaStream_t[numBag];
	DataPack *pDataPack = new DataPack[numBag];
	int *pThreadStatus = new int[numBag];
	vector<pthread_t> vTid;
	for(int bag = 0; bag < numBag; bag++)
	{
		pThreadStatus[bag] = 1;//it is not used
		checkCudaErrors(cudaStreamCreate(&pStream[bag]));
		taskParam[bag].pCudaStream = &pStream[bag];
		taskParam[bag].thread_status = &pThreadStatus[bag];

		taskParam[bag].pDataPack = NULL;
		taskParam[bag].pCudaContext = &context;

		taskParam[bag].pResult = NULL;
		thdInput[bag].pObj = &gbdtTask;
		thdInput[bag].pThdParam = &taskParam[bag];
		pDataPack[bag].nNumofSeg = 1;
		pDataPack[bag].pnSizeofSeg = new int[1];
		pDataPack[bag].pnSizeofSeg[0] = 1;
		pDataPack[bag].ypData = new char*[1];
		pDataPack[bag].ypData[0] = new char[4];
		memcpy(pDataPack[bag].ypData[0], &bag, sizeof(int));
		taskParam[bag].pDataPack = &pDataPack[bag];

		pthread_attr_t attr;
		pthread_t tid;//cpu thread id
		pthread_attr_init(&attr);
//		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);//PTHREAD_CREATE_DETACHED
		int nCreateTd = pthread_create(&tid, &attr, gbdtTask.Process, (void*) &thdInput[bag]);
		pthread_attr_destroy(&attr);
		cout << "thread id " << tid << endl;
		if(nCreateTd != 0)
			cerr << "creating thread failed" << endl;
		vTid.push_back(tid);
	}
	//synchronise threads
	for(int i = 0; i < numBag; i++)
	{
		cout << "join thread id " << vTid[i] << endl;
		pthread_join(vTid[i], NULL);
	}
	clock_t end_train_time = clock();

	//save the trees to a file
	end_whole = clock();

	double total_copy_indexCom = (double(indexCom.m_total_copy)) / CLOCKS_PER_SEC;
	cout << "total copy time = " << total_copy_indexCom << endl;
	double total_init = (double(end_init - start_init) / CLOCKS_PER_SEC);
	cout << "total init time = " << total_init << endl;
	double total_train = (double(end_train_time - start_train_time) / CLOCKS_PER_SEC);
	cout << "total training time (-extra idx comp) = " << total_train - total_copy_indexCom << endl;
	double total_all = (double(end_whole - begin_whole) / CLOCKS_PER_SEC);
	cout << "all sec = " << total_all << endl;
	cout << "Done" << endl;

	manager.freeMemForTrainingIns();
	manager.freeMemForTestingIns();

	//free host memory
	delete []pInsId;
	delete []plFeaStartPos;

	ReleaseCuda(context);
	return 0;
}
