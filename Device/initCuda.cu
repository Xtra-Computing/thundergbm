/*
 * initCuda.cpp
 *
 *  Created on: 10/12/2014
 *      Author: Zeyi Wen
 */

#include <helper_cuda.h>
#include <cuda.h>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

/**
 * @brief: initialize CUDA device
 */

bool InitCUDA(char gpuType, CUcontext &context)
{
    int count;

    checkCudaErrors(cudaGetDeviceCount(&count));
    if(count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    CUdevice device;

    int i;
    bool bUseTesla = false;
    for(i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
        	cout << prop.name << endl;
        	if(prop.name[0] == gpuType && prop.name[1] == 'e')
        	{
        		cout << "Using " << prop.name << endl;
       			bUseTesla = true;
//        		checkCudaErrors(cudaSetDevice(i));
        		cuDeviceGet(&device, i);
        		cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
        		break;
        	}
            if(prop.major >= 1)
            {
            	cout << count << " device(s) with compute capability " << prop.major << endl;
            }
        }
    }

    if(i == count)
    {
        cerr << "There is no device of " << gpuType << " series. Please reset the parameter of \""
        	 << __PRETTY_FUNCTION__ << "\"" <<endl;
        return false;
    }

    if(!bUseTesla)
    {
    	checkCudaErrors(cudaSetDevice(0));
    }

    return true;
}

bool ReleaseCuda(CUcontext &context)
{
	cuCtxDetach(context);
	return true;
}
