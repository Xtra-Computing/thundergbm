/*
 * DeviceTrainer.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICETRAINER_H_
#define DEVICETRAINER_H_

#include <climits>
#include "../DeviceHost/BaseClasses/BaseTrainer.h"
#include "Splitter/DeviceSplitter.h"
#include "../SharedUtility/CudaMacro.h"

class DeviceTrainer: public BaseTrainer
{
public:
	DeviceTrainer(DeviceSplitter *pSplitter):BaseTrainer(pSplitter){}
	virtual ~DeviceTrainer(){}

	virtual void InitTree(RegTree &tree, void *pStream, int bagId);
	virtual void GrowTree(RegTree &tree, void *pStream, int bagId);
	virtual void ReleaseTree(vector<RegTree> &v_Tree);

	static void StoreFinalTree(TreeNode *pAllNode, int numofNode, void *pStream, int bagId);
};

const int BLOCK_SIZE_ = 512;

const int NUM_BLOCKS = 32 * 56;

#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
#define KERNEL_LOOP2(i, n) \
  for (int i = blockIdx.y * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.y)
__global__ void kernel_div(uint *keys, int n_keys, int n_f);
__global__ void MarkPartition2(int preMaxNid, int *pFvToInsId, int *pInsIdToNodeId,
							  int totalNumFv, uint *pParitionMarker, uint *tid2fid, int n_f);

#endif /* DEVICETRAINER_H_ */
