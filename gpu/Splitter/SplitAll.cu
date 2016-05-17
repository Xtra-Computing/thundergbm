/*
 * DeviceSplitterSplitNode.cu
 *
 *  Created on: 12 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <algorithm>

#include "../../pureHost/MyAssert.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/SplitNodeMemManager.h"
#include "DeviceSplitter.h"
#include "../Preparator.h"
#include "DeviceSplitAllKernel.h"

using std::cout;
using std::endl;
using std::pair;
using std::make_pair;
using std::sort;


/**
 * @brief: split all splittable nodes of the current level
 * @numofNode: for computing new children ids
 */
void DeviceSplitter::SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel)
{
	int preMaxNodeId = m_nNumofNode - 1;

	int nNumofSplittableNode = splittableNode.size();
	PROCESS_ERROR(nNumofSplittableNode > 0);
	PROCESS_ERROR(splittableNode.size() == vBest.size());
	PROCESS_ERROR(vBest.size() == rchildStat.size());
	PROCESS_ERROR(vBest.size() == lchildStat.size());

	GBDTGPUMemManager manager;
	SNGPUManager snManager;//splittable node memory manager
	//copy the obtained tree nodes
	for(int t = 0; t < tree.nodes.size(); t++)
	{
		manager.MemcpyHostToDevice(tree.nodes[t], snManager.m_pTreeNode + t, sizeof(TreeNode) * 1);
	}

	//copy the splittable nodes to GPU memory
	for(int s = 0; s < splittableNode.size(); s++)
	{
		manager.MemcpyHostToDevice(splittableNode[s], manager.m_pSplittableNode + s, sizeof(TreeNode));
	}

	//compute the base_weight of tree node, also determines if a node is a leaf.
	ComputeWeight<<<1, 1>>>(snManager.m_pTreeNode, manager.m_pSplittableNode, manager.m_pSNIdToBuffId,
			  	  	  	  	  manager.m_pBestSplitPoint, manager.m_pSNodeStat, rt_eps, LEAFNODE,
			  	  	  	  	  m_labda, nNumofSplittableNode, bLastLevel);

	//original cpu code, now for testing
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = splittableNode[n]->nodeId;
//		cout << "node " << nid << " needs to split..." << endl;
		map<int, int>::iterator itBufferPos = mapNodeIdToBufferPos.find(nid);
		assert(itBufferPos != mapNodeIdToBufferPos.end());
		int bufferPos = mapNodeIdToBufferPos[nid];
		PROCESS_ERROR(bufferPos < vBest.size());
		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		tree.nodes[nid]->loss = vBest[bufferPos].m_fGain;
		tree.nodes[nid]->base_weight = ComputeWeightSparseData(bufferPos);
		if(vBest[bufferPos].m_fGain <= rt_eps || bLastLevel == true)
		{
			//weight of a leaf node
			tree.nodes[nid]->predValue = tree.nodes[nid]->base_weight;
			tree.nodes[nid]->rightChildId = LEAFNODE;
		}
	}

	//testing. Compare the results from GPU with those from CPU
//	cout << "numof tree nodes is " << tree.nodes.size() << endl;
	for(int t = 0; t < tree.nodes.size(); t++)
	{
		TreeNode tempNode;
		manager.MemcpyDeviceToHost(snManager.m_pTreeNode + t, &tempNode, sizeof(TreeNode) * 1);
		if(tempNode.loss != tree.nodes[t]->loss)
		{
			cout << "t=" << t << "; " << tempNode.loss << " v.s " << tree.nodes[t]->loss << endl;
		}
		PROCESS_ERROR(tempNode.loss == tree.nodes[t]->loss);
		PROCESS_ERROR(tempNode.base_weight == tree.nodes[t]->base_weight);
		PROCESS_ERROR(tempNode.predValue == tree.nodes[t]->predValue);
		PROCESS_ERROR(tempNode.rightChildId == tree.nodes[t]->rightChildId);
	}

	//copy the number of nodes in the tree to the GPU memory
	manager.MemcpyHostToDevice(&m_nNumofNode, snManager.m_pCurNumofNode, sizeof(int));

	CreateNewNode<<<1, 1>>>(snManager.m_pTreeNode, manager.m_pSplittableNode, snManager.m_pNewSplittableNode,
							manager.m_pSNIdToBuffId, manager.m_pBestSplitPoint,
							snManager.m_pParentId, snManager.m_pLeftChildId, snManager.m_pRightChildId,
							manager.m_pLChildStat, manager.m_pRChildStat, snManager.m_pNewNodeStat,
							snManager.m_pCurNumofNode, rt_eps, nNumofSplittableNode, bLastLevel);

	//cpu code, now for testing
	//for each splittable node, assign lchild and rchild ids
	map<int, pair<int, int> > mapPidCid;//(parent id, (lchildId, rchildId)).
	vector<TreeNode*> newSplittableNode;
	vector<nodeStat> newNodeStat;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = splittableNode[n]->nodeId;
//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = mapNodeIdToBufferPos[nid];
		map<int, int>::iterator itBufferPos = mapNodeIdToBufferPos.find(nid);
		assert(itBufferPos != mapNodeIdToBufferPos.end() && bufferPos == itBufferPos->second);
		PROCESS_ERROR(bufferPos < vBest.size());

		if(!(vBest[bufferPos].m_fGain <= rt_eps || bLastLevel == true))
		{
			int lchildId = m_nNumofNode;
			int rchildId = m_nNumofNode + 1;

			mapPidCid.insert(make_pair(nid, make_pair(lchildId, rchildId)));

			//push left and right child statistics into a vector
			PROCESS_ERROR(lchildStat[bufferPos].sum_hess > 0);
			PROCESS_ERROR(rchildStat[bufferPos].sum_hess > 0);
			newNodeStat.push_back(lchildStat[bufferPos]);
			newNodeStat.push_back(rchildStat[bufferPos]);

			//split into two nodes
			TreeNode *leftChild = new TreeNode[1];
			TreeNode *rightChild = new TreeNode[1];

			leftChild->nodeId = lchildId;
			leftChild->parentId = nid;
			leftChild->level = tree.nodes[nid]->level + 1;
			rightChild->nodeId = rchildId;
			rightChild->parentId = nid;
			rightChild->level = tree.nodes[nid]->level + 1;

			newSplittableNode.push_back(leftChild);
			newSplittableNode.push_back(rightChild);

			tree.nodes.push_back(leftChild);
			tree.nodes.push_back(rightChild);

			tree.nodes[nid]->leftChildId = leftChild->nodeId;
			tree.nodes[nid]->rightChildId = rightChild->nodeId;
			PROCESS_ERROR(vBest[bufferPos].m_nFeatureId >= 0);
			tree.nodes[nid]->featureId = vBest[bufferPos].m_nFeatureId;
			tree.nodes[nid]->fSplitValue = vBest[bufferPos].m_fSplitValue;


			m_nNumofNode += 2;
		}
	}

	//testing. Compare the new splittable nodes form GPU with those from CPU
	for(int n = 0; n < newSplittableNode.size(); n++)
	{
		TreeNode tempNode;
		manager.MemcpyDeviceToHost(snManager.m_pNewSplittableNode + n, &tempNode, sizeof(TreeNode) * 1);
		if(tempNode.nodeId != newSplittableNode[n]->nodeId)
		{
			cout << "n=" << n << "; " << tempNode.nodeId << " v.s " << newSplittableNode[n]->nodeId << endl;
		}
		PROCESS_ERROR(tempNode.nodeId == newSplittableNode[n]->nodeId);
		PROCESS_ERROR(tempNode.parentId == newSplittableNode[n]->parentId);
		PROCESS_ERROR(tempNode.level == newSplittableNode[n]->level);
	}

	//find all used unique feature ids
	manager.Memset(snManager.m_pFeaIdToBuffId, -1, sizeof(int) * snManager.m_maxNumofUsedFea);
	manager.Memset(snManager.m_pNumofUniqueFeaId, 0, sizeof(int));
	GetUniqueFid<<<1, 1>>>(snManager.m_pTreeNode, manager.m_pSplittableNode, nNumofSplittableNode,
							 snManager.m_pFeaIdToBuffId, snManager.m_pUniqueFeaIdVec, snManager.m_pNumofUniqueFeaId,
			 	 	 	 	 snManager.m_maxNumofUsedFea, LEAFNODE);

	//CPU code for getting all the used feature indices; now for testing.
	vector<int> vFid;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int fid = splittableNode[n]->featureId;
		int nid = splittableNode[n]->nodeId;
		if(fid == -1 && tree.nodes[nid]->rightChildId == LEAFNODE)
		{//leaf node should satisfy two conditions at this step
			continue;
		}
		PROCESS_ERROR(fid >= 0);
		vFid.push_back(fid);
	}
//	PrintVec(vFid);
	if(vFid.size() == 0)
		PROCESS_ERROR(nNumofSplittableNode == 1 || bLastLevel == true);
	PROCESS_ERROR(vFid.size() <= nNumofSplittableNode);

	//find unique used feature ids
	DataPreparator preparator;
	int *pFidHost = new int[vFid.size()];
	preparator.VecToArray(vFid, pFidHost);
	//push all the elements into a hash map
	int numofUniqueFid = 0;
	int *pUniqueFidHost = new int[vFid.size()];
	preparator.m_pUsedFIDMap = new int[snManager.m_maxNumofUsedFea];
	memset(preparator.m_pUsedFIDMap, -1, snManager.m_maxNumofUsedFea);
	for(int i = 0; i < vFid.size(); i++)
	{
		bool bIsNew = false;
		int hashValue = preparator.AssignHashValue(preparator.m_pUsedFIDMap, vFid[i], snManager.m_maxNumofUsedFea, bIsNew);
		if(bIsNew == true)
		{
			pUniqueFidHost[numofUniqueFid] = vFid[i];
			numofUniqueFid++;
		}
	}
	//comparing unique ids
	int *pUniqueIdFromDevice = new int[snManager.m_maxNumofUsedFea];
	int numofUniqueFromDevice = 0;
	manager.MemcpyDeviceToHost(snManager.m_pUniqueFeaIdVec, pUniqueIdFromDevice, sizeof(int) * snManager.m_maxNumofUsedFea);
	manager.MemcpyDeviceToHost(snManager.m_pNumofUniqueFeaId, &numofUniqueFromDevice, sizeof(int));
	PROCESS_ERROR(numofUniqueFromDevice == numofUniqueFid);
	for(int i = 0; i < numofUniqueFid; i++)
	{
		PROCESS_ERROR(pUniqueIdFromDevice[i] == pUniqueFidHost[i]);
	}

	delete[] pUniqueIdFromDevice;
	delete[] pFidHost;
	delete[] preparator.m_pUsedFIDMap;

	sort(vFid.begin(), vFid.end());
	vFid.resize(std::unique(vFid.begin(), vFid.end()) - vFid.begin());
	PROCESS_ERROR(vFid.size() <= nNumofSplittableNode);
	PROCESS_ERROR(vFid.size() == numofUniqueFid);
//	PrintVec(vFid);

	int testBufferPos = mapNodeIdToBufferPos[8];
	cout << "test buffer pos=" << testBufferPos << "; preMaxNodeId=" << preMaxNodeId << endl;

	//for each used feature to move instances to new nodes
	InsToNewNode<<<1, 1>>>(snManager.m_pTreeNode, manager.m_pdDFeaValue, manager.m_pDInsId,
						   	 manager.m_pFeaStartPos, manager.m_pDNumofKeyValue,
						   	 manager.m_pInsIdToNodeId, manager.m_pSNIdToBuffId, manager.m_pBestSplitPoint,
						   	 snManager.m_pUniqueFeaIdVec, snManager.m_pNumofUniqueFeaId,
							 snManager.m_pParentId, snManager.m_pLeftChildId, snManager.m_pRightChildId,
							 preMaxNodeId, manager.m_numofFea, manager.m_numofIns, LEAFNODE);

	//for each used feature to make decision
	for(int u = 0; u < numofUniqueFid; u++)
	{
		int ufid = pUniqueFidHost[u];
		PROCESS_ERROR(ufid < m_vvFeaInxPair.size() && ufid >= 0);

		//for each instance that has value on the feature
		vector<KeyValue> &featureKeyValues = m_vvFeaInxPair[ufid];
		int nNumofPair = featureKeyValues.size();
		for(int i = 0; i < nNumofPair; i++)
		{
			int insId = featureKeyValues[i].id;
			PROCESS_ERROR(insId < m_nodeIds.size());
			int nid = m_nodeIds[insId];

			if(nid < 0)//leaf node
				continue;

			PROCESS_ERROR(nid >= 0);
			int bufferPos = mapNodeIdToBufferPos[nid];
			map<int, int>::iterator itBufferPos = mapNodeIdToBufferPos.find(nid);
			assert(itBufferPos != mapNodeIdToBufferPos.end());
			int fid = vBest[bufferPos].m_nFeatureId;
			if(fid != ufid)//this feature is not the splitting feature for the instance.
				continue;

			map<int, pair<int, int> >::iterator it = mapPidCid.find(nid);

			if(it == mapPidCid.end())//node doesn't need to split (leaf node or new node)
			{
				if(tree.nodes[nid]->rightChildId != LEAFNODE)
				{
					PROCESS_ERROR(nid > preMaxNodeId);
					continue;
				}
				PROCESS_ERROR(tree.nodes[nid]->rightChildId == LEAFNODE);
				continue;
			}

			if(it != mapPidCid.end())
			{//internal node (needs to split)
				PROCESS_ERROR(it->second.second == it->second.first + 1);//right child id > than left child id

				double fPivot = vBest[bufferPos].m_fSplitValue;
				double fvalue = featureKeyValues[i].featureValue;
				if(fvalue >= fPivot)
				{
					m_nodeIds[insId] = it->second.second;//right child id
				}
				else
					m_nodeIds[insId] = it->second.first;//left child id
			}
		}
	}

	//testing. Compare ins id to node id
	int *insIdToNodeIdHost = new int[manager.m_numofIns];
	manager.MemcpyDeviceToHost(manager.m_pInsIdToNodeId, insIdToNodeIdHost, sizeof(int) * manager.m_numofIns);
	for(int i = 0; i < manager.m_numofIns; i++)
	{
		PROCESS_ERROR(insIdToNodeIdHost[i] == m_nodeIds[i]);
	}

	delete[] pUniqueFidHost;//for storing unique used feature ids

	//for those instances of unknown feature values.
	for(int i = 0; i < m_nodeIds.size(); i++)
	{
		int nid = m_nodeIds[i];
		if(nid == -1 || nid > preMaxNodeId)//processed node (i.e. leaf node or new node)
			continue;
		//newly constructed leaf node
		if(tree.nodes[nid]->rightChildId == LEAFNODE)
		{
			m_nodeIds[i] = -1;
		}
		else
		{
			map<int, pair<int, int> >::iterator it = mapPidCid.find(nid);
			m_nodeIds[i] = it->second.first;//by default the instance with unknown feature value going to left child

			PROCESS_ERROR(it != mapPidCid.end());
		}
	}

	mapNodeIdToBufferPos.clear();

	UpdateNodeStat(newSplittableNode, newNodeStat);

	splittableNode.clear();
	splittableNode = newSplittableNode;
}
