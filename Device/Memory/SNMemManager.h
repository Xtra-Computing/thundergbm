/*
 * SplitNodeGPUMemManager.h
 *
 *  Created on: 16/05/2016
 *      Author: zeyi
 */

#ifndef SPLITNODEGPUMEMMANAGER_H_
#define SPLITNODEGPUMEMMANAGER_H_

#include "../../DeviceHost/TreeNode.h"
#include "../../DeviceHost/NodeStat.h"

class SNGPUManager
{
public:
	//memory for the tree
	static TreeNode *m_pTreeNode;
	static int m_maxNumofNode;

	//memory for parent node to children ids
	static int *m_pParentId;
	static int *m_pLeftChildId, *m_pRightChildId;

	//memory for new node statistics
	static nodeStat *m_pNewNodeStat;
	static TreeNode *m_pNewSplittableNode;

	//memory for current number of nodes and number of new nodes
	static int *m_pCurNumofNode_d;
	static int *m_pNumofNewNode;

	//memory for used features in the current splittable nodes
	static int m_maxNumofUsedFea;
	static int *m_pFeaIdToBuffId;
	static int *m_pUniqueFeaIdVec;
	static int *m_pNumofUniqueFeaId;

	//host memory for reset purposes
	static TreeNode *m_pTreeNodeHost;

public:
	void allocMemForTree(int maxNumofNode);
	void allocMemForParenChildIdMapping(int maxNumofSplittable);
	void allocMemForNewNode(int maxNumofSplittable);
	void allocMemForUsedFea(int nMaxNumofUsedFeature);

	void resetForNextTree();
};


#endif /* SPLITNODEGPUMEMMANAGER_H_ */
