/*
 * dtMemManager.h
 *
 *  Created on: 25 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DTMEMMANAGER_H_
#define DTMEMMANAGER_H_

#include "../../SharedUtility/DataType.h"
#include "../../DeviceHost/TreeNode.h"

class DTGPUMemManager
{
public:
	static TreeNode *m_pAllTree;
	static int *m_pNumofNodeEachTree;
	static int *m_pStartPosOfEachTree;
	static int m_numofTree;
	static int m_numofTreeLearnt;
	static int m_maxTreeDepth;

public:
	void allocMemForTrees(int numofTree, int maxNumofNodePerTree, int maxDepth);
};



#endif /* DTMEMMANAGER_H_ */
