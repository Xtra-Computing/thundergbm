/*
 * Trainer.h
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *      @brief: GBDT trainer
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include <iostream>
#include <vector>
#include "RegTree.h"
#include "DatasetInfo.h"
#include "TreeNode.h"

using std::string;
using std::vector;

class Trainer
{
private:
	struct gdpair {
	  /*! \brief gradient statistics */
	  float grad;
	  /*! \brief second order gradient statistics */
	  float hess;
	  gdpair(void) {grad = 0; hess = 0;}
	  gdpair(float grad, float hess) : grad(grad), hess(hess) {}
	};

public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;
	int m_nNumofSplittableNode;
	vector<vector<float> > m_vvInstance;
	vector<float> m_vTrueValue;
	vector<float> m_vPredBuffer;
	vector<gdpair> m_vGDPair;
	DataInfo data;
	float m_labda;//the weight of the cost of complexity of a tree
	float m_gamma;//the weight of the cost of the number of trees

private:
	int m_nNumofNode;

private:
	struct SplitPoint{
		float m_fGain;
		float m_fSplitValue;
		int m_nFeatureId;

		SplitPoint()
		{
			m_fGain = 0;
			m_fSplitValue = 0;
			m_nFeatureId = -1;
		}

		void UpdateSplitPoint(float fGain, float fSplitValue, int nFeatureId)
		{
			if(fGain > m_fGain)
			{
				m_fGain = fGain;
				m_fSplitValue = fSplitValue;
				m_nFeatureId = nFeatureId;
			}
		}
	};

public:
	void InitTrainer(int nNumofTree, int nMaxDepth, float fLabda, float fGamma);
	void TrainGBDT(vector<vector<float> > &v_vInstance, vector<float> &v_fLabel, vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);

private:
	void ComputeGD(vector<float> &v_fPredValue);
	void CreateNode();
	float ComputeGain(float fSplitValue, int featureId, int dataStartId, int dataEndId);
	void ComputeWeight(TreeNode &node);
	void SplitNode(TreeNode &node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree);
	int Partition(SplitPoint &sp, int startId, int endId);
};



#endif /* TRAINER_H_ */
