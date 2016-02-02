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
#include <fstream>
#include <vector>
#include "RegTree.h"
#include "DatasetInfo.h"
#include "TreeNode.h"

using std::string;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;

class Trainer
{
private:
	struct gdpair {
	  /*! \brief gradient statistics */
	  double grad;
	  /*! \brief second order gradient statistics */
	  double hess;
	  gdpair(void) {grad = 0; hess = 0;}
	  gdpair(double grad, double hess) : grad(grad), hess(hess) {}
	};

public:
	int m_nMaxNumofTree;
	int m_nMaxDepth;
	int m_nNumofSplittableNode;
	vector<vector<double> > m_vvInstance;
	vector<double> m_vTrueValue;
	vector<double> m_vPredBuffer;
	vector<gdpair> m_vGDPair;
	DataInfo data;
	double m_labda;//the weight of the cost of complexity of a tree
	double m_gamma;//the weight of the cost of the number of trees

private:
	int m_nNumofNode;

private:
	struct SplitPoint{
		double m_fGain;
		double m_fSplitValue;
		int m_nFeatureId;

		SplitPoint()
		{
			m_fGain = 0;
			m_fSplitValue = 0;
			m_nFeatureId = -1;
		}

		void UpdateSplitPoint(double fGain, double fSplitValue, int nFeatureId)
		{
			if(fGain > m_fGain || (fGain == m_fGain && nFeatureId == m_nFeatureId))//second condition is for updating to a new split value
			{
				m_fGain = fGain;
				m_fSplitValue = fSplitValue;
				m_nFeatureId = nFeatureId;
			}
		}
	};

public:
	void InitTrainer(int nNumofTree, int nMaxDepth, double fLabda, double fGamma);
	void TrainGBDT(vector<vector<double> > &v_vInstance, vector<double> &v_fLabel, vector<RegTree> &v_Tree);
	void SaveModel(string fileName, const vector<RegTree> &v_Tree);

protected:
	void InitTree(RegTree &tree);
	void GrowTree(RegTree &tree);

private:
	void ComputeGD(vector<double> &v_fPredValue);
	void CreateNode();
	double ComputeGain(double fSplitValue, int featureId, int dataStartId, int dataEndId);
	void ComputeWeight(TreeNode &node);
	void SplitNode(TreeNode *node, vector<TreeNode*> &newSplittableNode, SplitPoint &sp, RegTree &tree);
	int Partition(SplitPoint &sp, int startId, int endId);


//for debugging
	void PrintTree(const RegTree &tree);
	void PrintPrediction(const vector<double> &vPred);
	void CheckPartition(int startId, int endId, int middle, SplitPoint &sp);

	template <class T> void Swap(T& x, T& y) { T t=x; x=y; y=t; }
};



#endif /* TRAINER_H_ */
