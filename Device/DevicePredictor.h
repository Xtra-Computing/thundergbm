/*
 * DevicePrediction.h
 *
 *  Created on: 23 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEPREDICTION_H_
#define DEVICEPREDICTION_H_


#include <vector>
#include "../Host/Tree/RegTree.h"
#include "../Host/KeyValue.h"
#include "../DeviceHost/BaseClasses/BasePredictor.h"

using std::vector;

class DevicePredictor: public BasePredictor
{
public:
	virtual void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<float_point> &v_fPredValue);
	static void GetUsedFeature(vector<int> &v_usedFeaSortedId, int *&pHashUsedFea, int *&pSortedUsedFea);
	static void GetTreeInfo(TreeNode *&pTree, int &numofNodeOfTheTree, int treeId);
	static void FillDenseIns(int insId, int numofUsedFea);
};

#endif /* DEVICEPREDICTION_H_ */
