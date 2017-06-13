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
#include "../SharedUtility/KeyValue.h"
#include "../DeviceHost/BaseClasses/BasePredictor.h"

using std::vector;

class DevicePredictor: public BasePredictor
{
public:
	virtual void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<real> &v_fPredValue, void *pStream, int bagId);
	static void GetUsedFeature(vector<int> &v_usedFeaSortedId, int *&pHashUsedFea, int *&pSortedUsedFea, void *pStream, int bagId);
	static void GetTreeInfo(TreeNode *&pTree, int &numofNodeOfTheTree, int treeId, void *pStream, int bagId);
};

#endif /* DEVICEPREDICTION_H_ */
