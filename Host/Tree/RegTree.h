/*
 * RegTree.h
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef REGTREE_H_
#define REGTREE_H_

#include <map>
#include <vector>
#include "../../DeviceHost/TreeNode.h"
#include "../../DeviceHost/DefineConst.h"

using std::vector;
using std::map;

class RegTree
{
public:
	vector<TreeNode*> nodes;

public:
	RegTree(){nodes.clear();}

	inline TreeNode* &operator[](int nid)
	{
	    return nodes[nid];
	}

 public:
  int GetLeafIndex(vector<real> &ins);

  int GetLeafIdSparseInstance(vector<real> &ins, map<int, int> &fidToDensePos);

//  /*!
//   * \brief get the prediction of regression tree, only accepts dense feature vector
//   * \param feats dense feature vector, if the feature is missing the field is set to NaN
//   * \param root_gid starting root index of the instance
//   * \return the leaf index of the given feature
//   */
//  inline float Predict(const FVec &feat, unsigned root_id = 0) const {
//    int pid = this->GetLeafIndex(feat, root_id);
//    return (*this)[pid].leaf_value();
//  }
//  /*! \brief get next position of the tree given current pid */
//  inline int GetNext(int pid, float fvalue, bool is_unknown) const {
//    float split_value = (*this)[pid].split_cond();
//    if (is_unknown) {
//      return (*this)[pid].cdefault();
//    } else {
//      if (fvalue < split_value) {
//        return (*this)[pid].cleft();
//      } else {
//        return (*this)[pid].cright();
//      }
//    }
//  }
};



#endif /* REGTREE_H_ */
