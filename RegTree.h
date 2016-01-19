/*
 * RegTree.h
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef REGTREE_H_
#define REGTREE_H_

#include <vector>
#include "TreeNode.h"

using std::vector;

class RegTree
{
public:
	vector<TreeNode> nodes;
	inline TreeNode &operator[](int nid)
	{
	    return nodes[nid];
	}

 public:
  /*!
   * \brief dense feature vector that can be taken by RegTree
   * to do tranverse efficiently
   * and can be construct from sparse feature vector
   */
  struct FVec {
    /*!
     * \brief a union value of value and flag
     * when flag == -1, this indicate the value is missing
     */
    union Entry{
      float fvalue;
      int flag;
    };
    std::vector<Entry> data;
    /*! \brief intialize the vector with size vector */
    inline void Init(size_t size) {
      Entry e; e.flag = -1;
      data.resize(size);
      std::fill(data.begin(), data.end(), e);
    }
    /*! \brief fill the vector with sparse vector */
//    inline void Fill(const RowBatch::Inst &inst) {
//      for (bst_uint i = 0; i < inst.length; ++i) {
//        if (inst[i].index >= data.size()) continue;
//        data[inst[i].index].fvalue = inst[i].fvalue;
//      }
//    }
//    /*! \brief drop the trace after fill, must be called after fill */
//    inline void Drop(const RowBatch::Inst &inst) {
//      for (bst_uint i = 0; i < inst.length; ++i) {
//        if (inst[i].index >= data.size()) continue;
//        data[inst[i].index].flag = -1;
//      }
//    }
    /*! \brief get ith value */
    inline float fvalue(size_t i) const {
      return data[i].fvalue;
    }
    /*! \brief check whether i-th entry is missing */
    inline bool is_missing(size_t i) const {
      return data[i].flag == -1;
    }
  };

  int GetLeafIndex(vector<float> &ins);

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
