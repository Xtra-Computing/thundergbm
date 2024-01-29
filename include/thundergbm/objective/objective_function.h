//
// Created by ss on 19-1-1.
//

#ifndef THUNDERGBM_OBJECTIVE_FUNCTION_H
#define THUNDERGBM_OBJECTIVE_FUNCTION_H

#include <thundergbm/syncarray.h>
#include <thundergbm/dataset.h>

class ObjectiveFunction {
public:
    //todo different target type
    virtual void
    get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) = 0;
    virtual void
    predict_transform(SyncArray<float_type> &y){};

    //init base score
    //SyncArray<GHPair> &gh_pair for tmporary storage    
    virtual float init_base_score(const SyncArray<float_type> &y,SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair){
        LOG(INFO)<<"not implement base score function!!!!!!";
        return 0;
    }

    virtual void configure(GBMParam param, const DataSet &dataset) = 0;
    virtual string default_metric_name() = 0;

    static ObjectiveFunction* create(string name);

    //a file containing the number of instances per query; similar to XGBoost
    static bool need_load_group_file(string name);
    static bool need_group_label(string name);
    virtual ~ObjectiveFunction() = default;
};

#endif //THUNDERGBM_OBJECTIVE_FUNCTION_H
