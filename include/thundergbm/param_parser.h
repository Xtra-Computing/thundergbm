//
// Created by zeyi on 1/10/19.
//

#ifndef THUNDERGBM_PARAM_PARSER_H
#define THUNDERGBM_PARAM_PARSER_H

#include <fstream>
#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>

class ParamParser{
public:
    void parse_param(GBMParam &model_param, int argc, char **argv);
};

#endif //THUNDERGBM_PARAM_PARSER_H
