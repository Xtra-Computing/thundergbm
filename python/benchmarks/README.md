
## Introduction
    
   This is some python scripts that can help you reproduce our experimental results. We would like to thank the author of [GBM-Benchmarks]([https://github.com/RAMitchell/GBM-Benchmarks](https://github.com/RAMitchell/GBM-Benchmarks)) which our scripts are built on top of.
   
## Requirements
 _python3.x_&nbsp;&nbsp;&nbsp;&nbsp; _numpy_ &nbsp;&nbsp;&nbsp;&nbsp;_sklearn_&nbsp;&nbsp;&nbsp;&nbsp;_xgboost_ &nbsp;&nbsp;&nbsp;&nbsp; _lightgbm_ &nbsp;&nbsp;&nbsp;&nbsp; _catboost_&nbsp;&nbsp;&nbsp;&nbsp; _thundergbm_

## How to run 

You can use command `python3 experiments.py [model_name] [device_type] [dataset_name]` to run the scripts. And the candidate values of each parameter are as follows:

 - model_name
	 - xgboost
	 - lightgbm
	 - catboost
	 - thundergbm
 - device_type
	 - cpu
	 - gpu
 - dataset_name
	 - news20
	 - higgs
	 - log1p
	 - cifar
	 
## Files descriptions
 - model
	 - base_model.py
	 - datasets. py
	 - catboost_model.py
	 - lightgb_model.py
	 - xgboost_model.py
	 - thundergbm_model.py
 - utils
	 - data_utils,py
	 - file_utils.py
 - experiments. py
 - convert_dataset_plk.py

Floder **_model_** contains the model file of each libraries which inherit from `BaseModel` in `base_model.py.` Floder **_utils_** contains a few tools including `data heleper` and  `file I/O helper`. `convert_dataset_plk.py` is used for converting normal `libsvm` file to Python pickle file. This is because the datasets we used for experiments sometime have large size which lead to time-consuming data loading step. By using pickle, we can sharply reduce consuming time in data loading step. `experiments.py` is the main entrance of our scripts.

## How to add more datasets

As the optional datasets of our scripts are limited, you can add the datasets you want. You can achieve this by modifying the file `utils/data_utils.py`. There are some dataset template in that script which may help you add your own dataset easily.
