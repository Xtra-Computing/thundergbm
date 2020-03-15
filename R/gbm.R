# Created by: Qinbin
# Created on: 3/15/20

check_location <- function(){
  if(Sys.info()['sysname'] == 'Windows'){
    if(!file.exists("../build/bin/Debug/thundergbm.dll")){
      stop("Please build the library first (or check you called this while your workspace is set to the thundergbm/R/ directory)!")
    }
    dyn.load("../build/bin/Debug/thundergbm.dll")
  } else if(Sys.info()['sysname'] == 'Linux'){
    if(!file.exists("../build/lib/libthundergbm.so")){
      stop("Please build the library first (or check you called this while your workspace is set to the thundergbm/R/ directory)!")
    }
    dyn.load("../build/lib/libthundergbm.so")
  } else if(Sys.info()['sysname'] == 'Darwin'){
    if(!file.exists("../build/lib/libthundergbm.dylib")){
      stop("Please build the library first (or check you called this while your workspace is set to the thundergbm/R/ directory)!")
    }
    dyn.load("../build/lib/libthundergbm.dylib")
  } else{
    stop("OS not supported!")
  }
}
check_location() # Run this when the file is sourced

gbm_train_R <-
  function(
    depth = 6, n_trees = 40, n_gpus = 1, verbose =  1,
    profiling = 0, data = 'None', max_num_bin = 255, column_sampling_rate = 1,
    bagging = 0, n_parallel_trees = 1, learning_rate = 1, objective = 'reg:linear',
    num_class = 1, min_child_weight = 1, lambda_tgbm = 1, gamma = 1,
    tree_method = 'auto', model_out = 'tgbm.model'
  )
    {
    check_location()
    if(!file.exists(data)){stop("The file containing the training dataset provided as an argument in 'data' does not exist")}
    res <- .C("train_R", as.integer(depth), as.integer(n_trees), as.integer(n_gpus), as.integer(verbose),
              as.integer(profiling), as.character(data), as.integer(max_num_bin), as.double(column_sampling_rate),
              as.integer(bagging), as.integer(n_parallel_trees), as.double(learning_rate), as.character(objective),
              as.integer(num_class), as.integer(min_child_weight), as.double(lambda_tgbm), as.double(gamma),
              as.character(tree_method), as.character(model_out))
  }

gbm_predict_R <-
  function(
    test_data = 'None', model_in = 'tgbm.model', verbose = 1
  )
    {
    check_location()
    if(!file.exists(test_data)){stop("The file containing the training dataset provided as an argument in 'data' does not exist")}
    if(!file.exists(model_in)){stop("The file containing the model provided as an argument in 'model_in' does not exist")}
    res <- .C("predict_R", as.character(test_data), as.character(model_in), as.integer(verbose))
  }

