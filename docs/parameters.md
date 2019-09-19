ThunderGBM Parameters
=====================
This page is for parameter specification in ThunderGBM. The parameters used in ThunderGBM are identical to XGBoost (except some newly introduced parameters), so existing XGBoost users can easily get used to ThunderGBM.

### Key arameters for both *python* and *c++* command line
* ``verbose`` [default=1]

    - Printing information: 0 for silence and 1 for printing more infomation.
    
* ``depth`` [default=6]
 
    - The maximum depth of the decision trees. Shallow trees tend to have better generality, and deep trees are more likely to overfit the training data.

* ``n_trees`` [default=40]

    - The number of training iterations. ``n_trees`` equals to the number of trees in GBDTs.
    
* ``n_gpus`` [default=1]

    - The number of GPUs to be used in the training.
    
* ``max_num_bin`` [default=255]
    
    - The maximum number of bins in a histogram.
    
* ``column_sampling_rate`` [default=1]

    - The sampling ratio of subsampling columns (i.e., features)
    
* ``bagging`` [default=0]

    - This option is for training random forests. Setting it to 1 to perform bagging.
    
* ``n_parallel_trees`` [default=1]

    - This option is used for random forests to specify how many trees per iteration.
    
* ``learning_rate`` [default=1, alias(only for c++): ``eta``]

    - valid domain: [0,1]. This option is to set the weight of newly trained tree. Use ``eta < 1`` to mitigate overfitting.
    
* ``objective`` [default="reg:linear"]
    
    - valid options include ``reg:linear``, ``reg:logistic``, ``multi:softprob``,  ``multi:softmax``, ``rank:pairwise`` and ``rank:ndcg``.
    - ``reg:linear`` is for regression, ``reg:logistic`` and ``binary:logistic`` are for binary classification.
    - ``multi:softprob`` and ``multi:softmax`` are for multi-class classification. ``multi:softprob`` outputs probability for each class, and ``multi:softmax`` outputs the label only.
    - ``rank:pairwise`` and ``rank:ndcg`` are for ranking problems.
    
* ``num_class`` [default=1]
    - set the number of classes in the multi-class classification. This option is not compulsory.
 
* ``min_child_weight`` [default=1]

    - The minimum sum of instance weight (measured by the second order derivative) needed in a child node.

* ``lambda_tgbm`` [default=1, alias(only for c++): ``lambda`` or ``reg_lambda``]

    - L2 regularization term on weights.
    
* ``gamma`` [default=1, alias(only for c++): ``min_split_loss``]

    - The minimum loss reduction required to make a further split on a leaf node of the tree. ``gamma`` is used in the pruning stage.
* ``tree_method`` [default="auto"]

    - "auto": select the approach of finding best splits using the builtin heuristics.
    
    - "exact": find the best split using enumeration on all the possible feature values.
    
    - "hist": find the best split using histogram based approach.

### Parameters only for *c++* command line:
* ``data`` [default="../dataset/test_dataset.txt"]

    - The path to the training data set

* ``model_out`` [default="tgbm.model"]
    
    - The file name of the output model. This option is used in training.
    
* ``model_in`` [default="tgbm.model"]

    - The file name of the input model. This option is used in prediction.
