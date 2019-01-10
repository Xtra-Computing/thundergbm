ThunderGBM Parameters
=====================
This page is for parameter specification in ThunderGBM. The parameters used in ThunderGBM are identical to XGBoost (except some newly introduced parameters), so existing XGBoost users can easily get used to ThunderGBM.

### Key Parameters
* ``verbosity`` [default=1]

    - Printing information: 0 for silence and 1 for printing more infomation.
    
* ``max_depth`` [default=6]
 
    - The maximum depth of the decision trees. Shallow trees tend to have better generality, and deep trees are more likely to overfit the training data.

* ``num_round`` [default=40]

    - The number of training iterations. ``num_round`` equals to the number of trees in GBDTs.
    
* ``n_gpus`` [default=1]

    - The number of GPUs to be used in the training.
    
* ``data``
    
    - The path to the training data set
    
* ``max_bin`` [default=255]
    
    - The maximum number of bins in a histogram.
    
* ``colsample`` [default=1]

    - The sampling ratio of subsampling columns (i.e., features)
    
* ``bagging`` [default=0]

    - This option is for training random forests. Setting it to 1 to perform bagging.
    
* ``num_parallel_tree`` [default=1]

    - This option is used for random forests to specify how many trees per iteration.
    
* ``eta`` [default=1, alias: ``learning_rate``]

    - valid domain: [0,1]. This option is to set the weight of newly trained tree. Use ``eta < 1`` to mitigate overfitting.
    
* ``objective`` [default="reg:linear"]
    
    - valid options include ``reg:linear`` and ``reg:logistic`` for regression and binary classification; ``multi:softprob`` and ``multi:softmax`` for multi-class classification.
    - ``multi:softprob`` outputs probability for each class, and ``multi:softmax`` outputs the label only.
    
* ``num_class`` [default=1]

    - set the number of classes in the multi-class classification. This option is not compulsory.
    
* ``min_child_weight`` [default=1]

    - The minimum sum of instance weight (measured by the second order derivative) needed in a child node.

* ``lambda`` [default=1, alias: ``reg_lambda``]

    - L2 regularization term on weights.
    
* ``gamma`` [default=1, alias: ``min_split_loss``]

    - The minimum loss reduction required to make a further split on a leaf node of the tree. ``gamma`` is used in the pruning stage.

* ``model_out`` [default="tgbm.model"]
    
    - The file name of the output model. This option is used in training.
    
* ``model_in`` [default="tgbm.model"]

    - The file name of the input model. This option is used in prediction.