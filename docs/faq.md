Frequently Asked Questions (FAQs)
======
This page is dedicated to summarizing some frequently asked questions about ThunderGBM.

## FAQs of users

* **What is the data format of the input file?** 
  ThunderGBM uses the LibSVM format. You can also use ThunderGBM using the scikit-learn interface.

* **Can ThunderGBM run on CPUs?** 
  No. ThunderGBM is specifically optimized for GBDTs and Random Forests on GPUs.
  
* **Can ThunderGBM run on multiple GPUs?**
  Yes. You can use the ``n_gpus`` options to specify how many GPUs you want to use. Please refer to [Parameters](parameters.md) for more information.