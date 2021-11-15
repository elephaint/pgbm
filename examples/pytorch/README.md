# Examples #

This folder contains examples of PGBM. The examples illustrate the following:
* Example 1: How to train PGBM on CPU
* Example 2: How to train PGBM on GPU
* Example 4: How to train PGBM using a validation loop.
* Examples 5-6: How PGBM compares to other methods such as NGBoost and LightGBM.
* Example 7: How the choice of output distribution can be optimized after training.
* Example 8: How to use autodifferentiation for loss functions where no analytical gradient or hessian is provided.
* Example 9: How to plot the feature importance of a learner after training.
* Example 10: How we employed PGBM to forecast Covid-19 daily hospital admissions in the Netherlands.
* Example 11: How to save and load a PGBM model. Train and predict using different devices (CPU or GPU).
* Example 12: How to continue training and using checkpoints to save model state during training.
* Example 15: How to use monotone constraints to improve model performance.

Note: to use the `higgs` dataset in any of the examples, download [here](https://archive.ics.uci.edu/ml/datasets/HIGGS), unpack and save `HIGGS.csv` to your local working directory.