# Examples #

This folder contains examples of PGBM. Our examples are aimed to illustrate the following:
* How to train PGBM: on CPU, GPU and multi-GPU.
* How to train PGBM using a validation loop.
* How PGBM compares to other methods such as NGBoost and LightGBM.
* How the choice of output distribution can be optimized after training.
* How PGBM can be used in a hierarchical time series forecasting problem.

Example index:
1. Probabilistic regression for the boston housing dataset, training on CPU.
2. Probabilistic regression for the boston housing dataset, training on single GPU.
3. Probabilistic regression for the boston housing dataset, training on multi-GPU.
4. Probabilistic regression for the boston housing dataset, using a validation loop.
5. Probabilistic regression for the boston housing dataset, comparison to NGBoost.
6. Training time comparison between PGBM (trained on GPU) and LightGBM (CPU) for a range of datasets.
7. How to use a validation loop to optimize choice of distribution and tree correlation parameter.

# Hyperparameters #
PGBM employs the following set of hyperparameters:
* `min_split_gain` the minimum gain for a node to split when building a tree. default=0.0
* ``:

# GPU training #
For training on GPU, it is required to set the following hyperparameters:
```
params['device'] = 'gpu'
params['output_device'] = 'gpu'
```
By default PGBM will train on the GPU at the first index (0) and return the results at that device. This corresponds to the following parameter:
```
params['gpu_device_ids'] = (0,)
```
If one would like to perform multi-gpu training and use different device ids, just list the device ids:
```
params['gpu_device_ids'] = (1, 2, 3)
```
In the latter case, GPUs corresponding to id 1 to 3 will be used, where device with id 1 will be used to return the results to. 