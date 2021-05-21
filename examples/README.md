# Examples #

This folder contains examples of PGBM. Our examples are aimed to illustrate the following:
* Examples 1-3: How to train PGBM: on CPU, GPU and multi-GPU.
* Example 4: How to train PGBM using a validation loop.
* Examples 5-6: How PGBM compares to other methods such as NGBoost and LightGBM.
* Example 7: How the choice of output distribution can be optimized after training.
* Example 8: How PGBM can be used in a hierarchical time series forecasting problem.

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
When training on GPU, PGBM will select the GPU at the first index (0) by default and return the results at that device. This corresponds to the following parameter:
```
params['gpu_device_ids'] = (0,)
```
If one would like to perform multi-gpu training and use different device ids, just list the device ids:
```
params['gpu_device_ids'] = (1, 2, 3)
```
In the latter case, GPUs corresponding to id 1 to 3 will be used, where device with id 1 will be used to return the results to. 