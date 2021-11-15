# Examples #

We provide an extensive [set of examples on our Github page.](https://github.com/elephaint/pgbm/tree/main/examples). An overview is given below.

## Torch backend ##

### Python API ###
* [Example 1](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_housing_cpu.py): How to train PGBM on CPU
* [Example 2](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_housing_gpu.py): How to train PGBM on GPU
* [Example 4](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example04_housing_validation.py): How to train PGBM using a validation loop.
* [Example 5](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example05_housing_vsngboost.py): How PGBM compares to NGBoost
* [Example 6](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example06_trainingtimevslightgbm.py): How PGBM compares to LightGBM.
* [Example 7](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example07_optimizeddistribution.py): How the choice of output distribution can be optimized after training.
* [Example 8](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example08_housing_autodiff.py): How to use autodifferentiation for loss functions where no analytical gradient or hessian is provided.
* [Example 9](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example09_housing_featimportance.py): How to plot the feature importance of a learner after training.
* [Example 10](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example10_covidhospitaladmissions.py): How we employed PGBM to forecast Covid-19 daily hospital admissions in the Netherlands.
* [Example 11](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example11_housing_saveandload.py): How to save and load a PGBM model. Train and predict using different devices (CPU or GPU).
* [Example 12](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example12_housing_checkpointing.py): How to continue training and using checkpoints to save model state during training.

### Scikit-learn API ###
* [Example 15](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example15_monotone_constraints.py): How to use monotone constraints to improve model performance.

## Numba backend ##

### Python API ###
* [Example 1](https://github.com/elephaint/pgbm/blob/main/examples/numba/example01_housing_cpu.py): How to train PGBM
* [Example 4](https://github.com/elephaint/pgbm/blob/main/examples/numba/example04_housing_validation.py): How to train PGBM using a validation loop.
* [Example 5](https://github.com/elephaint/pgbm/blob/main/examples/numba/example05_housing_vsngboost.py): How PGBM compares to NGBoost
* [Example 6](https://github.com/elephaint/pgbm/blob/main/examples/numba/example06_trainingtimevslightgbm.py): How PGBM compares to LightGBM.
* [Example 7](https://github.com/elephaint/pgbm/blob/main/examples/numba/example07_optimizeddistribution.py): How the choice of output distribution can be optimized after training.
* [Example 9](https://github.com/elephaint/pgbm/blob/main/examples/numba/example09_housing_featimportance.py): How to plot the feature importance of a learner after training.
* [Example 11](https://github.com/elephaint/pgbm/blob/main/examples/numba/example11_housing_saveandload.py): How to save and load a PGBM model. 
* [Example 12](https://github.com/elephaint/pgbm/blob/main/examples/numba/example12_housing_checkpointing.py): How to continue training and using checkpoints to save model state during training.

### Scikit-learn API ###
* [Example 13](https://github.com/elephaint/pgbm/blob/main/examples/numba/example15_monotone_constraints.py): How to use monotone constraints to improve model performance.

## Torch-distributed backend ##
* [Example 13]:(https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/example13_housing_dist.py) How to train the housing dataset using our distributed backend.
* [Example 14]:(https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/example14_higgs_dist.py) How to train the Higgs dataset using our distributed backend.


