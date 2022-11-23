# Examples #

We provide an extensive [set of examples on our Github page](https://github.com/elephaint/pgbm/tree/main/examples). An overview is given below.

## Torch backend ##
* [Example 1](https://github.com/elephaint/pgbm/blob/main/examples/torch/example01_housing_cpu.py): How to train PGBM on CPU
* [Example 1a](https://github.com/elephaint/pgbm/blob/main/examples/torch/example01_housing_cpu.ipynb): How to train PGBM on CPU (Jupyter Notebook)
* [Example 2](https://github.com/elephaint/pgbm/blob/main/examples/torch/example02_housing_gpu.py): How to train PGBM on GPU
* [Example 4](https://github.com/elephaint/pgbm/blob/main/examples/torch/example04_housing_validation.py): How to train PGBM using a validation loop.
* [Example 5](https://github.com/elephaint/pgbm/blob/main/examples/torch/example05_housing_vsngboost.py): How PGBM compares to NGBoost
* [Example 6](https://github.com/elephaint/pgbm/blob/main/examples/torch/example06_trainingtimevslightgbm.py): How PGBM training time compares to LightGBM.
* [Example 7](https://github.com/elephaint/pgbm/blob/main/examples/torch/example07_optimizeddistribution.py): How the choice of output distribution can be optimized after training.
* [Example 8](https://github.com/elephaint/pgbm/blob/main/examples/torch/example08_housing_autodiff.py): How to use autodifferentiation for loss functions where no analytical gradient or hessian is provided.
* [Example 9](https://github.com/elephaint/pgbm/blob/main/examples/torch/example09_housing_featimportance.py): How to plot the feature importance of a learner after training using partial dependence plots.
* [Example 9a](https://github.com/elephaint/pgbm/blob/main/examples/torch/example09a_housing_featimportance_shaps.py): How to plot the feature importance of a learner after training using Shapley values.
* [Example 10](https://github.com/elephaint/pgbm/blob/main/examples/torch/example10_covidhospitaladmissions.py): How we employed PGBM to forecast Covid-19 daily hospital admissions in the Netherlands.
* [Example 11](https://github.com/elephaint/pgbm/blob/main/examples/torch/example11_housing_saveandload.py): How to save and load a PGBM model. Train and predict using different devices (CPU or GPU).
* [Example 12](https://github.com/elephaint/pgbm/blob/main/examples/torch/example12_housing_checkpointing.py): How to continue training and using checkpoints to save model state during training.
* [Example 15](https://github.com/elephaint/pgbm/blob/main/examples/torch/example15_monotone_constraints.py): How to use monotone constraints to improve model performance.

## Scikit-learn backend ##
* [Example 1](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example01_housing_cpu.py): How to train PGBM
* [Example 4](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example04_housing_validation.py): How to train PGBM using a validation loop.
* [Example 5](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example05_housing_vsngboost.py): How PGBM compares to NGBoost
* [Example 6](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example06_trainingtimevslightgbm.py): How PGBM compares to LightGBM.
* [Example 7](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example07_hyperparamoptimization.py): How parameters can be optimized using GridSearchCV.
* [Example 9](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example09_housing_featimportance.py): How to plot the feature importance of a learner after training using Shapley values.
* [Example 11](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example11_housing_saveandload.py): How to save and load a PGBM model. 
* [Example 12](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example12_housing_checkpointing.py): How to continue training after saving a model.
* [Example 13](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example13_monotone_constraints.py): How to use monotone constraints to improve model performance.
* [Example 14](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example14_probregression.py): How HistGradientBoostingRegressor with PGBM fares against quantile regression methods.

## Torch-distributed backend ##
* [Example 13](https://github.com/elephaint/pgbm/blob/main/examples/torch_dist/example13_housing_dist.py): How to train the housing dataset using our distributed backend.
* [Example 14](https://github.com/elephaint/pgbm/blob/main/examples/torch_dist/example14_higgs_dist.py): How to train the Higgs dataset using our distributed backend.


