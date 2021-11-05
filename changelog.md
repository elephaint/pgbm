## v1.4 ##
* Pytorch version complete code rewrite improving speed by up to 3x on GPU. 
* Replaced boston_housing by california housing as key example due to ethical concerns regarding its features (see here: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html).
* PyTorch: distributed version now separate of the vanilla version; this improves speed of the vanilla version. (Hopefully temporary solution until TorchScript will support distributed functions).
* Removed experimental TPU support for now.
* Parameters are now attributes of the learner instead of part of a dictionary `param`.
* Renamed the regularization parameter `lambda` to `reg_lambda` to avoid confusion with Python's `lambda` function.
* Rewrote splitting procedure on all versions, removing bugs observed in hyperparameter tuning.

## v1.3 ##
* Added `monotone_constraints` as parameter to initialization of `PGBMRegressor` rather than as part of `fit`.
* Speed improvements of both Numba and PyTorch version.

## v1.2 ##
* Fixed a bug in `monotone_constraints` calculation.
* Added a sklearn wrapper for both backends - `PGBMRegressor` is now available as a sklearn estimator.
* Renamed `levels_train` attribute in `train` function to `sample_weight` and `levels_valid` to `eval_sample_weight`, such that it is easier to understand what these parameters to.
* Added `sample_weight` and `eval_sample_weight` to Numba backend.
* Added stability constant epsilon to variance calculation to prevent division by zero (mostly happened on Numba backend, due to its higher precision in case there is a zero gradient mean in a leaf)
* Fixed bug that caused error for `min_data_in_leaf`, was caused by too low precision (BFloat16 of split count array in CUDA kernel). Set default `min_data_in_leaf` back to `2`.

## v1.1 ##
* Fixed a bug in bin calculation of Torch version that caused incorrect results on most outer quantiles of feature values.
* Added `monotone_constraints` as a parameter. This allows to force the algorithm to maintain an positive or negative monotonic relationship of the output with respect to the input features.
* Included automatic type conversion to `float64` in Numba version.
* Set minimum for `min_data_in_leaf`to `3`. There were some stability issues with the setting at `2` which led to division by zero in rare cases, and this resolves it.

## v1.0 ##
* Fixed bug where it was not possible to use `feature_fraction<1` on gpu because random number generator was cpu-based.
* Added possibility to output learned mean and variance when using `predict_dist` function.

## v0.9 ##
* Experimental TPU support for Google Cloud.
* Python 3.7 compatibility.
* Jupyter Notebook examples.

## v0.8 ##
* Added `studentt` distribution to Numba backend (with `df=3`).
* Added variance clipping to normal distribution of Numba backend.
* Some Numba backend code rewriting.
* JIT'ed `crps_ensemble` in Numba backend.
* Fixed bug where Torch-backend could not read Numba-backend trained models.
* Simpler bin calculation in Torch backend using torch.quantile.
* Completely rewrote distributed training. 
* Changed default seed.
* Bagging and feature subsampling is now only done in case these parameters are set different from their default values. This offers slight speedup for larger datasets.
* Fixed bug with `min_data_in_leaf`.
* Set default `tree_correlation` parameter to `log_10(n_samples_train) / 100` as per our paper.
* Added checkpointing, allowing users to continue training a model.

As of this version, the following is deprecated:
* The hyperparameter `gpu_device_ids` is replaced by a hyperparameter `gpu_device_id`.
* The vanilla `pgbm` package no longer offers parallel training; to perform parallel training `pgbm_dist` should be used.
* The hyperparameter `output_device` has been deprecated. All training is always performed on the chosen `device`. For parallelization, use `pgbm_dist`. 

## v0.7 ##
* Added `optimize_distribution` function to fit best distribution more easily.
* Fixed bug in Numba backend Poisson distribution.
* Improved speed of Numba backend version.
* Parallelized pre-computing split decisions on numba backend. Changed dtype to int16 instead of int32.
* Reduced integer size of CUDA kernel to short int.
* Split examples to support example for both backends.

### v0.6.1 ###
* Fixed bug in Numba feature importance calculation.

## v0.6 ##
* Fixed bug in Numba version where parallel construction of pre-computing splits failed.
* Fixed bug in Numba version where variance of distributions (other than Normal) was not properly clipped.
* Fixed Gamma distribution in Numba version.

### v0.5.1 ###
* Fixed bug in PyPi release where the custom CUDA kernel was not included in the distribution.

## v0.5 ##
* Restructuring of the package to avoid requirement to install Torch when using Numba backend and vice versa. From this version, to use the Numba backend users should use the package `pgbm_nb` whereas for the torch backend users should use `pgbm`. As of this version, `PGBM_numba` is deprecated and should be replaced by `PGBM`, where the backend is determined by whether the user imports the class `PGBM` from `pgbm` (Torch backend) or from `pgbm_nb` (Numba backend). The latter also facilitates easier switching between backends, by simply replacing the import at the start of a script. See also the updated examples.

## v0.4 ##
* Critical bug fix in Numba backend version.
* Modified load function in Numba backend version to improve consistency with Torch backend version.

## v0.3 ##
* Complete rewrite of prediction algorithm, enabling parallelization over the tree ensemble which speeds up prediction times. Added a 'parallel' option to the predict functions to allow users to choose prediction mode.
* Added truncation of learned tree arrays after training, to reduce storage cost of a PGBM model.
* Added appropriate type conversion when loading a PGBM model.
* Rewrote several matrix selection parts in favor of matrix multiplication, to speed up the algorithm during training.
* Renamed 'n_samples' in 'predict_dist' to 'n_forecasts' to avoid confusion between number of samples in a dataset and the number of forecasts that a user wants to create for a learned distribution.
* Removed pandas dependency. The PGBM backend now supports only torch and numpy arrays as datasets, whereas the Numba backend only supports numpy arrays.

## v0.2 ##
* Added a Numba-backend supported version of PGBM (PGBM_numba).
* Bugfixes in relation to saving and loading PGBM models.

## v0.1 ##
* Initial release.