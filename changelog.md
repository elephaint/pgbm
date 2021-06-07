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