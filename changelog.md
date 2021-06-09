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