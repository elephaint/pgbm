# PGBM #

Probabilistic Gradient Boosting Machines (PGBM) is a gradient boosting framework. It provides the following advantages:
* Probabilistic regression estimates instead of only point estimates.
* GPU-acceleration.
* Auto-differentiation of custom loss functions.

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [the paper](arxiv-link).

### Installation ###
* Clone the repository
* Go to {your_installation_directory}/pgbm
* Run the demo from the folder pgbm/demos/demo_pgbm.py to verify the installation is correct. Use both 'gpu' and 'cpu' as device to check if you are able to train on both GPU and CPU.
* The first time it may take a bit longer to import pgbm as it relies on JIT compilation for the custom CUDA kernel. 

#### Dependencies ####

* PyTorch >= 1.7.0 with CUDA 11.0 for GPU acceleration (download from: https://pytorch.org/get-started/locally/)
* PGBM uses a custom CUDA kernel which needs to be compiled; this may require installing a suitable compiler (e.g. gcc) although installing PyTorch according to the official docs should install all the required dependencies.

To run the experiments comparing against baseline models install the following packages using `pip` or  `conda`:

* pandas with xlrd 1.2.0
* scikit-learn
* properscoring
* numpy
* ngboost
* lightgbm
* matplotlib

### Examples ###
See the folder pgbm/demos. In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric (see the demo folder for an example).

### Experiments ###

The experiments from our paper can be replicated by running the scripts in the (experiments folder). Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to pgbm/datasets.

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), [Sebastian Schelter](mailto:s.schelter@uva.nl), [Maarten de Rijke](mailto:m.derijke@uva.nl). [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://linktopaper). Accepted for publication at SIGKDD '21.

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### ToDo ###
We intend to have the package as lightweight as possible.

- [x] Add extreme value distributions such as Gumbel and Weibull to distribution choices.
- [ ] Remove properscoring dependency (crps_ensemble can be calculated much faster on GPU)
- [ ] Full support of Torch distributed (across multiple GPUs and nodes, now only across multiple GPUs supported).
- [ ] Set default values for learning parameters.
- [ ] Remove JIT-compilation dependency and offer as an installable package via `pip` or `conda`.