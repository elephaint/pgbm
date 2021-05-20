# Examples #

This folder contains examples of PGBM. Our examples are aimed to illustrate the following:
1. How to train PGBM, on CPU, GPU and multi-GPU.
2. How to train PGBM using a validation loop.
3. How PGBM compares to other methods such as NGBoost and LightGBM.
4. How the choice of output distribution can be optimized after training.

Example index:
1. Probabilistic regression for the boston housing dataset, training on CPU.
2. Probabilistic regression for the boston housing dataset, training on single GPU.
3. Probabilistic regression for the boston housing dataset, training on multi-GPU.
4. Probabilistic regression for the boston housing dataset, using a validation loop.
5. Probabilistic regression for the boston housing dataset, comparison to NGBoost.
6. Training time comparison between PGBM (trained on GPU) and LightGBM (CPU) for a range of datasets.
7. How to use a validation loop to optimize choice of distribution and tree correlation parameter.

PGBM employs the following set of hyperparameters:
* 'min_split_gain': [default=0.0] the minimum gain for a node to split when building a tree.
* ''