.. pgbm documentation master file, created by
   sphinx-quickstart on Tue Aug 24 09:11:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png
   :align: right
   :width: 200
   :alt: Airlab Amsterdam

Welcome to PGBM's documentation!
================================

**Probabilistic Gradient Boosting Machines** (PGBM) is a probabilistic
gradient boosting framework in Python based on PyTorch/Numba, developed
by Airlab in Amsterdam. It provides the following advantages over
existing frameworks: 

- Probabilistic regression estimates instead of only point estimates. (`example <https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_housing_cpu.py>`__)
- Auto-differentiation of custom loss functions. (`example <https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example08_housing_autodiff.py>`__, `example <https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example10_covidhospitaladmissions.py>`__)
- Native GPU-acceleration. (`example <https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_housing_gpu.py>`__)
- Distributed training for CPU and GPU, across multiple nodes. (`examples <https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/>`__)
- Ability to optimize probabilistic estimates after training for a set of common distributions, without retraining the model. (`example <https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example07_optimizeddistribution.py>`__)

It is aimed at users interested in solving large-scale tabular
probabilistic regression problems, such as probabilistic time series
forecasting. For more details, read `our
paper <https://arxiv.org/abs/2106.01682>`__ or check out the
`examples <https://github.com/elephaint/pgbm/tree/main/examples>`__.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

	Installation <installation.md>
	Quick Start <quick_start.md>
	Features <features.md>
	Parameters <parameters.md>
	Examples <examples.md>
	Distributed Learning <distributed_learning.md>
	Function reference <function_reference.rst>
	Support <support.md>