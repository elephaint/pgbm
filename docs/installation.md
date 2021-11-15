# Installation #
We recommend installation of PGBM in a (virtual) Python environment via PyPi.

Don't know how to setup a Python environment and/or access a terminal? We recommend you start by installing [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). This allows you to create Python virtual environments and access them via a terminal. 

##  Installing via PyPi ##
Execute the following commands in a terminal after you have activated your Python (virtual) environment.

* Installing __without__ required dependencies: `pip install pgbm`. Use this if you have installed the below dependencies separately.
* Installing __with__ required dependencies (except for the Visual Studio Build Tools for Windows users):
  * Torch CPU+GPU: `pip install pgbm[torch-gpu] --find-links https://download.pytorch.org/whl/cu102/torch_stable.html`
  * Torch CPU-only: `pip install pgbm[torch-cpu]`
  * Numba: `pip install pgbm[numba]`
  * All versions (Torch CPU+GPU and Numba): `pip install pgbm[all] --find-links https://download.pytorch.org/whl/cu102/torch_stable.html`
  
## Installing from source ##
Clone the repository and run `setup.py install` in the newly created directory `pgbm`, i.e. run the following code from a terminal within a Python (virtual) environment of your choice:

  ```
  git clone https://github.com/elephaint/pgbm.git
  cd pgbm
  python setup.py install
  ```
When installing from source you need to install the dependencies separately.

## Dependencies ##
We offer PGBM using two backends, PyTorch (`import pgbm`) and Numba (`import pgbm_nb`). Both versions have separate dependencies.

### Torch backend ###
* [`torch>=1.8.0`, with CUDA Toolkit >= 10.2 for GPU acceleration](https://pytorch.org/get-started/locally/). Verify that PyTorch can find a cuda device on your machine by checking whether `torch.cuda.is_available()` returns `True` after installing PyTorch.
* [`ninja>=1.10.2.2`](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages) for compiling the custom c++ extensions.
* GPU training: the CUDA device should have [CUDA compute ability 2.x or higher](https://en.wikipedia.org/wiki/CUDA).
* Windows users: you may need to install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/?q=build+tools) and make sure you add compiler `cl` to your `PATH` environment variable (see [here](https://stackoverflow.com/a/65812244)). Verify that Windows can find `cl` by executing `where cl` in a Windows command line terminal.

### Numba backend ###
[`numba>=0.53.1`](https://numba.readthedocs.io/en/stable/user/installing.html). The Numba backend does not support differentiable loss functions and GPU training is also not supported.

## Verification ##
Both backends use [JIT-compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) so you incur additional compilation time the first time you use PGBM. The compiled code will be subsequently cached for all versions.

To verify, download & run an example from the examples folder to verify the installation is correct:
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_housing_cpu.py) to verify ability to train & predict on CPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_housing_gpu.py) to verify ability to train & predict on GPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example01_housing_cpu.py) to verify ability to train & predict on CPU with Numba backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/example13_housing_dist.py) to verify ability to perform distributed CPU, GPU, multi-CPU and/or multi-GPU training.