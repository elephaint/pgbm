# Installation #
We recommend installation of PGBM in a (virtual) Python environment via PyPi.

Don't know how to setup a Python environment and/or access a terminal? We recommend you start by installing [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). This allows you to create Python virtual environments and access them via a terminal. 

##  Installing via PyPi ##
First, make sure you have separately installed the various dependencies (see below). Then, execute the following commands in a terminal after you have activated your Python (virtual) environment.

`pip install pgbm`
  
## Installing from source ##
Clone the repository and run `setup.py install` in the newly created directory `pgbm`, i.e. run the following code from a terminal within a Python (virtual) environment of your choice:

  ```
  git clone https://github.com/elephaint/pgbm.git
  cd pgbm
  python setup.py install
  ```
When installing from source you need to install the dependencies separately.

## Dependencies ##
We offer PGBM using two backends, PyTorch (`import pgbm.torch`) and scikit-learn (`import pgbm.sklearn`). Only the PyTorch-backend requires you to install dependencies separately.

### Torch backend ###
* [`torch>=1.8.0`, with CUDA Toolkit >= 10.2 for GPU acceleration](https://pytorch.org/get-started/locally/). Verify that PyTorch can find a cuda device on your machine by checking whether `torch.cuda.is_available()` returns `True` after installing PyTorch.
* GPU training: the CUDA device should have [CUDA compute ability 2.x or higher](https://en.wikipedia.org/wiki/CUDA).
* Windows users: you may need to install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/?q=build+tools) and make sure you add compiler `cl` to your `PATH` environment variable (see [here](https://stackoverflow.com/a/65812244)). Verify that Windows can find `cl` by executing `where cl` in a Windows command line terminal.

## Verification ##
Both backends use [JIT-compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) so you incur additional compilation time the first time you use PGBM. The compiled code will be subsequently cached for all versions.

To verify, download & run an example from the examples folder to verify the installation is correct:
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example01_housing_cpu.py) to verify ability to train & predict on CPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example02_housing_gpu.py) to verify ability to train & predict on GPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example01_housing_cpu.py) to verify ability to train & predict on CPU with Scikit-learn backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/torch_dist/example13_housing_dist.py) to verify ability to perform distributed CPU, GPU, multi-CPU and/or multi-GPU training.