# Distributed learning #

To facilitate distributed training, PGBM leverages the `torch.distributed` backend. 

We expose PGBM for distributed learning via `PGBMDist`. This version works equivalently as `PGBM`, but allows to distribute the computations across multiple nodes and devices. Each node and device will have its own copy of the model and each model needs to be initialized with the total world size and process rank, for example:
```
from pgbm.torch import PGBMDist
model = PGBMDist(world_size=1, rank=0)
```
In the following, we refer to the [two examples we provide in our Github repository](https://github.com/elephaint/pgbm/tree/main/examples/torch_dist).

Note: to use the `higgs` dataset in any of the examples, download [here](https://archive.ics.uci.edu/ml/datasets/HIGGS), unpack and save `HIGGS.csv` to your local working directory.

**Distributed training arguments**

The following arguments can be set on the command line when using our template for distributed training:
* `-n`, `--nodes`, `default=1`: number of nodes used in the distributed training setting.
* `-p`, `--processes`, `default=1`: number of processes per node. For multi-GPU training, this should be equal to the number of GPUs per node.
* `-nr`, `--nr`, `default=0`: rank of the node within all nodes.
* `-b`, `--backend`, `default='gloo'`: backend for distributed training. Valid options are, dependent on your OS, PyTorch installation and distributed setting: `gloo`, `nccl`, `mpi`. 
* `-d`, `--device`, `default='cpu'`: device for training. Valid options: `cpu`, `gpu`.
* `--MASTER_ADDR`, `default='127.0.0.1'`: IP address of master process for distributed training.
* `--MASTER_PORT`, `default='29500'`: Port of node of master process for distributed training.

**Notes**
* For more details on using distributed training with PyTorch, see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
* Each process will have its own copy of the final PGBM model. Hence, to save a model, simply invoke a save method in one of the processes (i.e. `model.save(filename)`) or use checkpointing by setting `checkpoint=True` in the parameters dict.
* For multi-gpu training it is assumed that each node has the same amount of GPUs. 
* It should be possible to train with multiple GPUs of different types or generations, as long as it is a device with CUDA compute ability 2.x or up.

**Limitations**

It is possible to use autodifferentiation for custom loss functions during distributed training by setting `derivatives=approx`, however the gradient and hessian information is not (yet) shared across processes. This is not necessary for standard loss functions where the gradient information of example A depends only on the predicted value of example A, but for more complex loss functions this might be an issue (for example in the case of hierarchical time series forecasting). We hope to address this in future releases. 

## Distributed CPU training ##
The examples should be run from a terminal from each node, i.e. run `python [filename].py`. Each node should satisfy all the dependencies to run PGBM.

### Single node, CPU training ###
Note that this is equivalent to non-distributed training. 
* Example 1: `python example13_housing_dist.py`.
* Example 2: `python example14_higgs_dist.py`.

### Multiple nodes, CPU training ###
* Example 1. For example when training on two nodes. The master node is located at IP address `192.168.0.1` (this is a mock example) and the port is `29500`.
  * On first (master) node: `python example13_housing_dist.py -n 2 -nr 0 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On second node: `python example13_housing_dist.py -n 2 -nr 1 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.

* Example 2. For example when training on four nodes. The master node is located at IP address `192.168.0.1` (this is a mock example) and the port is `29500`.
  * On first (master) node: `python example14_higgs_dist.py -n 4 -nr 0 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On second node: `python example14_higgs_dist.py -n 4 -nr 1 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On third node: `python example14_higgs_dist.py -n 4 -nr 2 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On fourth node: `python example14_higgs_dist.py -n 4 -nr 3 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.

## Distributed GPU training ## 
The examples should be run from a terminal from each node, i.e. run `python [filename].py`. 

### Single node, single GPU ###
Note that this is equivalent to non-distributed training. Training will be performed on the GPU on the first index (0) of the node. 
* Example 1: `python example13_housing_dist.py -d gpu`.
* Example 2: `python example13_higgs_dist.py -d gpu`.

### Single node, multiple GPUs ###
For example when training on 4 GPUs:
* Example 1: `python example13_housing_dist.py -d gpu -p 4`.
* Example 2: `python example14_higgs_dist.py -d gpu -p 4`.

### Multiple nodes, multiple GPUs ###
* Example 1. For example when training on two nodes with 8 GPUs each. The master node is located at IP address `192.168.0.1` (this is a mock example) and the port is `29500`.
  * On first (master) node: `python example13_housing_dist.py -d gpu -p 8 -n 2 -nr 0 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On second node: `python example13_housing_dist.py -d gpu -p 8 -n 2 -nr 1 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
* Example 2. For example when training on four nodes with 2 GPUs each. 
  * On first (master) node: `python example14_higgs_dist.py -d gpu -p 2 -n 4 -nr 0 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On second node: `python example14_higgs_dist.py -d gpu -p 2 -n 4 -nr 1 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On third node: `python example14_higgs_dist.py -d gpu -p 2 -n 4 -nr 2 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
  * On fourth node: `python example14_higgs_dist.py -d gpu -p 2 -n 4 -nr 3 --MASTER_ADDR 192.168.0.1 --MASTER_PORT 29500`.
