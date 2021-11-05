# Examples #

This folder contains examples of PGBM. The examples illustrate the following:
* Example 13/14: How to train PGBM in a distributed setting. See below on how to test for each setting.
* dist_template: this is a template that can be used for distributed training. 

Note: to use the `higgs` dataset in any of the examples, download [here](https://archive.ics.uci.edu/ml/datasets/HIGGS), unpack and save `HIGGS.csv` to your local working directory.

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
