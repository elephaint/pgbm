#!/bin/bash

set -e
set -x

# python $1/examples/torch/example01_housing_cpu.py
# python $1/examples/sklearn/example01_housing_cpu.py
python $1/pgbm/tests/test_training.py