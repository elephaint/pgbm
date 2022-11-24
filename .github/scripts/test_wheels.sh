#!/bin/bash

set -e
set -x

# if [[ "$OSTYPE" != "linux-gnu" ]]; then
#     # The Linux test environment is run in a Docker container and
#     # it is not possible to copy the test configuration file (yet)
#     cp $CONFTEST_PATH $CONFTEST_NAME
# fi

python $1/examples/torch/example01_housing_cpu.py
python $1/examples/sklearn/example01_housing_cpu.py