#!/bin/bash

set -e
set -x

cd ../

python -m venv test_env
source test_env/bin/activate

python -m pip install pgbm/dist/*.tar.gz

# Run the tests on the installed source distribution
mkdir tmp_for_test
cd tmp_for_test
pwd

python $1/../tests/test_training.py