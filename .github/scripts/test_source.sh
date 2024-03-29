#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install pgbm/pgbm/dist/*.tar.gz
python -m pip install pytest numba torch --extra-index-url https://download.pytorch.org/whl/cpu ninja lightgbm
# Run the tests on the installed source distribution
mkdir tmp_for_test
cd tmp_for_test
pwd
pytest --pyargs pgbm.sklearn
pytest --pyargs pgbm.torch