#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

python -m pip install pgbm/pgbm/dist/*.tar.gz
python -m pip install pytest numba torch --extra-index-url https://download.pytorch.org/whl/cpu ninja
# Run the tests on the installed source distribution

# python pgbm/pgbm/tests/test_training.py
mkdir tmp_for_test
cp pgbm/pgbm/conftest.py tmp_for_test
cd tmp_for_test
pwd
# pytest pgbm/pgbm/sklearn/tests/
# pytest pgbm/pgbm/torch/tests/
pytest sklearn
pytest torch