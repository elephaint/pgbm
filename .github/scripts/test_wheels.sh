#!/bin/bash

set -e
set -x
pwd
INVENV=$(python -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')

pytest --pyargs pgbm.sklearn
pytest --pyargs pgbm.torch