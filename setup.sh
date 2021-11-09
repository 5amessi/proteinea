#!/bin/bash
set -e
# get parent dir
home="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )/proteinea"

# delete Python virtual environment and recreate it
rm -rf $home/test/env

virtualenv --python=python3.6 $home/env

# enter the venv and install dependencies
source $home/env/bin/activate

pip install -r $home/requirements.txt
