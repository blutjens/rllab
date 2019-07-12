#!/bin/bash
set -e

PROJECT_PATH="/home/$USER/Desktop/vector-solenoid/rllab"

cd $PROJECT_PATH
python setup.py install

#pip3 install -e $PROJECT_PATH/gym #
#pip install scikit-learn --upgrade

cd $PROJECT_PATH/rllab/sandbox/vime/experiments
#python -m jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

exec "$@"
