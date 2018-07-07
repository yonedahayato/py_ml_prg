#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo $SCRIPT_DIR

# install library
# docker build -f ./dockerfile/Dockerfile_lib -t py_ml_prg_lib .

docker build -f ./dockerfile/Dockerfile -t py_ml_prg .
docker run -v $SCRIPT_DIR:/home/study -it --rm py_ml_prg python iris_classification.py
docker rmi py_ml_prg
