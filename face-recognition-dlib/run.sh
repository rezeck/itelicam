#!/usr/bin/env bash

RECOVER=PYTHONPATH
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages

python3.5 $1

export PYTHONPATH=RECOVER
