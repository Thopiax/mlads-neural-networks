#!/bin/bash

source my_env/bin/activate
python condor/param_scripts/condor_main.py $@
deactivate
