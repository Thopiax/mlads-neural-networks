#!/bin/bash

export PYTHONPATH=venv/bin/python 
export PYTHONHOME=venv/bin/python 

source venv/bin/activate
python -m condor.param_scripts.condor_main $@
deactivate
