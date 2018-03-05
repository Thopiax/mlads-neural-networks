#!/bin/bash

source my_env/bin/activate
python -m condor.param_scripts.condor_main $@
deactivate
