#!/bin/bash

source venv/bin/activate
python -m condor.param_scripts.condor_main $@
deactivate
