#!/bin/bash

source my_env/bin/activate
python "$1"
deactivate
