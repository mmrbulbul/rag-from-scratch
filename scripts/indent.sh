#!/bin/bash 
python3 -m isort **/**.py
python3 -m autopep8 --in-place **/**.py
