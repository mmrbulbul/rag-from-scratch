#!/bin/bash 
python3 -m isort $(find rag-systems -name "*.py")
python3 -m autopep8 --in-place $(find rag-systems -name "*.py")
