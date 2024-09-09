#!/bin/bash 
python3 -m isort $(find rag_systems -name "*.py")
python3 -m autopep8 --in-place $(find rag_systems -name "*.py")
