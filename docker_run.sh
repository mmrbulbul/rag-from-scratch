#!/bin/bash 

sudo docker run -it --rm --net=host \
    --gpus=all  -v $(pwd):/workspace huggingface/transformers-pytorch-gpu bash