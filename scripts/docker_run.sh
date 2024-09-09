#!/bin/bash 

sudo docker run -it --net=host \
    --gpus=all -v $(pwd):/workspace:rw huggingface/transformers-pytorch-gpu bash