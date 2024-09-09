Practice repo to build various rag systems and evaluate them.


## Environment preparation
1. Pull the docker image 
    ```bash 
    docker pull huggingface/transformers-pytorch-gpu
    ```

    **Docker:** [huggingface/transformers-pytorch-gpu:latest](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu)

    **Image ID:**   `f9b1f0a252ca`

2. Run the docker container `./docker_run.sh`

3. Install python requirements
    ```bash
    cd workspace
    pip install -r requirements.txt
    ```
4. Save the container as a new image if necessary.

