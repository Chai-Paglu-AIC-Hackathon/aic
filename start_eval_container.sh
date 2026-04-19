#!/bin/bash

# Indicate distrobox to use Docker as container manager
export DBX_CONTAINER_MANAGER=docker

# Create and enter the eval container
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
# If you do *not* have an NVIDIA GPU, remove the --nvidia flag for GPU support
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval

# Inside the container, start the environment
/entrypoint.sh ground_truth:=false start_aic_engine:=true