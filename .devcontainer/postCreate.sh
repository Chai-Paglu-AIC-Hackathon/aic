#!/usr/bin/env bash
set -euo pipefail

# Add other dependencies
# NOTE: This is already installed as part of the aic_eval base image, 
# but might as well do it here in case we switch base images in the future 
cd /ws_aic && pixi install --locked

# Add ROS setup to bashrc
echo "source /opt/ros/kilted/setup.bash" >> ~/.bashrc
echo "\nDevcontainer setup complete!!"