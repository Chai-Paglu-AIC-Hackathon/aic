#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  curl ca-certificates git \
  build-essential cmake pkg-config \
  python3 python3-venv \
  jq unzip \
  bash-completion \
  locales

# Ensure UTF-8 locale (ROS tooling sometimes expects it)
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

# Install Pixi (per guide)
# Guide command: curl -fsSL https://pixi.sh/install.sh | sh
# We do it non-interactively and ensure PATH via PIXI_HOME in devcontainer.json
curl -fsSL https://pixi.sh/install.sh | bash

# Make sure pixi is usable in non-login shells too
if ! grep -q 'PIXI_HOME' ~/.bashrc; then
  {
    echo ''
    echo '# Pixi'
    echo 'export PIXI_HOME="$HOME/.pixi"'
    echo 'export PATH="$PIXI_HOME/bin:$PATH"'
  } >> ~/.bashrc
fi