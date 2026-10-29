#!/usr/bin/env bash
set -euo pipefail
# Run explicitly with sudo on Ubuntu 24.04. No upgrades or source builds.
apt-get update
apt-get install -y cmake ninja-build python3-venv \
  llvm-18-dev=1:18.1.3-1ubuntu1 libmlir-18-dev=1:18.1.3-1ubuntu1 \
  mlir-18-tools=1:18.1.3-1ubuntu1 clang-18=1:18.1.3-1ubuntu1 libopenblas-dev
