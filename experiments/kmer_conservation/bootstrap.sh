#!/bin/bash
set -euo pipefail
# Native EC2 worker: terminate and delete its root disk at the ten-hour deadline.
shutdown -h +600
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv git curl unzip build-essential time
install -d -o ubuntu -g ubuntu /data/issue568
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/aws.zip
unzip -q /tmp/aws.zip -d /tmp
/tmp/aws/install
touch /data/issue568/bootstrap.done
