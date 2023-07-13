#!/bin/bash -l
set -euo pipefail

# conda env create -f env.yaml

conda install pytorch torchvision torchaudio cpuonly -c pytorch
# second argument is required for installing dependencies
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cardio-rl==0.0.4
pip install hydra-core --upgrade