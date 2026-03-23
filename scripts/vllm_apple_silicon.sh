#!/bin/bash

# vLLM for CPU install
# git clone https://github.com/vllm-project/vllm.git
# cd vllm
# uv venv
# source .venv/bin/activate
# uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
# uv pip install -e .

# vLLM Metal for Apple Silicon install
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source $HOME/.venv-vllm-metal/bin/activate
