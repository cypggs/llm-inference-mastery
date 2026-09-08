#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PATH=/usr/local/miniconda3/envs/py312/bin:$PATH
cd /workspace
nohup python -m sglang.launch_server \
  --model-path /model/ModelScope/Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8001 --mem-fraction-static 0.85 --tool-call-parser qwen25 \
  > /workspace/sglang.log 2>&1 &
disown
echo "SGLang launched PID=$!"
