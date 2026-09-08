#!/bin/bash
export PATH=/usr/local/miniconda3/envs/py312/bin:$PATH
cd /workspace
guidellm run \
  --backend "kind=openai_http,target=http://localhost:8000/v1,model=Qwen2.5-7B-Instruct" \
  --tokenizer "kind=hf_auto,model=/model/ModelScope/Qwen/Qwen2.5-7B-Instruct" \
  --profile "kind=sweep" \
  --data "kind=synthetic_text,prompt_tokens=256,output_tokens=128" \
  --constraint "kind=max_duration,seconds=40" \
  --output "kind=json,path=/workspace/guidellm_sweep.json" \
  > /workspace/guidellm.log 2>&1
echo "EXIT=$?" >> /workspace/guidellm.log
