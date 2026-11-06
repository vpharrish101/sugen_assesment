#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: LLAMA_SERVER_BIN=/path/to/llama-server ./start_llama_server.sh /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf' >&2
  exit 2
fi

llama_server_bin=${LLAMA_SERVER_BIN:-llama-server}
gpu_layers=${LLAMA_GPU_LAYERS:-99}
threads=${LLAMA_THREADS:-4}

exec "$llama_server_bin" -m "$1" \
  --host 127.0.0.1 --port 8097 \
  -c 4096 -np 1 -t "$threads" -tb "$threads" -ngl "$gpu_layers" \
  -b 256 -ub 128 --cache-ram 0 --no-cache-idle-slots
