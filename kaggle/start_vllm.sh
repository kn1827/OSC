#!/usr/bin/env bash
# Start / stop the vLLM servers used by config/vllm/*.yaml on a Kaggle "GPU T4 x2" machine.
#
#   bash kaggle/start_vllm.sh debate    # solver A + solver B on GPU 0, critic on GPU 1
#   bash kaggle/start_vllm.sh verify    # verifier on GPU 0 (run after `stop`)
#   bash kaggle/start_vllm.sh stop      # kill every vLLM server
#   CRITIC=mistral bash kaggle/start_vllm.sh debate   # if Gemma-2 fails to load on T4
#
# T4 = 16 GB, compute capability 7.5: fp16 only (no bf16), AWQ 4-bit fits two 7-8B models per GPU.
# Two servers on one GPU each get a fixed share of its memory (--gpu-memory-utilization), and
# are started one after the other because vLLM measures free memory at start-up.
# Logs: logs/vllm_<port>.log.  Tune with MAXLEN (context length) and MAXSEQS (batch size).
set -euo pipefail

MAXLEN=${MAXLEN:-6144}
MAXSEQS=${MAXSEQS:-16}
mkdir -p logs

start() {   # model gpu port memory_share
  echo "starting $1 on GPU $2, port $3 (memory share $4)"
  CUDA_VISIBLE_DEVICES=$2 nohup python -m vllm.entrypoints.openai.api_server \
    --model "$1" --port "$3" --dtype half --gpu-memory-utilization "$4" \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --enforce-eager \
    > "logs/vllm_$3.log" 2>&1 &
}

wait_up() { # port
  for _ in $(seq 1 180); do
    if curl -s "localhost:$1/v1/models" > /dev/null; then echo "port $1 ready"; return 0; fi
    sleep 5
  done
  echo "port $1 did not come up — see logs/vllm_$1.log"; tail -n 30 "logs/vllm_$1.log"; return 1
}

case "${1:-}" in
  debate)
    start Qwen/Qwen2.5-7B-Instruct-AWQ 0 8001 0.45; wait_up 8001
    start hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 0 8002 0.45; wait_up 8002
    if [ "${CRITIC:-gemma}" = "mistral" ]; then
      start solidrust/Mistral-7B-Instruct-v0.3-AWQ 1 8003 0.85; wait_up 8003
    else
      start hugging-quants/gemma-2-9b-it-AWQ-INT4 1 8003 0.85; wait_up 8003
    fi
    ;;
  verify)
    start stelterlab/phi-4-AWQ 0 8004 0.85; wait_up 8004
    ;;
  stop)
    pkill -f vllm.entrypoints.openai.api_server || true
    sleep 5; echo "stopped"
    ;;
  *)
    echo "usage: bash kaggle/start_vllm.sh {debate|verify|stop}"; exit 1
    ;;
esac
