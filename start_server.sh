#!/usr/bin/env bash
# launch script for gpt-oss-20b q4_k_m on cpu-only hardware.
#
# numactl wrapper pins the process to NUMA node 0 (one socket's 22 physical
# cores + its local ~96 GB of RAM). the 11 GB model fits comfortably on
# one node, and pinning eliminates cross-socket QPI traffic that the OS
# would otherwise incur if threads drifted to socket 1.
#
# --threads: 22 = physical cores on one socket of the dual Xeon Gold 6152.
#   going higher only helps with --numa distribute + numactl --interleave,
#   and for an 11 GB model the single-socket win is usually ≥ the dual-
#   socket win. do NOT use hyperthreads (would be 44).
#
# --ctx-size: 32768 is solid for multi-step agentic loops. lower to 16384
#   if ram is tight; raise to 65536 for long-running agents.
#
# --batch-size 2048: default; speeds up prompt processing. --ubatch-size
#   stays at 512 (the physical micro-batch cap).
#
# --flash-attn on: reduces attention memory bandwidth at long context.
#
# --mlock: pins the model in RAM. combined with numactl --membind=0, the
#   11 GB of weights stay resident on node 0 and never page out.
#
# --jinja: required. activates the harmony jinja chat template embedded in
#   the gguf. without it, tool calls and reasoning_effort both fail.
#
# --chat-template-kwargs: sets the default reasoning effort for all requests.
#   "low" is appropriate for react agents. can be overridden per-request.
#   valid values: "low" | "medium" | "high"
#
# sampling values (temp, top-p, top-k, min-p) are openai's recommendations
# for gpt-oss. do not lower temperature.

numactl --cpunodebind=0 --membind=0 \
llama-server \
  --model                "/mnt/research/active/ipi/models/gpt-oss-20b-Q4_K_M.gguf" \
  --threads              22 \
  --ctx-size             32768 \
  --batch-size           2048 \
  --ubatch-size          512 \
  --flash-attn \
  --mlock \
  --temp                 1.0 \
  --top-p                1.0 \
  --top-k                0 \
  --min-p                0.0 \
  --jinja \
  --chat-template-kwargs '{"reasoning_effort": "low"}' \
  --host                 127.0.0.1 \
  --port                 8080
