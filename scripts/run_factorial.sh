#!/bin/bash
# Run the N>1 unanimous cells of the factorial
cd "$(dirname "$0")/.."

CORPORA=(syn-benign-info syn-low syn-medium syn-high)
SAMPLES=(3 5 7)

for N in "${SAMPLES[@]}"; do
  for corpus in "${CORPORA[@]}"; do
    out=logs/factorial/results.jsonl.${corpus}-n${N}-unanimous
    if [ -f "$out" ]; then
      echo "$(date -Is)  skip $corpus n=$N (already archived)"
      continue
    fi
    echo "$(date -Is)  === run $corpus n=$N unanimous ==="
    rm -f logs/agent.log logs/results.jsonl
    if python main.py --corpus test-data/$corpus --samples $N --voting unanimous; then
      mv logs/agent.log    logs/factorial/agent.log.${corpus}-n${N}-unanimous
      mv logs/results.jsonl "$out"
      echo "$(date -Is)  done $corpus n=$N"
    else
      echo "$(date -Is)  FAIL $corpus n=$N (exit $?) — partial logs removed"
      rm -f logs/agent.log logs/results.jsonl
    fi
  done
done
echo "$(date -Is)  FACTORIAL DONE"
