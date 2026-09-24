#!/usr/bin/env bash
# Queue heavy jobs through task-spooler, one at a time (8GB Mac).
# usage: scripts/queue.sh <command...>        queue a job (runs from repo root, logs/tsp-<id>.log when done)
#        scripts/queue.sh ls | tail [id] | rm <id> | kill | clear
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TSP="$(command -v tsp || command -v ts || true)"
[[ -n "$TSP" ]] || { echo "task-spooler not found: brew install task-spooler (binary: ts) / apt install task-spooler (binary: tsp)"; exit 1; }
export TS_SLOTS=1
export TS_MAXFINISHED=50
mkdir -p "$ROOT/logs"
case "${1:-}" in
  ls|"")   "$TSP" -l ;;
  tail)    shift; "$TSP" -t "${1:-}" ;;
  rm)      shift; "$TSP" -r "$1" ;;
  kill)    "$TSP" -k ;;
  clear)   "$TSP" -C ;;
  *)
    id=$("$TSP" -L "$(echo "$*" | cut -c1-40)" bash -lc "cd '$ROOT' && source .venv/bin/activate && $*")
    echo "queued job $id: $*"
    echo "  follow: scripts/queue.sh tail $id   (logs/tsp-$id.log after finish)"
    "$TSP" -D "$id" bash -lc "cp \"\$('$TSP' -o $id)\" '$ROOT/logs/tsp-$id.log' 2>/dev/null || true" >/dev/null
    ;;
esac
