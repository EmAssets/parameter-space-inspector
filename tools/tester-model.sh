#!/usr/bin/env bash
# The tester model: a dedicated Ollama instance, isolated from the model under test.
#
# PSI's model under test runs in PSI's own llama-server (and the inside engine). The model that tests it (Prometheus,
# Compare framings, self-consistency, fact checks, neuron naming) runs here instead: its own process, its own port
# (127.0.0.1:11435) and its own model folder (~/.psi-tester), sharing nothing with PSI's engines or with an Ollama
# you run for other things on 11434. It uses the Metal GPU on Apple Silicon. In PSI the models are local/tester/<model>.
#
#   tools/tester-model.sh start            start the instance (in the background; log in ~/.psi-tester/ollama.log)
#   tools/tester-model.sh pull <model>     download a model into the tester's own folder, e.g. llama3.1:8b
#   tools/tester-model.sh list             models the tester holds
#   tools/tester-model.sh status           is it up, and what is loaded
#   tools/tester-model.sh stop             stop it
#
# Docker instead (full container isolation, but CPU only on a Mac: Docker cannot reach the Metal GPU):
#   docker compose -f tools/tester-docker-compose.yml up -d     (same port, so PSI needs no change)
set -euo pipefail

DIR="${PSI_TESTER_DIR:-$HOME/.psi-tester}"
PORT="${PSI_TESTER_PORT:-11435}"
export OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODELS="$DIR/models" OLLAMA_KEEP_ALIVE="${PSI_TESTER_KEEP_ALIVE:-30m}"
PIDFILE="$DIR/ollama.pid" LOG="$DIR/ollama.log"

OLLAMA="$(command -v ollama || true)"
[ -z "$OLLAMA" ] && [ -x /Applications/Ollama.app/Contents/Resources/ollama ] && OLLAMA=/Applications/Ollama.app/Contents/Resources/ollama
[ -z "$OLLAMA" ] && { echo "Ollama is not installed (https://ollama.com/download)." >&2; exit 1; }

up() { curl -sf -m 2 "http://$OLLAMA_HOST/api/version" >/dev/null; }

case "${1:-status}" in
  start)
    if up; then echo "tester already up on $OLLAMA_HOST"; exit 0; fi
    mkdir -p "$DIR/models"
    nohup "$OLLAMA" serve >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 30); do up && { echo "tester up on $OLLAMA_HOST (models in $OLLAMA_MODELS, pid $(cat "$PIDFILE"))"; exit 0; }; sleep 0.5; done
    echo "tester did not start; see $LOG" >&2; exit 1 ;;
  stop)
    # Only an ollama process is stopped: after a reboot the pid file may name an unrelated process.
    if [ -f "$PIDFILE" ] && ps -p "$(cat "$PIDFILE")" -o comm= 2>/dev/null | grep -q ollama && kill "$(cat "$PIDFILE")" 2>/dev/null; then rm -f "$PIDFILE"; echo "tester stopped"; else rm -f "$PIDFILE"; echo "tester not running (from this script)"; fi ;;
  pull)
    [ -n "${2:-}" ] || { echo "usage: $0 pull <model>" >&2; exit 2; }
    up || { echo "start the tester first: $0 start" >&2; exit 1; }
    "$OLLAMA" pull "$2" ;;
  list)
    up || { echo "tester is not running" >&2; exit 1; }
    "$OLLAMA" list ;;
  status)
    if up; then echo "tester up on $OLLAMA_HOST, models in $OLLAMA_MODELS"; "$OLLAMA" ps; else echo "tester is not running ($0 start)"; fi ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
