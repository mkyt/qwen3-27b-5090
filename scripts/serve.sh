#!/usr/bin/env bash
# Start / stop the serve container from anywhere (scripts/windows/serve.cmd calls this through wsl.exe).
#   scripts/serve.sh up          # docker compose up -d serve, then wait until the server answers (the engine keeps loading);
#                                # the image is the one pulled (or built) last, built here if there is none
#   scripts/serve.sh up --idle   # the same with ENGINE_AUTOSTART=0: no VRAM until the engine is started at /ui
#   scripts/serve.sh up --build  # with the image built from this checkout (after a code change)
#   scripts/serve.sh update      # pull the published image (ghcr.io), then up
#   scripts/serve.sh down        # docker compose down
#   scripts/serve.sh status      # the container's state and /health
#   scripts/serve.sh keepalive   # block while the container exists: keeps the WSL distro from shutting down when idle
# The settings are those of compose.yaml and the environment (MAX_SEQS=4 scripts/serve.sh up, ...).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8000
NAME=qwen3-27b-5090

case "${1:-}" in
  up)
    build=()
    for arg in "${@:2}"; do
      case "$arg" in
        --idle) export ENGINE_AUTOSTART=0 ;;
        --build) build=(--build) ;;
        *) echo "unknown option $arg" >&2; exit 2 ;;
      esac
    done
    # Another environment on the port (vLLM, SGLang) also holds the GPU: say so instead of a bind error.
    other=$(docker ps --filter "publish=$PORT" --format '{{.Names}}' | grep -vx "$NAME" || true)
    if [ -n "$other" ]; then
      echo "port $PORT is taken by: $other (stop it first; only one model fits on the GPU)" >&2
      exit 1
    fi
    docker compose up -d "${build[@]}" serve
    echo "waiting for the server..."
    for _ in $(seq 900); do
      if curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
        echo "server up: http://127.0.0.1:$PORT/ui (/health turns 200 when the engine is ready)"
        exit 0
      fi
      if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
        docker compose logs --tail 30 serve >&2
        exit 1
      fi
      sleep 1
    done
    echo "the server did not answer in 15 minutes: docker compose logs serve" >&2
    exit 1
    ;;
  update)
    docker compose pull serve
    exec "$0" up "${@:2}"
    ;;
  down)
    docker compose down
    ;;
  status)
    state=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo "not created")
    health=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
    echo "container $NAME: $state, /health: $health (200 ready, 503 engine loading or stopped)"
    ;;
  keepalive)
    exec 9>"/tmp/$NAME-keepalive.lock"
    flock -n 9 || exit 0  # (one loop is enough)
    while [ -n "$(docker ps -a --filter "name=^$NAME\$" --filter status=running --filter status=restarting -q)" ]; do
      sleep 30
    done
    ;;
  *)
    echo "usage: $0 up [--idle] [--build] | update [--idle] | down | status | keepalive" >&2
    exit 2
    ;;
esac
