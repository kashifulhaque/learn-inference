#!/usr/bin/env bash
#
# Deploys learn-inference to the VM over SSH.
#
# Syncs the repository, keeps whatever .env is already on the server, rebuilds
# the image, and waits for the health check to pass.
#
# Usage:
#   scripts/deploy.sh [--host HOST] [--path PATH] [--key KEY]

set -euo pipefail

HOST="${LI_HOST:-ifkash@vm.ifkash.dev}"
REMOTE_PATH="${LI_REMOTE_PATH:-~/docs/learn-inference}"
SSH_KEY="${LI_SSH_KEY:-$HOME/.ssh/id_ed25519_kh}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --path) REMOTE_PATH="$2"; shift 2 ;;
    --key)  SSH_KEY="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

echo "==> Syncing $REPO_ROOT to $HOST:$REMOTE_PATH"
ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE_PATH"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'node_modules' \
  --exclude 'frontend/dist' \
  --exclude '__pycache__' \
  --exclude '.env' \
  --exclude 'data' \
  -e "ssh ${SSH_OPTS[*]}" \
  "$REPO_ROOT/" "$HOST:$REMOTE_PATH/"

echo "==> Checking for .env on the server"
if ! ssh "${SSH_OPTS[@]}" "$HOST" "test -f $REMOTE_PATH/.env"; then
  echo "No .env on the server. Copy .env.example to $REMOTE_PATH/.env and fill" >&2
  echo "it in, then run this script again." >&2
  exit 1
fi

echo "==> Building and starting"
ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_PATH && docker compose up -d --build"

echo "==> Waiting for the health check"
for attempt in $(seq 1 60); do
  if ssh "${SSH_OPTS[@]}" "$HOST" \
      "curl -sf http://127.0.0.1:8087/api/health > /dev/null"; then
    echo "==> Healthy"
    ssh "${SSH_OPTS[@]}" "$HOST" "curl -s http://127.0.0.1:8087/api/health"
    echo
    exit 0
  fi
  sleep 3
done

echo "The app did not become healthy. Recent logs:" >&2
ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_PATH && docker compose logs --tail 60 app" >&2
exit 1
