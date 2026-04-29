#!/bin/sh
set -e

echo ">>> Starting tailscaled..."
tailscaled --state=mem: --tun=userspace-networking &
TAILSCALED_PID=$!

# Give tailscaled a moment to initialize
sleep 2

echo ">>> Joining Tailscale network..."
tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname=railway-backend --accept-routes

echo ">>> Tailscale status:"
tailscale status

echo ">>> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000