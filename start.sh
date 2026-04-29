#!/bin/sh

echo ">>> Starting tailscaled..."
tailscaled --state=mem: --tun=userspace-networking --socks5-server=127.0.0.1:1055 &
TAILSCALED_PID=$!
echo ">>> tailscaled PID: $TAILSCALED_PID"

sleep 3

echo ">>> Joining Tailscale network..."
tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname=railway-backend --accept-routes
TS_EXIT=$?

if [ $TS_EXIT -ne 0 ]; then
  echo ">>> ERROR: tailscale up failed with exit code $TS_EXIT"
  exit 1
fi

echo ">>> Tailscale status:"
tailscale status

echo ">>> Creating TCP tunnel to PostgreSQL via Tailscale..."
# Forward local port 5433 → your PC's PostgreSQL via Tailscale SOCKS5 proxy
socat TCP-LISTEN:5433,fork,reuseaddr SOCKS5:127.0.0.1:100.102.226.118:5432,socksport=1055 &
sleep 2
echo ">>> TCP tunnel established on localhost:5433"

echo ">>> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}