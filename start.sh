#!/bin/sh

echo ">>> Starting tailscaled with SOCKS5 proxy..."
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

echo ">>> Creating TCP tunnels to PostgreSQL via Tailscale SOCKS5..."
# Async engine (asyncpg) uses port 5433
socat TCP-LISTEN:5433,fork,reuseaddr SOCKS5:127.0.0.1:100.102.226.118:5432,socksport=1055 &
# Sync engine (utils.py fetchers) uses port 5434
socat TCP-LISTEN:5434,fork,reuseaddr SOCKS5:127.0.0.1:100.102.226.118:5432,socksport=1055 &
sleep 2
echo ">>> TCP tunnels established: async=5433, sync=5434"

echo ">>> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}


