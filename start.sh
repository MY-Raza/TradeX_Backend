#!/bin/sh
set -e

echo ">>> Starting tailscaled with SOCKS5 proxy..."
tailscaled --state=mem: --tun=userspace-networking --socks5-server=127.0.0.1:1055 &
TAILSCALED_PID=$!
echo ">>> tailscaled PID: $TAILSCALED_PID"
sleep 3

echo ">>> Joining Tailscale network..."
tailscale up \
  --authkey="${TAILSCALE_AUTHKEY}" \
  --hostname=railway-backend \
  --accept-routes \
  --ephemeral \
  --timeout=30s
TS_EXIT=$?
if [ $TS_EXIT -ne 0 ]; then
  echo ">>> ERROR: tailscale up failed with exit code $TS_EXIT"
  exit 1
fi

# NOTE: 'tailscale status' removed — it hangs in userspace-networking mode
# waiting for peer discovery and blocks the rest of the script indefinitely.

echo ">>> Creating TCP tunnels to PostgreSQL via Tailscale SOCKS5..."
socat TCP-LISTEN:5433,fork,reuseaddr \
  SOCKS5:127.0.0.1:100.102.226.118:5432,socksport=1055 &
SOCAT_ASYNC_PID=$!
echo ">>> socat async (port 5433) PID: $SOCAT_ASYNC_PID"

socat TCP-LISTEN:5434,fork,reuseaddr \
  SOCKS5:127.0.0.1:100.102.226.118:5432,socksport=1055 &
SOCAT_SYNC_PID=$!
echo ">>> socat sync  (port 5434) PID: $SOCAT_SYNC_PID"

sleep 2

if ! kill -0 $SOCAT_ASYNC_PID 2>/dev/null; then
  echo ">>> ERROR: socat (async/5433) died — is Tailscale peer 100.102.226.118 online?"
  exit 1
fi
if ! kill -0 $SOCAT_SYNC_PID 2>/dev/null; then
  echo ">>> ERROR: socat (sync/5434) died — is Tailscale peer 100.102.226.118 online?"
  exit 1
fi

echo ">>> TCP tunnels established: async=5433, sync=5434"
echo ">>> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"