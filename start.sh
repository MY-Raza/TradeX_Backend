#!/bin/sh
set -e  # exit on any error — makes failures visible immediately

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
  --timeout=30s
TS_EXIT=$?
if [ $TS_EXIT -ne 0 ]; then
  echo ">>> ERROR: tailscale up failed with exit code $TS_EXIT"
  exit 1
fi

echo ">>> Tailscale status:"
tailscale status

# ---------------------------------------------------------------------------
# socat: tunnel localhost:5433 → DB machine via Tailscale SOCKS5
# We run socat in background. If the DB peer is temporarily unreachable,
# socat will retry on the next connection attempt — it does NOT exit.
# fork  = handle multiple simultaneous connections
# retry = keep accepting even if a forwarded connection fails
# ---------------------------------------------------------------------------
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

# Sanity-check: confirm socat processes are still alive
if ! kill -0 $SOCAT_ASYNC_PID 2>/dev/null; then
  echo ">>> ERROR: socat (async/5433) died immediately — check Tailscale peer 100.102.226.118"
  exit 1
fi
if ! kill -0 $SOCAT_SYNC_PID 2>/dev/null; then
  echo ">>> ERROR: socat (sync/5434) died immediately — check Tailscale peer 100.102.226.118"
  exit 1
fi

echo ">>> TCP tunnels established: async=5433, sync=5434"
echo ">>> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"