FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    iptables \
    socat \
    build-essential \
    libta-lib-dev \        # ← ADD THIS
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://tailscale.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]