FROM python:3.11-slim

# Install dependencies FIRST (curl is needed for Tailscale install)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# NOW install Tailscale (curl is available)
RUN curl -fsSL https://tailscale.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]