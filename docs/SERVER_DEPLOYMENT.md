# AI Sub Distributed Deployment Guide

This guide covers setting up `ai-sub-server` for persistent 24/7 video ingestion and task coordination, alongside connecting remote ephemeral `ai-sub-worker` instances.

---

## Architecture Overview

AI Sub separates distributed subtitling into two primary components:

1. **Central Server (`ai-sub-server`)**: Runs on a persistent host (e.g., an Oracle Cloud Free Tier VM). It ingests videos from local watch folders or YouTube feeds, breaks them into segments, coordinates worker task leases, and synchronizes finalized subtitles and intermediate states to Git.
2. **Workers (`ai-sub-worker`)**: Ephemeral, distributed nodes that lease tasks, stream media chunks from the server, execute Gemini multimodal inference and web search, and return structured results back to the server.

### Security & Transport

The server application binds locally to `127.0.0.1:8000` without built-in authentication or encryption. All transport and security responsibilities are offloaded to standard production infrastructure:

- **TLS / SSL Encryption**: Managed via Nginx and Let's Encrypt.
- **Authentication**: Enforced via Nginx HTTP Basic Auth (`.htpasswd`).
- **Brute-Force Protection**: Enforced via Fail2ban monitoring Nginx authentication failures.

---

## 1. Server Deployment (`/home/ubuntu/ai-sub-server`)

### Step 1: Install System Dependencies

Install Python 3, virtual environment tools, Nginx, Certbot, Fail2ban, Apache HTTP utilities, and download tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git nginx fail2ban apache2-utils wget curl
```

### Step 2: Open Network Ports

#### a. Open Ports 80 and 443 in Oracle Cloud Console

Ensure security lists and virtual cloud network (VCN) ingress rules permit TCP traffic on ports `80` and `443`.

#### b. Open Ports 80 and 443 in the Ubuntu Firewall

```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

### Step 3: Set Up Certbot

```bash
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/local/bin/certbot
sudo certbot --nginx # Follow interactive prompts
sudo certbot renew --dry-run # Test renewal automation
```

### Step 4: Configure Nginx

#### a. Create HTTP Basic Authentication Credentials

```bash
sudo htpasswd -c /etc/nginx/.htpasswd worker_user
```

#### b. Create `/etc/nginx/sites-available/ai-sub`

```nginx
server {
    server_name sub.yourdomain.com;
    listen [::]:443 ssl ipv6only=on;
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/sub.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sub.yourdomain.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    auth_basic "AI Sub Server";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    if ($host = sub.yourdomain.com) {
        return 301 https://$host$request_uri;
    }

    listen 80;
    listen [::]:80;
    server_name sub.yourdomain.com;
    return 404;
}
```

#### c. Enable the Configuration

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/ai-sub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 5: Configure Fail2ban for HTTP Authentication Protection

#### a. Create `/etc/fail2ban/jail.d/nginx-http-auth.conf`:

```ini
[nginx-http-auth]
enabled  = true
port     = http,https
logpath  = /var/log/nginx/error.log
maxretry = 5
findtime = 600
bantime  = 3600
```

#### b. Restart Fail2ban to activate brute-force protection:

```bash
sudo systemctl restart fail2ban
sudo fail2ban-client status nginx-http-auth
```

### Step 6: Install Node.js LTS via NVM (For YouTube PO Token Provider)

The default YouTube intake configuration enables `bgutil-ytdlp-pot-provider`, which compiles a local Node.js server using TypeScript (`npm ci && npx tsc`). Install NVM and the Node.js LTS runtime under your `ubuntu` user:

```bash
# 1. Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh | bash

# 2. Source NVM into the current shell session
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 3. Install and set Node.js LTS as the default alias
nvm install --lts
nvm alias default 'lts/*'

# 4. Verify node, npm, and npx are accessible
node -v
npm -v
npx -v
```

### Step 7: Set Up Directory, Python Environment, and Package Installation

Create the server directory, initialize a Python virtual environment, and install `ai-sub` with the `server` extra:

```bash
mkdir -p /home/ubuntu/ai-sub-server
cd /home/ubuntu/ai-sub-server

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "ai-sub[server]"
```

### Step 8: Configure Server and Systemd Service

#### a. Download and Edit Configuration

```bash
wget -O server.yaml https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/server.yaml.example
nano server.yaml
```

#### b. Set Up Systemd Service for Auto-Restart

```bash
wget -O server.sh https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/server.sh
chmod +x server.sh

sudo wget -O /etc/systemd/system/ai-sub-server.service https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/ai-sub-server.service
sudo systemctl daemon-reload
sudo systemctl enable ai-sub-server
sudo systemctl start ai-sub-server
sudo journalctl -u ai-sub-server -f
```

---

## 2. Worker Deployment (`/home/ubuntu/ai-sub-worker`)

Workers are ephemeral and can run on any host with Python 3.11+.

### Step 1: Set Up Directory and Python Environment

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip wget

mkdir -p /home/ubuntu/ai-sub-worker
cd /home/ubuntu/ai-sub-worker

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ai-sub
```

### Step 2: Configure `worker.yaml`

Download the example configuration file and update your server URL, HTTP Basic Authentication credentials, and API keys:

```bash
wget -O worker.yaml https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/worker.yaml.example
nano worker.yaml
```

### Step 3: Configure Auto-Updating Launcher (`worker.sh`)

Download `docs/worker.sh` and make it executable:

```bash
wget -O worker.sh https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/worker.sh
chmod +x worker.sh
```

### Step 4: Scheduled Execution via Crontab

Configure recurring cron tasks with environment variable overrides to schedule model execution windows:

```bash
crontab -e
```

Add scheduled runs:

```cron
# Every 4 hours: Run up to 50 tasks using gemini-3.5-flash-lite, stopping new task claims after 3 hours (180 mins)
7 */4 * * * cd /home/ubuntu/ai-sub-worker && AISUB_WORKER_MODELS='google-gla:gemini-3.5-flash-lite' AISUB_WORKER_MAX_TASKS=50 AISUB_WORKER_MAX_DURATION_MINUTES=180 ./worker.sh > /tmp/ai-sub-worker.log 2>&1

# Daily at 21:00 UTC: Offer gemini-3.8-flash and gemini-3.6-flash during peak availability windows
0 21 * * * cd /home/ubuntu/ai-sub-worker && AISUB_WORKER_MODELS='google-gla:gemini-3.8-flash,google-gla:gemini-3.6-flash' AISUB_WORKER_MAX_DURATION_MINUTES=180 ./worker.sh > /tmp/ai-sub-worker-38f.log 2>&1
```

### Worker Operational Behavior

- When the server task queue is empty, the worker exits cleanly in ~1 second.
- In-flight tasks drain gracefully before exiting when the duration limit is reached.
- Quota exhaustion (HTTP 429) automatically disables the affected model in memory for the remainder of the worker run without disrupting other models or tasks.
