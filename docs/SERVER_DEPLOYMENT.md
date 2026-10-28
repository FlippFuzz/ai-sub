# AI Sub Distributed Deployment Guide

This guide covers setting up `ai-sub-server` for 24/7 video ingestion and task coordination, along with connecting remote ephemeral `ai-sub-worker` instances.

---

## Architecture Overview

AI Sub separates subtitling into two main components:

1. **Central Server (`ai-sub-server`)**: Runs on a persistent host (e.g., an Oracle Cloud Free Tier VM). It ingests videos from local watch folders or YouTube feeds, breaks them into segments, coordinates worker task leases, and pushes finalized subtitles and intermediate state to Git.
2. **Workers (`ai-sub-worker`)**: Ephemeral, distributed nodes that claim tasks, stream media chunks from the server, run Gemini multimodal inference and web search, and return structured results to the server.

### Security & Transport

The server application binds locally to `127.0.0.1:8000` without built-in authentication or encryption. All security responsibilities are offloaded to standard production infrastructure:

- **TLS / SSL Encryption**: Managed via Nginx and Let's Encrypt.
- **Authentication**: Enforced via Nginx HTTP Basic Auth (`.htpasswd`).
- **Brute-Force Protection**: Enforced via Fail2ban monitoring Nginx auth failures.

---

## 1. Server Deployment (`/home/ubuntu/ai-sub-server`)

### Step 1: Install System Dependencies

Install Python 3, virtual environment tools, Nginx, Certbot, Fail2ban, and Apache utilities:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git nginx certbot python3-certbot-nginx fail2ban apache2-utils wget
```

### Step 2: Set Up Directory and Python Environment

Create the server directory and set up a Python virtual environment:

```bash
mkdir -p /home/ubuntu/ai-sub-server
cd /home/ubuntu/ai-sub-server

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ai-sub
```

### Step 3: Configure `server.yaml`

Download the example configuration file and configure your Git archive, intake paths, and pipelines:

```bash
wget -O server.yaml https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/server.yaml.example
nano server.yaml
```

### Step 4: Configure Auto-Updating Launcher (`server.sh`)

Download `server.sh` and make it executable:

```bash
wget -O server.sh https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/server.sh
chmod +x server.sh
```

### Step 5: Configure Nginx with Basic Auth

Generate credentials for your worker nodes:

```bash
sudo htpasswd -c /etc/nginx/.htpasswd worker_user
```

Create `/etc/nginx/sites-available/ai-sub`:

```nginx
server {
    listen 80;
    server_name sub.yourdomain.com;

    client_max_body_size 100M;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    auth_basic "AI Sub Coordinator";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/ai-sub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 6: Acquire Let's Encrypt SSL Certificate

Use Certbot to configure SSL on your domain:

```bash
sudo certbot --nginx -d sub.yourdomain.com
```

### Step 7: Configure Fail2ban for HTTP Auth Protection

Create `/etc/fail2ban/jail.d/nginx-http-auth.local`:

```ini
[nginx-http-auth]
enabled  = true
port     = http,https
logpath  = /var/log/nginx/error.log
maxretry = 5
findtime = 600
bantime  = 3600
```

Restart Fail2ban to activate brute-force protection:

```bash
sudo systemctl restart fail2ban
sudo fail2ban-client status nginx-http-auth
```

### Step 8: Start Server as a Systemd Service

Download the systemd service file into `/etc/systemd/system/`:

```bash
sudo wget -O /etc/systemd/system/ai-sub-server.service https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/ai-sub-server.service
sudo systemctl daemon-reload
sudo systemctl enable ai-sub-server
sudo systemctl start ai-sub-server
sudo journalctl -u ai-sub-server -f
```

---

## 2. Worker Deployment (`/home/ubuntu/ai-sub-worker`)

Workers are ephemeral and can run on any machine with Python 3.11+.

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

Download the example configuration file and set your server URL, HTTP Basic Auth credentials, and API keys:

```bash
wget -O worker.yaml https://raw.githubusercontent.com/FlippFuzz/ai-sub/main/docs/worker.yaml.example
nano worker.yaml
```

### Step 3: Configure Auto-Updating Launcher (`worker.sh`)

Download `worker.sh` and make it executable:

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
# Every 4 hours: Run up to 50 tasks of 3.5-flash-lite, stopping new claims after 3 hours (180 mins)
7 */4 * * * cd /home/ubuntu/ai-sub-worker && AISUB_WORKER_MODELS='google-gla:gemini-3.5-flash-lite' AISUB_WORKER_MAX_TASKS=50 AISUB_WORKER_MAX_DURATION_MINUTES=180 ./worker.sh > /tmp/ai-sub-worker.log 2>&1

# Daily at 21:00 UTC: Offer gemini-3.8-flash during peak availability window
0 21 * * * cd /home/ubuntu/ai-sub-worker && AISUB_WORKER_MODELS='google-gla:gemini-3.8-flash,google-gla:gemini-3.6-flash' AISUB_WORKER_MAX_DURATION_MINUTES=180 ./worker.sh > /tmp/ai-sub-worker-38f.log 2>&1
```

- When the server queue is empty, the worker exits cleanly in ~1 second.
- In-flight tasks drain gracefully before exiting when the duration limit is reached.
- Quota exhaustion (HTTP 429) automatically disables the affected model until 00:00 Pacific Time in `.ai-sub-quota.json` without failing the entire run.
