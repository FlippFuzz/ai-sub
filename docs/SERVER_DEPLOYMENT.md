# AI Sub Server & Remote Worker Deployment Guide

This guide covers setting up `ai-sub-server` for 24/7 orchestration on an Oracle Cloud Free Tier VM (2 OCPU, 12 GB RAM, 100 GB Disk) and connecting remote ephemeral `ai-sub-worker` instances.

---

## 1. Server Configuration (`server.yaml`)

Both the server and worker use strict YAML configuration files without hidden environment variables or CLI flags.

Place `server.yaml` (or `<hostname>.server.yaml`) in your server workspace directory (e.g. `/home/ubuntu/ai-sub-server/server.yaml`):

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  work_dir: "./work"
  heartbeat_timeout_seconds: 300
  rescan_interval_minutes: 15

git:
  repo_url: "https://github.com/YourUsername/subtitles-archive.git"
  branch: "main"
  pat: "ghp_your_personal_access_token"

sources:
  watch_folders:
    - "/home/ubuntu/videos/intake"
  youtube:
    - "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"
    - "https://www.youtube.com/@ChannelName/videos"
  proxy: "http://127.0.0.1:40000"
  cookie_file: "/home/ubuntu/ai-sub-server/cookie.txt"

pipelines:
  - name: "combo_38f"
    summary_model: "google-gla:gemini-3.5-flash-lite"
    subtitles_model: "google-gla:gemini-3.8-flash"
    lyrics_model: "google-gla:gemini-3.5-flash-lite"

  - name: "combo_36f"
    summary_model: "google-gla:gemini-3.5-flash-lite"
    subtitles_model: "google-gla:gemini-3.6-flash"
    lyrics_model: "google-gla:gemini-3.5-flash-lite"

  - name: "combo_35l"
    summary_model: "google-gla:gemini-3.5-flash-lite"
    subtitles_model: "google-gla:gemini-3.5-flash-lite"
    lyrics_model: "google-gla:gemini-3.5-flash-lite"
```

---

## 2. Nginx Reverse Proxy with Basic Auth & SSL

### Create HTTP Basic Auth Credentials

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd worker_user
```

### Nginx Virtual Host (`/etc/nginx/sites-available/ai-sub`)

```nginx
server {
    listen 80;
    server_name sub.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name sub.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/sub.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sub.yourdomain.com/privkey.pem;

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

Enable and reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/ai-sub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 3. Running the Server (Systemd Service)

Create `/etc/systemd/system/ai-sub-server.service`:

```ini
[Unit]
Description=AI Sub Central Server Coordinator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-sub-server
Environment="PATH=/home/ubuntu/ai-sub-server/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/ai-sub-server/venv/bin/ai-sub-server server.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-sub-server
sudo systemctl start ai-sub-server
sudo journalctl -u ai-sub-server -f
```

---

## 4. Remote Worker Setup (Ephemeral / Multi-Server / Cron)

Workers only need Python 3.11+ and the `ai-sub` package.

### Configuration (`worker.yaml` or `<servername>.worker.yaml`)

Copy `docs/worker.yaml.example` to `worker.yaml` (or `<servername>.worker.yaml`):

```yaml
server_url: "https://sub.yourdomain.com"
server_auth: "worker_user:your_secure_password"
worker_id: "oracle-e2-worker-1"
concurrency: 2

gemini_key: "AIzaSy..."

models:
  - "google-gla:gemini-3.8-flash"
  - "google-gla:gemini-3.6-flash"
  - "google-gla:gemini-3.5-flash-lite"

search:
  tools:
    - "tinyfish"
    - "lrclib"
    - "duckduckgo"
  tinyfish_key: "tf-..."
```

### File Resolution Order

When running `ai-sub-worker`, the configuration is discovered in this exact sequence:

1. Positional argument: `ai-sub-worker sub.yourdomain.com.worker.yaml`
2. `<hostname>.worker.yaml`
3. `worker.yaml`

### Running the Worker

#### Manual Invocation:

```bash
# Auto-discovers <hostname>.worker.yaml or worker.yaml
ai-sub-worker

# Or explicitly specify a server config file
ai-sub-worker sub.yourdomain.com.worker.yaml
```

#### Automated Crontab (e.g. Run Every 15 Minutes):

```cron
*/15 * * * * cd /home/ubuntu/worker && ./venv/bin/ai-sub-worker sub.yourdomain.com.worker.yaml >> /tmp/ai-sub-worker.log 2>&1
```

- When the server queue is empty, all slots exit cleanly in ~1 second.
- Processes multiple video chunks in parallel based on `concurrency: 2`.
- When daily quotas are exhausted (HTTP 429), the worker records the event in `.ai-sub-quota.json` (resetting at 00:00 Pacific Time), drops that model from claiming, and continues running tasks for unexhausted models.
- While processing, each active task runs its own independent background heartbeat every 30 seconds to maintain its lease.
