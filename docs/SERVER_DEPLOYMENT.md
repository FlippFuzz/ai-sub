# AI Sub Server & Remote Worker Deployment Guide

This guide covers setting up `ai-sub-server` for 24/7 orchestration on an Oracle Cloud Free Tier VM (2 OCPU, 12 GB RAM, 100 GB Disk) and connecting remote ephemeral `ai-sub-worker` instances.

---

## 1. Server Configuration (`server.yaml`)

Place `server.yaml` in your server workspace directory (e.g. `/home/ubuntu/ai-sub-server/server.yaml`):

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
  pat_env: "GITHUB_PAT"

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

## 3. Fail2ban Setup for Nginx Basic Auth

Create `/etc/fail2ban/jail.d/nginx-auth.local`:

```ini
[nginx-http-auth]
enabled = true
port    = http,https
logpath = %(nginx_error_log)s
maxretry = 5
bantime  = 86400
findtime = 600
```

Restart Fail2ban:

```bash
sudo systemctl restart fail2ban
```

---

## 4. Running the Server (Systemd Service)

Create `/etc/systemd/system/ai-sub-server.service`:

```ini
[Unit]
Description=AI Sub Central Server Coordinator
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-sub-server
Environment="GITHUB_PAT=your_github_token_here"
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

## 5. Remote Worker Setup (Ephemeral / Cron)

Workers only need Python 3.11+ and the `ai-sub` package.

### Worker Environment (`.env` or export):

```bash
export GEMINI_API_KEY="your_gemini_key_for_this_worker"
export TINYFISH_API_KEY="your_tinyfish_key_for_this_worker"
```

### Manual Run:

```bash
ai-sub-worker \
  --server-url "https://sub.yourdomain.com" \
  --server-auth "worker_user:your_secure_password"
```

### Automated Crontab (e.g. Run Every 15 Minutes):

```cron
*/15 * * * * cd /home/ubuntu/worker && export GEMINI_API_KEY="AIzaSy..." && export TINYFISH_API_KEY="tf-..." && ./venv/bin/ai-sub-worker --server-url "https://sub.yourdomain.com" --server-auth "worker_user:password" >> /tmp/ai-sub-worker.log 2>&1
```

- When the queue is empty, the worker exits cleanly in ~1 second.
- When daily quotas are exhausted (429), it records the event in `.ai-sub-quota.json` (resetting at 00:00 Pacific Time) and exits cleanly.
- While processing, it sends a heartbeat ping every 30 seconds to hold the lease.
