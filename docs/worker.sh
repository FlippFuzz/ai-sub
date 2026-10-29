#!/usr/bin/env bash
set -e

# Change into worker directory
cd /home/ubuntu/ai-sub-worker

# Activate virtual environment
source venv/bin/activate

# Auto-update ai-sub package under a file lock to prevent concurrent pip race conditions
exec 200>/tmp/ai-sub-pip.lock
flock 200
pip install --upgrade ai-sub --quiet
flock -u 200
exec 200>&-

# Execute worker passing through any environment variables and arguments
exec ai-sub-worker "$@"