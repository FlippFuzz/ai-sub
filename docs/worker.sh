#!/usr/bin/env bash
set -e

# Change into worker directory
cd /home/ubuntu/ai-sub-worker

# Activate virtual environment
source venv/bin/activate

# Auto-update ai-sub package
pip install --upgrade ai-sub --quiet

# Execute worker passing through any environment variables and arguments
exec ai-sub-worker "$@"