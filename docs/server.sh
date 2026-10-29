#!/usr/bin/env bash
set -e

# Change into server directory
cd /home/ubuntu/ai-sub-server

# Load NVM (Node Version Manager) if installed in ~/.nvm
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    set +e
    # shellcheck source=/dev/null
    \. "$NVM_DIR/nvm.sh"
    # Ensure default or current node version is selected and on PATH
    nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1
    set -e
fi

# Activate virtual environment
source venv/bin/activate

# Auto-update ai-sub package
pip install --upgrade "ai-sub[server]" --quiet

# Execute server coordinator passing through any arguments
exec ai-sub-server "$@"