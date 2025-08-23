#!/bin/bash

# Start script for Candles Validator
# This script is designed to work with PM2 and the auto-updater

set -e


# Check if restart is requested (auto-updater functionality)
if [ -f ".validator_restart" ]; then
    echo "Restart flag detected, cleaning up..."
    rm -f .validator_restart
    echo "Restart flag cleaned up, PM2 will restart the validator automatically"
fi

# Source environment variables if .env file exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check required environment variables
if [ -z "$COINDESK_API_KEY" ]; then
    echo "Warning: COINDESK_API_KEY not set. Some features may not work properly."
fi

echo "Starting Candles Validator..."
echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "UV version: $(uv --version)"

# Start the validator using uv
#exec uv run python -m candles.validator.validator
uv run -m candles.validator.validator --netuid 31  --wallet.name "$WALLET_NAME" --wallet.hotkey "$WALLET_HOTKEY" --logging.trace "$@"
