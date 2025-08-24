#!/bin/bash

# Example environment configuration for the Candles Validator with auto-update
# Copy this file to .env and modify the values as needed


# Custom Wallet Configuration
export WALLET_NAME="set_this_to_your_wallet_name"
export WALLET_HOTKEY="set_this_to_your_hotkey_name"

# Log Level
export LOG_LEVEL="info"  # Options: trace, debug, info, warning, error


# Candletao
CANDLETAO_BEARER_TOKEN=
CANDLETAO_DOMAIN=

# CoinDesk
COINDESK_API_KEY=

# Update Check Interval (in seconds)
# 3600 = 1 hour
export AUTO_UPDATE_INTERVAL="3600"

# Auto-Update Branch
# main = production branch
# testnet = testnet branch
export AUTO_UPDATE_BRANCH="main"

# Configuration File Path
export VALIDATOR_CONFIG_FILE="validator_config.json"

echo "Environment variables set for Candles Validator with auto-update"
echo "Auto-Update Branch: $AUTO_UPDATE_BRANCH"

echo "Check Interval: $AUTO_UPDATE_INTERVAL seconds"
echo "Config File: $VALIDATOR_CONFIG_FILE"
