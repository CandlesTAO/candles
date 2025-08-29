# Validator Auto-Update Setup

This guide shows how to enable automatic updates for your validator using environment variables.

## Auto-Update Environment Variables

Add these environment variables to your `.env` file to enable auto-update functionality:

```bash
# Enable automatic updates
export ENABLE_AUTO_UPDATER="true"

# Set update check interval (in seconds, default: 43200 = 12 hours)
export AUTO_UPDATE_INTERVAL="3600"

# Set the branch to monitor for updates (default: main)
export AUTO_UPDATE_BRANCH="main"

# Set the validator configuration file path
export VALIDATOR_CONFIG_FILE="validator_config.json"
```

## Example .env Configuration

```bash
# Copy the example environment file
cp example_dotenv.sh .env

# Add these auto-update variables to your .env:
export ENABLE_AUTO_UPDATER="true"
export AUTO_UPDATE_INTERVAL="3600"
export AUTO_UPDATE_BRANCH="main"
export VALIDATOR_CONFIG_FILE="validator_config.json"

# Your other validator configuration
export WALLET_NAME="my_validator"
export WALLET_HOTKEY="my_hotkey"
export COINDESK_API_KEY="your_api_key"
```

That's it! Your validator will now automatically check for updates and restart with the latest version.
