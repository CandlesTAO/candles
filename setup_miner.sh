#!/bin/bash

# Candles Subnet Miner Setup Script
# Usage: ./setup_miner.sh <wallet_name> <hotkey_name>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    print_error "Usage: $0 <wallet_name> <hotkey_name>"
    print_error "Example: $0 my_wallet my_hotkey"
    exit 1
fi

WALLET_NAME="$1"
HOTKEY_NAME="$2"

print_status "Setting up Candles Subnet Miner with wallet: $WALLET_NAME, hotkey: $HOTKEY_NAME"

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended. Consider using a regular user account."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
print_status "Updating system packages..."
sudo apt-get update -qq

# Install required system dependencies
print_status "Installing system dependencies..."
sudo apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    tk-dev \
    tcl-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    print_status "Installing Python 3.12..."

    # Add deadsnakes PPA for newer Python versions
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

    # Create symlink if python3.12 is not the default
    if ! command -v python3.12 &> /dev/null; then
        print_error "Failed to install Python 3.12"
        exit 1
    fi

    print_status "Python 3.12 installed successfully"
fi

# Install or update uv
print_status "Installing/updating uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    print_status "uv already installed, updating..."
    uv self update
fi

# Verify uv installation
if ! command -v uv &> /dev/null; then
    print_error "uv installation failed. Please install manually: https://github.com/astral-sh/uv"
    exit 1
fi

print_status "uv version: $(uv --version)"

# Get external IP address
print_status "Detecting external IP address..."
EXTERNAL_IP=""

# Function to validate if response is a valid IP address
is_valid_ip() {
    local ip=$1
    # Check basic format first
    if [[ ! $ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        return 1
    fi

    # Validate each octet is in range 0-255
    local IFS='.'
    local octets
    read -ra octets <<< "$ip"

    # Ensure we have exactly 4 octets
    if [ ${#octets[@]} -ne 4 ]; then
        return 1
    fi

    for octet in "${octets[@]}"; do
        # Check if octet is numeric
        if ! [[ "$octet" =~ ^[0-9]+$ ]]; then
            return 1
        fi
        # Check if octet is in valid range (0-255)
        if [ "$octet" -lt 0 ] || [ "$octet" -gt 255 ] 2>/dev/null; then
            return 1
        fi
    done

    return 0
}

# Function to get IP from a service
get_ip_from_service() {
    local url=$1
    local result=""

    if command -v curl &> /dev/null; then
        result=$(curl -s -4 --max-time 5 "$url" 2>/dev/null | tr -d '\n\r' | grep -oE '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' | head -1)
    elif command -v wget &> /dev/null; then
        result=$(wget -qO- -4 --timeout=5 "$url" 2>/dev/null | tr -d '\n\r' | grep -oE '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' | head -1)
    fi

    if is_valid_ip "$result"; then
        echo "$result"
        return 0
    fi
    return 1
}

# Try multiple IP detection services
if command -v curl &> /dev/null || command -v wget &> /dev/null; then
    print_status "Trying IP detection services..."

    # List of services to try
    services=(
        "https://api.ipify.org"
        "https://icanhazip.com"
        "https://ifconfig.co/ip"
        "https://api.myip.com"
        "https://ipinfo.io/ip"
        "https://ifconfig.me"
        "https://checkip.amazonaws.com"
        "https://ipecho.net/plain"
    )

    for service in "${services[@]}"; do
        result=$(get_ip_from_service "$service")
        if is_valid_ip "$result"; then
            EXTERNAL_IP="$result"
            print_status "Successfully detected IP from $service"
            break
        fi
    done
fi

# Fallback to local IP if external IP detection fails
if [ -z "$EXTERNAL_IP" ]; then
    print_warning "Could not detect external IP, trying local IP"
    local_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || echo "")
    if is_valid_ip "$local_ip"; then
        EXTERNAL_IP="$local_ip"
    else
        print_error "Failed to detect any valid IP address"
        exit 1
    fi
fi

print_status "Using IP address: $EXTERNAL_IP"

# Install project dependencies
print_status "Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    uv sync
else
    print_error "pyproject.toml not found. Make sure you're in the correct directory."
    exit 1
fi

# Create custom miner script
CUSTOM_MINER_SCRIPT="./miner_${WALLET_NAME}_${HOTKEY_NAME}"

print_status "Creating custom miner script: $CUSTOM_MINER_SCRIPT"

cat > "$CUSTOM_MINER_SCRIPT" << EOF
#!/usr/bin/env bash
# Custom miner script for wallet: $WALLET_NAME, hotkey: $HOTKEY_NAME
# Generated on: $(date)

uv run candles/miner/miner.py \\
    --axon.external_ip $EXTERNAL_IP \\
    --netuid 31 \\
    --axon.port 8092 \\
    --wallet.name $WALLET_NAME \\
    --wallet.hotkey $HOTKEY_NAME \\
    --blacklist.validator_min_stake 0 \\
    --logging.info \\
    --neuron.epoch_length 50 \\
    "\$@"
EOF

# Make the script executable
chmod +x "$CUSTOM_MINER_SCRIPT"

# Update CLAUDE.md with the new script information
if [ -f "CLAUDE.md" ]; then
    print_status "Updating CLAUDE.md with custom miner script information..."

    # Add a section about custom miner scripts if it doesn't exist
    if ! grep -q "Custom Miner Scripts" CLAUDE.md; then
        cat >> CLAUDE.md << EOF

### Custom Miner Scripts
Custom miner scripts are generated by setup_miner.sh with user-specific wallet and hotkey configurations:
\`\`\`bash
# Run custom miner (generated by setup script)
./${CUSTOM_MINER_SCRIPT##*/}
\`\`\`
EOF
    fi
fi

# Run tests to verify installation
print_status "Running tests to verify installation..."
if uv run pytest tests/ -v --tb=short; then
    print_status "All tests passed!"
else
    print_warning "Some tests failed, but this might be expected in certain environments"
fi

# Final instructions
print_status "Setup completed successfully!"
echo
echo "==================================================================================="
echo -e "${GREEN}Miner setup complete!${NC}"
echo
echo "Custom miner script created: $CUSTOM_MINER_SCRIPT"
echo "Wallet name: $WALLET_NAME"
echo "Hotkey name: $HOTKEY_NAME"
echo "External IP: $EXTERNAL_IP"
echo
echo "To start your miner:"
echo "  $CUSTOM_MINER_SCRIPT"
echo
echo "To check logs in real-time:"
echo "  $CUSTOM_MINER_SCRIPT | tee miner.log"
echo
echo "To run in background:"
echo "  nohup $CUSTOM_MINER_SCRIPT > miner.log 2>&1 &"
echo
echo "To check if your miner is running:"
echo "  ps aux | grep miner"
echo
echo "Additional options can be passed to the miner script:"
echo "  $CUSTOM_MINER_SCRIPT --help"
echo "==================================================================================="
