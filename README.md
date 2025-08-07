<div align="center">
  <img src="candles_sn31.png" alt="Candles Subnet" width="250" />
</div>

# Candles Subnet

A decentralized cryptocurrency candle prediction network built on Bittensor. Miners compete to predict cryptocurrency price movements (candle colors and values), while validators score predictions against real market data.



## 🏗️ How It Works

- **Miners**: Generate predictions for cryptocurrency candles (price direction and closing values)
- **Validators**: Request predictions from miners and score them against actual market data
- **Rewards**: Miners are rewarded based on prediction accuracy (both color and price proximity)

### Prediction Types
- **Hourly**: Next hour candle predictions
- **Daily**: Next day candle predictions
- **Weekly**: Next week candle predictions

## 📦 Installation

### Prerequisites

1. **Install Rust** (required for Bittensor):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Install Astral uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.local/bin/env
   ```

3. **Python 3.12+** is required

### Quick Miner Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/thealligatorking/Candles
   cd Candles
   ```

2. **Use the automated setup script**:
   ```bash
   ./setup_miner.sh <wallet_name> <hotkey_name>
   ```

   Example:
   ```bash
   ./setup_miner.sh my_wallet my_hotkey
   ```

   This script will:
   - Install system dependencies
   - Set up Python environment with uv
   - Install project dependencies
   - Create a custom miner script for your wallet/hotkey
   - Run tests to verify installation
> [!NOTE]  
   > If you have issues with running your miner after using the setup script it is likely because you have an ipv6 address. You can fix this by running the following command:
   > ```bash
   > curl -4 ifconfig.me
   > ```
   > Update your external_ip in the miner script to use the ipv4 address (if you have an ipv6 address)

3. **Start your miner**:
   ```bash
   ./miner_<wallet_name>_<hotkey_name>
   ```

   **Run with PM2** (recommended for production):
   ```bash
   # Install PM2 globally
   npm install -g pm2

   # Start miner with PM2 using ecosystem config
   pm2 start ecosystem.config.js

   # Monitor running processes
   pm2 status
   pm2 logs
   ```

### Manual Installation

If you prefer manual setup:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run tests**:
   ```bash
   uv run pytest
   ```

3. **Start miner**:
   ```bash
   ./miner
   ```

4. **Start validator** (if running a validator - still in development):
   ```bash
   ./validator
   ```

## 🔧 Development

### Linting
```bash
./scripts/ruff.sh
```

### Testing
```bash
uv run pytest tests/ --asyncio-mode=auto
```

### Environment Variables

For validators, set:
```bash
export COINDESK_API_KEY="your_api_key_here"
```

For miners using prediction files, optionally set:
```bash
export PREDICTIONS_FILE_PATH="/path/to/your/predictions.csv"
```

## 🔮 Custom Prediction Files

Miners can provide their own predictions instead of using random generation by placing CSV files in specific locations. The miner will automatically detect and use these files when available.

### 🌐 Web Interface for Predictions Generation

Create and download prediction CSV files easily using the web interface at:
**[candlestao.com](https://www.candlestao.com/)**

This web tool allows you to generate properly formatted prediction files without manually creating CSV files.

### File Placement Options

**Option 1: Default Directory (Recommended)**
```bash
# Create the candles data directory
mkdir -p ~/.candles/data/

# Place your prediction files here with these exact names:
~/.candles/data/hourly_predictions.csv   # For hourly predictions
~/.candles/data/daily_predictions.csv    # For daily predictions
~/.candles/data/weekly_predictions.csv   # For weekly predictions
```

**Option 2: Custom Path via Environment Variable**
```bash
# Set a custom file path (applies to all intervals)
export PREDICTIONS_FILE_PATH="/path/to/your/predictions.csv"
```

### CSV File Format

Your prediction files must follow this exact format:

```csv
timestamp,color,confidence,price
1704067200,red,0.85,45.50
1704070800,green,0.92,46.20
1704074400,red,0.78,44.90
```

**Column Requirements:**
- `timestamp`: Unix timestamp for the prediction interval start time
- `color`: Predicted candle color (`red` or `green`)
- `confidence`: Confidence level (0.0 to 1.0)
- `price`: Predicted closing price (decimal number)

### How It Works

1. **File Discovery**: The miner searches for prediction files in this order:
   - `~/.candles/data/[interval]_predictions.csv` (e.g., `hourly_predictions.csv`)
   - File specified by `PREDICTIONS_FILE_PATH` environment variable
   - If no files found, generates random predictions

2. **Prediction Matching**: When a validator requests a prediction:
   - Miner loads predictions from the appropriate file
   - Filters predictions to only include future intervals
   - Matches the exact `interval_id` requested by the validator
   - Returns the matching prediction if found

3. **Fallback Behavior**: If no prediction file is found or no matching prediction exists, the miner automatically generates random predictions to ensure continuous operation.

### Example Setup

```bash
# Create prediction directory
mkdir -p ~/.candles/data/

# Add your predictions to the directory
cp hourly_predictions.csv ~/.candles/data/

# Your miner will now use these predictions automatically
./miner_<wallet_name>_<hotkey_name>
```

## 🌐 Network Information

   | Network  | netuid |
   |----------|--------|
   | mainnet  | 31     |
   | testnet  | 357    |


**Symbol**: TAO-USD (primary trading pair)

## 📚 Documentation

More comprehensive documentation is coming soon. For now, see:
- `candles/` - Source code with inline documentation

## 💬 Community

Join our Discord server for support, updates, and community discussions:

**[🕯️ Candles Discord Server](https://discord.gg/MTZP7CJQ)**

## 🤝 Contributing

This project is in active development. Contributions, feedback, and bug reports are welcome!

## 📄 License

MIT License - see LICENSE file for details.

---

**🔥 Ready to predict the future of crypto? Join our subnet and start mining!**
