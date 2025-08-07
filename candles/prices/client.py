# Standard Lib
from enum import StrEnum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta, timezone

# Third Party
import aiohttp

# Bittensor
import bittensor
# Local
from .base import BasePriceClient
from .schemas import CoinDeskResponseOHLC

@dataclass
class APIConfig:
    base_url: str
    api_key: Optional[str] = None
    api_key_header: Optional[str] = None
    api_params: Optional[dict] = None

class PriceProvider(StrEnum):
    COINDESK = "coindesk"

    @property
    def config(self) -> APIConfig:
        if self == PriceProvider.COINDESK:
            return APIConfig(
                base_url="https://data-api.coindesk.com/index/cc/v1/historical",
                api_params={
                    "market": "cadli",
                    "instrument": "",
                    "limit": 1,
                    "aggregate": 1,
                    "fill": "true",
                    "apply_mapping": "true",
                    "response_format": "JSON",
                    "to_ts": "",
                    "groups": "OHLC",
                }
            )
        raise ValueError(f"Unknown provider: {self}")

class PriceClient(BasePriceClient):

    def __init__(self, api_key: str, provider: str, *args, **kwargs):
        super().__init__(api_key, provider, *args, **kwargs)
        self.provider_enum = PriceProvider(provider)
        self.api_config = self.provider_enum.config

    async def get_price_by_interval(self, symbol: str, interval_id: str) -> CoinDeskResponseOHLC | None:
        if self.provider_enum == PriceProvider.COINDESK:
            timestamp, interval = interval_id.split("::")

            match interval:
                case "hourly":
                    coindesk_interval = "hours"
                case "daily":
                    coindesk_interval = "days"
                case "weekly":
                    return await self.get_weekly_candle(symbol, int(timestamp))
                case _:
                    raise ValueError(f"Unsupported interval: {interval}")

            url = f"{self.api_config.base_url}/{coindesk_interval}"
            self.api_config.api_params["api_key"] = self.api_key
            self.api_config.api_params["instrument"] = f"{symbol}"
            self.api_config.api_params["to_ts"] = timestamp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=self.api_config.api_params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return CoinDeskResponseOHLC.parse_response(data["Data"][0])

    async def get_weekly_candle(self, symbol: str, week_start_timestamp: int) -> CoinDeskResponseOHLC:
        if self.provider_enum != PriceProvider.COINDESK:
            raise ValueError("Weekly candle method only supports CoinDesk provider")

        week_start = datetime.fromtimestamp(week_start_timestamp, tz=timezone.utc)

        if week_start.weekday() != 0:
            raise ValueError(f"Week start must be a Monday, got {week_start.strftime('%A')}")
        if week_start.hour != 0 or week_start.minute != 0 or week_start.second != 0:
            raise ValueError("Week start must be at midnight (00:00:00)")

        week_end = week_start + timedelta(days=6, hours=23)

        # Format timestamps for API calls
        week_start_ts = int(week_start.timestamp())
        week_end_ts = int(week_end.timestamp())

        open_interval_id = f"{week_start_ts}::hourly"
        close_interval_id = f"{week_end_ts}::hourly"

        # Fetch both hourly candles
        open_candle = await self.get_price_by_interval(symbol, open_interval_id)
        close_candle = await self.get_price_by_interval(symbol, close_interval_id)
        bittensor.logging.info(f"Open candle: {open_candle}")
        bittensor.logging.info(f"Close candle: {close_candle}")

        from ..core.data import CandleColor
        weekly_color = CandleColor.GREEN if close_candle.close >= open_candle.open else CandleColor.RED

        return CoinDeskResponseOHLC(
            open=open_candle.open,
            close=close_candle.close,
            timestamp=str(week_start_ts),
            color=weekly_color
        )

