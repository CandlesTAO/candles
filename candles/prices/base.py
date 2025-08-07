
from typing import Protocol

from ..core.data import CandleColor

class BasePriceClient(Protocol):


    def __init__(self, api_key: str, provider: str, *args, **kwargs):
        self.api_key = api_key
        self.provider = provider


    def get_price_by_interval(self, symbol: str, interval: str) -> list[float]:
        ...

    def get_candle_color_by_price_history(self, price_history: list[float]) -> CandleColor:
        ...
