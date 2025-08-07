# Standard Lib

# Third Party
from pydantic import BaseModel

# Local
from ..core.data import CandleColor

class CoinDeskResponseOHLC(BaseModel):
    open: float
    close: float
    timestamp: str
    color: CandleColor


    @classmethod
    def parse_response(cls, data: dict) -> "CoinDeskResponseOHLC":
        """Parse a response dictionary into a CoinDeskResponseOHLC model.

        Args:
            data: Dictionary containing OHLC data with keys like 'OPEN', 'CLOSE', 'TIMESTAMP'

        Returns:
            ResponseOHLC: Parsed model instance
        """
        return cls(
            open=float(data['OPEN']),
            close=float(data['CLOSE']),
            timestamp=str(data['TIMESTAMP']),
            color=CandleColor.GREEN if float(data['CLOSE']) >= float(data['OPEN']) else CandleColor.RED
        )
