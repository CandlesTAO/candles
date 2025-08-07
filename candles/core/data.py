from decimal import Decimal
from enum import StrEnum
from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
)


class CandlesBaseModel(BaseModel):

    class Config:
        use_enum_values = True


class CandleColor(StrEnum):
    """
    The color of the candle.
    """
    RED = "red"
    GREEN = "green"

class TimeInterval(StrEnum):
    """
    The interval of the prediction.
    """
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"



class Prediction(CandlesBaseModel):
    """
    Base class for all predictions.
    """

    prediction_id: PositiveInt = Field(
        default=None,
        description="Unique ID that represents a predication. e.g. 1"
    )

    miner_uid: Optional[NonNegativeInt] = Field(
        default=None,
        description="Unique ID that represents a miner. e.g. 1"
    )

    hotkey: Optional[str] = Field(
        default=None,
        description="A unique identifier for the miner. e.g. 5Hq9123456789012345678901234567890123456789"
    )

    prediction_date: Optional[datetime] = Field(
        default=None,
        description="The datetime the prediction was made. e.g. 2021-01-01 00:00:00"
    )

    interval_id: str = Field(description="Unique ID that represents a interval. Format: timestamp::interval, e.g. 1715000000::HOURLY")

    is_closed: bool = False
    closed_date: Optional[datetime] = Field(default=None, description="The datetime the prediction was closed. e.g. 2021-01-01 00:00:00")



class CandlePrediction(Prediction):
    """
    A prediction of the color of a candle and the price at the end of the interval.
    """
    interval: TimeInterval = Field(description="The interval of the prediction. e.g. HOURLY, DAILY, WEEKLY")

    color: Optional[CandleColor] = Field(default=None, description="The color of the candle. e.g. RED, GREEN")
    price: Optional[Decimal] = Field(default=None, description="The price of the candle. e.g. 100.00")
    confidence: Optional[Decimal] = Field(default=None, description="The confidence of the prediction. e.g. 0.95")
