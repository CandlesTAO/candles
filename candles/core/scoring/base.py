# Standard Lib
from abc import ABC, abstractmethod
from typing import Any

# Third Party
from pydantic import BaseModel

# Local
from ..data import CandlePrediction


class ScoringResult(BaseModel):
    """Result of scoring a prediction."""

    prediction_id: int
    miner_uid: int
    interval_id: str
    color_score: float
    price_score: float
    confidence_weight: float
    final_score: float
    actual_color: str
    actual_price: float


class BaseScorer(ABC):
    """Base class for prediction scoring."""

    @abstractmethod
    async def score_prediction(self, prediction: CandlePrediction, actual_data: dict[str, Any] | None = None) -> ScoringResult:
        """Score a prediction against actual market data.

        Args:
            prediction: The prediction to score
            actual_data: Dictionary containing actual market data

        Returns:
            ScoringResult: The scoring result
        """
        pass
