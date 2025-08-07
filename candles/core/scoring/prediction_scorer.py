# Standard Lib
import math
from typing import Any
import bittensor

# Local
from ..data import CandlePrediction, CandleColor
from .base import BaseScorer, ScoringResult
from ...prices.client import PriceClient


class PredictionScorer(BaseScorer):
    """Scores predictions based on color accuracy, price proximity, and confidence."""

    def __init__(self, price_client: PriceClient, symbol: str = "TAO-USD"):
        """Initialize the prediction scorer.

        Args:
            price_client: Client for fetching price data
            symbol: Trading symbol to score predictions for
        """
        self.price_client = price_client
        self.symbol = symbol

    async def score_prediction(self, prediction: CandlePrediction, actual_data: dict[str, Any] | None = None) -> ScoringResult:
        """Score a prediction against actual market data.

        Args:
            prediction: The prediction to score
            actual_data: Optional actual data (if not provided, will fetch from price client)

        Returns:
            ScoringResult: The scoring result
        """
        # Get actual market data
        # you'll want to later optimise this to where you're not making extraneous double calls
        actual_ohlc = await self.price_client.get_price_by_interval(self.symbol, prediction.interval_id)

        # Calculate color accuracy score
        color_score = self._calculate_color_score(prediction.color, actual_ohlc.color)

        # Calculate price proximity score
        price_score = self._calculate_price_score(
            float(prediction.price),
            actual_ohlc.close
        )

        # Get confidence weight
        confidence_weight = float(prediction.confidence) if prediction.confidence else 0.5

        # Calculate final weighted score
        final_score = self._calculate_final_score(
            color_score,
            price_score,
            confidence_weight
        )
        bittensor.logging.info(f"[yellow]Scoring result: {final_score}[/yellow]")
        return ScoringResult(
            prediction_id=prediction.prediction_id,
            miner_uid=prediction.miner_uid or 0,
            interval_id=prediction.interval_id,
            color_score=color_score,
            price_score=price_score,
            confidence_weight=confidence_weight,
            final_score=final_score,
            actual_color=actual_ohlc.color.value,
            actual_price=actual_ohlc.close
        )

    def _calculate_color_score(self, predicted_color: CandleColor, actual_color: CandleColor) -> float:
        """Calculate score based on color prediction accuracy.

        Args:
            predicted_color: The predicted candle color
            actual_color: The actual candle color

        Returns:
            float: Score between 0.0 and 1.0
        """
        bittensor.logging.debug(f"Predicted color: {predicted_color}, Actual color: {actual_color}")
        return 1.0 if predicted_color == actual_color else 0.0

    def _calculate_price_score(self, predicted_price: float, actual_price: float) -> float:
        """Calculate score based on price prediction accuracy using percentage error.

        Args:
            predicted_price: The predicted price
            actual_price: The actual price

        Returns:
            float: Score between 0.0 and 1.0 (higher is better)
        """
        if actual_price == 0:
            return 0.0

        # Calculate percentage error
        bittensor.logging.debug(f"Predicted price: {predicted_price}, Actual price: {actual_price}")
        percentage_error = abs(predicted_price - actual_price) / actual_price
        # Convert to score using exponential decay
        # This gives high scores for low errors and rapidly decreases as error increases
        score = math.exp(-percentage_error * 10)  # 10 is a scaling factor
        return min(1.0, max(0.0, score))

    def _calculate_final_score(self, color_score: float, price_score: float, confidence: float) -> float:
        """Calculate the final weighted score.

        Args:
            color_score: Score for color prediction accuracy
            price_score: Score for price prediction accuracy
            confidence: Confidence level of the prediction

        Returns:
            float: Final weighted score
        """
        # Weight color prediction more heavily (60%) than price proximity (40%)
        base_score = (color_score * 0.6) + (price_score * 0.4)

        # Apply confidence weighting
        # Higher confidence predictions get more weight when correct,
        # but are penalized more when wrong
        if base_score > 0.5:
            # Good prediction - boost by confidence
            weighted_score = base_score + (confidence - 0.5) * 0.2
        else:
            # Poor prediction - penalize by confidence
            weighted_score = base_score - (confidence - 0.5) * 0.2

        return min(1.0, max(0.0, weighted_score))
