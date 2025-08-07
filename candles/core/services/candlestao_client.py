import os
from typing import Any
import aiohttp
import json
from datetime import datetime

from ..data import CandlesBaseModel


class CandleTAOPredictionSubmission(CandlesBaseModel):
    """Model for prediction submission data matching PredictionDataSchema"""
    prediction_id: int
    miner_uid: int
    hotkey: str
    prediction_date: datetime
    interval_id: str
    is_closed: bool
    closed_date: datetime | None = None
    interval: str
    color: str
    price: str
    confidence: str


class CandleTAOScoreSubmission(CandlesBaseModel):
    """Model for score submission data matching ScoreDataSchema"""
    prediction_id: int
    miner_uid: int
    interval_id: str
    color_score: float
    price_score: float
    confidence_weight: float
    final_score: float
    actual_color: str
    actual_price: float
    timestamp: datetime


class CandleTAOMinerScoreSubmission(CandlesBaseModel):
    """Model for miner score submission data matching MinerScoreUpdateSchema"""
    miner_uid: int
    score: float
    last_scored_prediction_id: int


class CandleTAOClient:
    """Async client for submitting prediction scores to CandleTAO API"""

    def __init__(self):
        self.base_url = self._get_base_url()
        self.bearer_token = os.getenv("CANDLETAO_BEARER_TOKEN")
        self.timeout = aiohttp.ClientTimeout(total=30)

        if not self.bearer_token:
            raise ValueError("CANDLETAO_BEARER_TOKEN environment variable is required")

    def _get_base_url(self) -> str:
        """Construct base URL from environment variables"""
        domain = os.getenv("CANDLETAO_DOMAIN", "localhost")
        port = os.getenv("CANDLETAO_PORT")

        # Handle both IP addresses and domain names
        if domain.startswith("http://") or domain.startswith("https://"):
            base = domain
        else:
            # Default to https for domain names, http for localhost/IPs
            protocol = "http" if domain == "localhost" or domain.replace(".", "").isdigit() else "https"
            base = f"{protocol}://{domain}"

        return (
            f"{base}:{port}"
            if port
            and not domain.startswith("http://")
            and not domain.startswith("https://")
            else base
        )

    @property
    def predictions_endpoint(self) -> str:
        """Get the full predictions endpoint URL"""
        return f"{self.base_url}/api/predictions/"

    @property
    def scores_endpoint(self) -> str:
        """Get the full scores endpoint URL"""
        return f"{self.base_url}/api/predictions/scores"

    @property
    def miner_scores_endpoint(self) -> str:
        """Get the full miner scores endpoint URL"""
        return f"{self.base_url}/api/predictions/miner-scores"

    async def submit_predictions(self, predictions: list[CandleTAOPredictionSubmission]) -> dict[str, Any]:
        """
        Submit a batch of predictions to the CandleTAO API

        Args:
            predictions: List of prediction submissions to send

        Returns:
            API response as dictionary

        Raises:
            aiohttp.ClientError: On HTTP client errors
            ValueError: On invalid response data
        """
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

        # Convert predictions to dict format for JSON serialization
        # The API expects each prediction to be a separate object in the data list
        payload = {
            "data": [prediction.model_dump(mode='json') for prediction in predictions]
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.predictions_endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API request failed: {error_text}"
                    )

                try:
                    return await response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response from API: {e}")

    async def submit_scores(self, scores: list[CandleTAOScoreSubmission]) -> dict[str, Any]:
        """
        Submit a batch of prediction scores to the CandleTAO API

        Args:
            scores: List of score submissions to send

        Returns:
            API response as dictionary

        Raises:
            aiohttp.ClientError: On HTTP client errors
            ValueError: On invalid response data
        """
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

        # Convert scores to dict format for JSON serialization
        # The API expects each score to be a separate object in the data list
        payload = {
            "data": [
                {
                    **score.model_dump(mode='json'),
                    "submission_timestamp": [datetime.utcnow().isoformat()]
                }
                for score in scores
            ]
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.scores_endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API request failed: {error_text}"
                    )

                try:
                    return await response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response from API: {e}")

    async def submit_single_score(self, score: CandleTAOScoreSubmission) -> dict[str, Any]:
        """
        Submit a single prediction score

        Args:
            score: Single score submission to send

        Returns:
            API response as dictionary
        """
        return await self.submit_scores([score])

    async def submit_miner_scores(self, miner_scores: list[CandleTAOMinerScoreSubmission]) -> dict[str, Any]:
        """
        Submit a batch of miner scores to the CandleTAO API

        Args:
            miner_scores: List of miner score submissions to send

        Returns:
            API response as dictionary

        Raises:
            aiohttp.ClientError: On HTTP client errors
            ValueError: On invalid response data
        """
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

        # Convert miner scores to dict format for JSON serialization
        # The API expects the data structure to match MinerScoreUpdateSchema
        payload = {
            "data": [miner_score.model_dump(mode='json') for miner_score in miner_scores]
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.miner_scores_endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API request failed: {error_text}"
                    )

                try:
                    return await response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response from API: {e}")

    async def submit_single_prediction(self, prediction: CandleTAOPredictionSubmission) -> dict[str, Any]:
        """
        Submit a single prediction

        Args:
            prediction: Single prediction submission to send

        Returns:
            API response as dictionary
        """
        return await self.submit_predictions([prediction])


# Convenience functions for quick submission
async def submit_predictions(predictions: list[CandleTAOPredictionSubmission]) -> dict[str, Any]:
    """
    Convenience function to submit predictions without managing client instance

    Args:
        predictions: List of predictions to submit

    Returns:
        API response
    """
    client = CandleTAOClient()
    return await client.submit_predictions(predictions)


async def submit_prediction_scores(scores: list[CandleTAOScoreSubmission]) -> dict[str, Any]:
    """
    Convenience function to submit scores without managing client instance

    Args:
        scores: List of scores to submit

    Returns:
        API response
    """
    client = CandleTAOClient()
    return await client.submit_scores(scores)
