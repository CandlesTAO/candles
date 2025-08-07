# Standard Lib
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Local
from candles.core.data import CandleColor, TimeInterval
from candles.core.scoring.batch_scorer import PredictionBatchScorer
from candles.core.scoring.base import ScoringResult
from candles.prices.schemas import CoinDeskResponseOHLC


@pytest.fixture
def mock_price_client():
    """Mock PriceClient for testing."""
    client = MagicMock()
    client.get_price_by_interval = AsyncMock()
    return client


@pytest.fixture
def batch_scorer(mock_price_client):
    """Create PredictionBatchScorer instance with mocked dependencies."""
    return PredictionBatchScorer(mock_price_client, "TAO/USDT")


@pytest.fixture
def sample_predictions_data():
    """Sample predictions data grouped by interval."""
    return {
        "interval_1": [
            {
                "prediction_id": 1,
                "miner_uid": 100,
                "interval_id": "interval_1",
                "interval": TimeInterval.HOURLY,
                "color": CandleColor.GREEN,
                "price": "50.0",
                "confidence": "0.8",
                "prediction_date": datetime.now().isoformat(),
            },
            {
                "prediction_id": 2,
                "miner_uid": 101,
                "interval_id": "interval_1",
                "interval": TimeInterval.HOURLY,
                "color": CandleColor.RED,
                "price": "48.0",
                "confidence": "0.6",
                "prediction_date": datetime.now().isoformat(),
            },
        ],
        "interval_2": [
            {
                "prediction_id": 3,
                "miner_uid": 100,
                "interval_id": "interval_2",
                "interval": TimeInterval.DAILY,
                "color": CandleColor.GREEN,
                "price": "55.0",
                "confidence": "0.9",
                "prediction_date": datetime.now().isoformat(),
            }
        ],
    }


@pytest.fixture
def nested_predictions_data():
    """Sample predictions data with nested structure."""
    return {
        "interval_1": [
            {
                "prediction_id": 1,
                "miner_uid": 100,
                "interval_id": "interval_1",
                "interval": "hourly",
                "color": "green",
                "price": "50.0",
                "confidence": "0.8",
                "prediction_date": datetime.now().isoformat(),
            }
        ]
    }


@pytest.fixture
def sample_scoring_results():
    """Sample scoring results for aggregation tests."""
    return {
        "interval_1": [
            ScoringResult(
                prediction_id=1,
                miner_uid=100,
                interval_id="interval_1",
                color_score=1.0,
                price_score=0.9,
                confidence_weight=0.8,
                final_score=0.85,
                actual_color="green",
                actual_price=52.0,
            ),
            ScoringResult(
                prediction_id=2,
                miner_uid=101,
                interval_id="interval_1",
                color_score=0.0,
                price_score=0.7,
                confidence_weight=0.6,
                final_score=0.28,
                actual_color="green",
                actual_price=52.0,
            ),
        ],
        "interval_2": [
            ScoringResult(
                prediction_id=3,
                miner_uid=100,
                interval_id="interval_2",
                color_score=1.0,
                price_score=0.8,
                confidence_weight=0.9,
                final_score=0.88,
                actual_color="green",
                actual_price=54.0,
            )
        ],
    }


class TestPredictionBatchScorer:
    """Test cases for PredictionBatchScorer class."""

    @pytest.mark.asyncio
    async def test_score_predictions_by_interval(
        self, batch_scorer, sample_predictions_data, mock_price_client
    ):
        """Test scoring predictions grouped by interval."""
        # Mock the price client to return sample OHLC data
        mock_ohlc = CoinDeskResponseOHLC(
            open=48.0,
            close=52.0,
            timestamp="2024-01-01T12:00:00Z",
            color=CandleColor.GREEN,
        )
        mock_price_client.get_price_by_interval.return_value = mock_ohlc

        results = await batch_scorer.score_predictions_by_interval(
            sample_predictions_data
        )

        assert len(results) == 2
        assert "interval_1" in results
        assert "interval_2" in results
        assert len(results["interval_1"]) == 2
        assert len(results["interval_2"]) == 1

        # Verify first result
        first_result = results["interval_1"][0]
        assert isinstance(first_result, ScoringResult)
        assert first_result.prediction_id == 1
        assert first_result.miner_uid == 100

    @pytest.mark.asyncio
    async def test_score_predictions_nested_structure(
        self, batch_scorer, nested_predictions_data, mock_price_client
    ):
        """Test scoring predictions with nested data structure."""
        mock_ohlc = CoinDeskResponseOHLC(
            open=48.0,
            close=52.0,
            timestamp="2024-01-01T12:00:00Z",
            color=CandleColor.GREEN,
        )
        mock_price_client.get_price_by_interval.return_value = mock_ohlc

        results = await batch_scorer.score_predictions_by_interval(
            nested_predictions_data
        )

        assert len(results) == 1
        assert "interval_1" in results
        assert len(results["interval_1"]) == 1

        result = results["interval_1"][0]
        assert result.prediction_id == 1
        assert result.miner_uid == 100

    @pytest.mark.asyncio
    async def test_score_predictions_with_invalid_data(self, batch_scorer):
        """Test handling of invalid prediction data."""
        invalid_data = {
            "interval_1": [
                {
                    "prediction_id": "invalid",  # Should be int
                    "color": "invalid_color",  # Invalid color
                    "interval": "invalid_interval",  # Invalid interval
                }
            ]
        }

        results = await batch_scorer.score_predictions_by_interval(invalid_data)

        assert "interval_1" in results
        assert len(results["interval_1"]) == 0  # No successful results

    @pytest.mark.asyncio
    async def test_score_predictions_empty_intervals(self, batch_scorer):
        """Test handling of empty intervals."""
        empty_data = {"interval_1": [], "interval_2": []}

        results = await batch_scorer.score_predictions_by_interval(empty_data)

        assert len(results) == 2
        assert len(results["interval_1"]) == 0
        assert len(results["interval_2"]) == 0

    def test_get_miner_scores(self, batch_scorer, sample_scoring_results):
        """Test aggregating scores by miner."""
        miner_scores = batch_scorer.get_miner_scores(sample_scoring_results)

        assert len(miner_scores) == 2
        assert 100 in miner_scores
        assert 101 in miner_scores

        # Test miner 100 stats (has 2 predictions)
        miner_100_stats = miner_scores[100]
        assert miner_100_stats["prediction_count"] == 2
        assert miner_100_stats["total_score"] == 1.73  # 0.85 + 0.88
        assert miner_100_stats["average_score"] == 0.865  # 1.73 / 2
        assert miner_100_stats["color_accuracy"] == 1.0  # Both correct
        assert abs(miner_100_stats["price_accuracy"] - 0.85) < 0.001  # (0.9 + 0.8) / 2
        assert (
            abs(miner_100_stats["average_confidence"] - 0.85) < 0.001
        )  # (0.8 + 0.9) / 2

        # Test miner 101 stats (has 1 prediction)
        miner_101_stats = miner_scores[101]
        assert miner_101_stats["prediction_count"] == 1
        assert miner_101_stats["total_score"] == 0.28
        assert miner_101_stats["average_score"] == 0.28
        assert miner_101_stats["color_accuracy"] == 0.0
        assert miner_101_stats["price_accuracy"] == 0.7
        assert miner_101_stats["average_confidence"] == 0.6

    def test_get_miner_scores_empty_results(self, batch_scorer):
        """Test get_miner_scores with empty results."""
        empty_results = {}
        miner_scores = batch_scorer.get_miner_scores(empty_results)
        assert len(miner_scores) == 0

    def test_get_top_miners(self, batch_scorer, sample_scoring_results):
        """Test getting top performing miners."""
        miner_scores = batch_scorer.get_miner_scores(sample_scoring_results)
        top_miners = batch_scorer.get_top_miners(miner_scores, limit=5)

        assert len(top_miners) == 2

        # Should be sorted by average_score descending
        assert top_miners[0]["miner_uid"] == 100  # Higher average score
        assert top_miners[1]["miner_uid"] == 101  # Lower average score

        # Verify structure
        top_miner = top_miners[0]
        assert "miner_uid" in top_miner
        assert "average_score" in top_miner
        assert "prediction_count" in top_miner
        assert "color_accuracy" in top_miner
        assert "price_accuracy" in top_miner
        assert "average_confidence" in top_miner

    def test_get_top_miners_with_limit(self, batch_scorer, sample_scoring_results):
        """Test getting top miners with limit."""
        miner_scores = batch_scorer.get_miner_scores(sample_scoring_results)
        top_miners = batch_scorer.get_top_miners(miner_scores, limit=1)

        assert len(top_miners) == 1
        assert top_miners[0]["miner_uid"] == 100

    def test_get_top_miners_empty_scores(self, batch_scorer):
        """Test get_top_miners with empty miner scores."""
        empty_scores = {}
        top_miners = batch_scorer.get_top_miners(empty_scores)
        assert len(top_miners) == 0

    @pytest.mark.asyncio
    async def test_score_predictions_exception_handling(
        self, batch_scorer, mock_price_client
    ):
        """Test exception handling during scoring."""
        # Mock price client to raise exception
        mock_price_client.get_price_by_interval.side_effect = Exception(
            "Price fetch failed"
        )

        predictions_data = {
            "interval_1": [
                {
                    "prediction_id": 1,
                    "miner_uid": 100,
                    "interval_id": "interval_1",
                    "interval": "hourly",
                    "color": "green",
                    "price": "50.0",
                    "confidence": "0.8",
                    "prediction_date": datetime.now().isoformat(),
                }
            ]
        }

        results = await batch_scorer.score_predictions_by_interval(predictions_data)

        assert "interval_1" in results
        # Should handle exceptions gracefully and return empty results
        assert len(results["interval_1"]) == 0

    def test_different_symbols(self, mock_price_client):
        """Test batch scorer with different trading symbols."""
        scorer = PredictionBatchScorer(mock_price_client, "BTC/USDT")
        assert scorer.prediction_scorer.symbol == "BTC/USDT"
