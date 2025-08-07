# Standard Lib
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Local
from candles.core.data import CandlePrediction, CandleColor, TimeInterval
from candles.core.scoring.prediction_scorer import PredictionScorer
from candles.core.scoring.base import ScoringResult
from candles.prices.schemas import CoinDeskResponseOHLC


@pytest.fixture
def mock_price_client():
    """Mock PriceClient for testing."""
    client = MagicMock()
    client.get_price_by_interval = AsyncMock()
    return client


@pytest.fixture
def prediction_scorer(mock_price_client):
    """Create PredictionScorer instance with mocked dependencies."""
    return PredictionScorer(mock_price_client, "TAO/USDT")


@pytest.fixture
def sample_prediction():
    """Sample prediction for testing."""
    return CandlePrediction(
        prediction_id=1,
        miner_uid=100,
        interval_id="test_interval_1",
        interval=TimeInterval.HOURLY,
        color=CandleColor.GREEN,
        price=Decimal("50.0"),
        confidence=Decimal("0.8"),
        prediction_date=datetime.now(),
    )


@pytest.fixture
def sample_ohlc_data():
    """Sample OHLC data for testing."""
    return CoinDeskResponseOHLC(
        open=48.0, close=52.0, timestamp="2024-01-01T12:00:00Z", color=CandleColor.GREEN
    )


class TestPredictionScorer:
    """Test cases for PredictionScorer class."""

    @pytest.mark.asyncio
    async def test_score_prediction_with_actual_data(
        self, prediction_scorer, sample_prediction, sample_ohlc_data, mock_price_client
    ):
        """Test scoring with provided actual data."""
        # Mock the price client to return the sample data
        mock_price_client.get_price_by_interval.return_value = sample_ohlc_data
        
        result = await prediction_scorer.score_prediction(
            sample_prediction, actual_data=sample_ohlc_data.model_dump()
        )

        assert isinstance(result, ScoringResult)
        assert result.prediction_id == 1
        assert result.miner_uid == 100
        assert result.interval_id == "test_interval_1"
        assert result.color_score == 1.0  # Correct color prediction
        assert result.actual_color == "green"
        assert result.actual_price == 52.0
        assert result.confidence_weight == 0.8
        assert 0.0 <= result.price_score <= 1.0
        assert 0.0 <= result.final_score <= 1.0

    @pytest.mark.asyncio
    async def test_score_prediction_without_actual_data(
        self, prediction_scorer, sample_prediction, sample_ohlc_data, mock_price_client
    ):
        """Test scoring by fetching actual data from price client."""
        mock_price_client.get_price_by_interval.return_value = sample_ohlc_data

        result = await prediction_scorer.score_prediction(sample_prediction)

        mock_price_client.get_price_by_interval.assert_called_once_with(
            "TAO/USDT", "test_interval_1"
        )
        assert isinstance(result, ScoringResult)
        assert result.color_score == 1.0

    @pytest.mark.parametrize(
        "predicted_color,actual_color,expected_score",
        [
            (CandleColor.GREEN, CandleColor.GREEN, 1.0),
            (CandleColor.RED, CandleColor.RED, 1.0),
            (CandleColor.GREEN, CandleColor.RED, 0.0),
            (CandleColor.RED, CandleColor.GREEN, 0.0),
        ],
    )
    def test_calculate_color_score(
        self, prediction_scorer, predicted_color, actual_color, expected_score
    ):
        """Test color score calculation with different color combinations."""
        score = prediction_scorer._calculate_color_score(predicted_color, actual_color)
        assert score == expected_score

    @pytest.mark.parametrize(
        "predicted_price,actual_price,expected_range",
        [
            (50.0, 50.0, (0.9, 1.0)),  # Perfect prediction
            (50.0, 51.0, (0.8, 1.0)),  # Close prediction
            (50.0, 55.0, (0.4, 0.7)),  # Moderate error
            (50.0, 75.0, (0.0, 0.3)),  # Large error
            (50.0, 0.0, (0.0, 0.0)),  # Zero actual price
        ],
    )
    def test_calculate_price_score(
        self, prediction_scorer, predicted_price, actual_price, expected_range
    ):
        """Test price score calculation with different price scenarios."""
        score = prediction_scorer._calculate_price_score(predicted_price, actual_price)
        assert expected_range[0] <= score <= expected_range[1]

    def test_calculate_price_score_zero_actual_price(self, prediction_scorer):
        """Test price score calculation when actual price is zero."""
        score = prediction_scorer._calculate_price_score(50.0, 0.0)
        assert score == 0.0

    @pytest.mark.parametrize(
        "color_score,price_score,confidence,expected_range",
        [
            (1.0, 1.0, 0.8, (0.9, 1.0)),  # Perfect prediction with high confidence
            (1.0, 1.0, 0.2, (0.8, 1.0)),  # Perfect prediction with low confidence
            (0.0, 0.0, 0.8, (0.0, 0.2)),  # Poor prediction with high confidence
            (0.0, 0.0, 0.2, (0.0, 0.1)),  # Poor prediction with low confidence
            (0.6, 0.4, 0.5, (0.4, 0.6)),  # Average prediction with medium confidence
        ],
    )
    def test_calculate_final_score(
        self, prediction_scorer, color_score, price_score, confidence, expected_range
    ):
        """Test final score calculation with different combinations."""
        score = prediction_scorer._calculate_final_score(
            color_score, price_score, confidence
        )
        assert expected_range[0] <= score <= expected_range[1]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_prediction_with_no_confidence(
        self, prediction_scorer, sample_ohlc_data, mock_price_client
    ):
        """Test scoring prediction with no confidence value."""
        prediction = CandlePrediction(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval_1",
            interval=TimeInterval.HOURLY,
            color=CandleColor.GREEN,
            price=Decimal("50.0"),
            confidence=None,
            prediction_date=datetime.now(),
        )

        # Mock the price client to return the sample data
        mock_price_client.get_price_by_interval.return_value = sample_ohlc_data

        result = await prediction_scorer.score_prediction(
            prediction, actual_data=sample_ohlc_data.model_dump()
        )

        assert result.confidence_weight == 0.5  # Default confidence

    @pytest.mark.asyncio
    async def test_score_prediction_wrong_color_prediction(
        self, prediction_scorer, sample_ohlc_data, mock_price_client
    ):
        """Test scoring with incorrect color prediction."""
        prediction = CandlePrediction(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval_1",
            interval=TimeInterval.HOURLY,
            color=CandleColor.RED,  # Wrong color
            price=Decimal("52.0"),  # Correct price
            confidence=Decimal("0.9"),
            prediction_date=datetime.now(),
        )

        # Mock the price client to return the sample data
        mock_price_client.get_price_by_interval.return_value = sample_ohlc_data

        result = await prediction_scorer.score_prediction(
            prediction, actual_data=sample_ohlc_data.model_dump()
        )

        assert result.color_score == 0.0
        assert result.price_score > 0.9  # Should be high for exact price match
        assert result.final_score < 0.5  # Should be low due to wrong color

    @pytest.mark.asyncio
    async def test_different_symbols(
        self, mock_price_client, sample_prediction, sample_ohlc_data
    ):
        """Test scorer with different trading symbols."""
        scorer = PredictionScorer(mock_price_client, "BTC/USDT")
        mock_price_client.get_price_by_interval.return_value = sample_ohlc_data

        await scorer.score_prediction(sample_prediction)

        mock_price_client.get_price_by_interval.assert_called_once_with(
            "BTC/USDT", "test_interval_1"
        )
