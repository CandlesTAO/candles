# Standard Lib
import pytest
from decimal import Decimal
from datetime import datetime

# Local
from candles.core.data import CandlePrediction, CandleColor, TimeInterval
from candles.core.scoring.base import BaseScorer, ScoringResult


class TestScoringResult:
    """Test cases for ScoringResult model."""

    def test_scoring_result_creation(self):
        """Test creating a ScoringResult instance."""
        result = ScoringResult(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval",
            color_score=1.0,
            price_score=0.9,
            confidence_weight=0.8,
            final_score=0.85,
            actual_color="green",
            actual_price=52.0,
        )

        assert result.prediction_id == 1
        assert result.miner_uid == 100
        assert result.interval_id == "test_interval"
        assert result.color_score == 1.0
        assert result.price_score == 0.9
        assert result.confidence_weight == 0.8
        assert result.final_score == 0.85
        assert result.actual_color == "green"
        assert result.actual_price == 52.0

    def test_scoring_result_validation(self):
        """Test ScoringResult model validation."""
        # Valid result
        result = ScoringResult(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval",
            color_score=0.5,
            price_score=0.5,
            confidence_weight=0.5,
            final_score=0.5,
            actual_color="red",
            actual_price=48.0,
        )
        assert result is not None

    def test_scoring_result_json_serialization(self):
        """Test JSON serialization of ScoringResult."""
        result = ScoringResult(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval",
            color_score=1.0,
            price_score=0.9,
            confidence_weight=0.8,
            final_score=0.85,
            actual_color="green",
            actual_price=52.0,
        )

        json_data = result.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["prediction_id"] == 1
        assert json_data["final_score"] == 0.85

    def test_scoring_result_from_dict(self):
        """Test creating ScoringResult from dictionary."""
        data = {
            "prediction_id": 1,
            "miner_uid": 100,
            "interval_id": "test_interval",
            "color_score": 1.0,
            "price_score": 0.9,
            "confidence_weight": 0.8,
            "final_score": 0.85,
            "actual_color": "green",
            "actual_price": 52.0,
        }

        result = ScoringResult(**data)
        assert result.final_score == 0.85


class TestBaseScorer:
    """Test cases for BaseScorer abstract class."""

    def test_base_scorer_is_abstract(self):
        """Test that BaseScorer is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseScorer()

    def test_base_scorer_inheritance(self):
        """Test that BaseScorer can be inherited and implemented."""

        class MockScorer(BaseScorer):
            async def score_prediction(self, prediction, actual_data=None):
                return ScoringResult(
                    prediction_id=prediction.prediction_id,
                    miner_uid=prediction.miner_uid,
                    interval_id=prediction.interval_id,
                    color_score=1.0,
                    price_score=1.0,
                    confidence_weight=1.0,
                    final_score=1.0,
                    actual_color="green",
                    actual_price=50.0,
                )

        scorer = MockScorer()
        assert isinstance(scorer, BaseScorer)

    @pytest.mark.asyncio
    async def test_base_scorer_implementation(self):
        """Test that implemented BaseScorer works correctly."""

        class MockScorer(BaseScorer):
            async def score_prediction(self, prediction, actual_data=None):
                return ScoringResult(
                    prediction_id=prediction.prediction_id,
                    miner_uid=prediction.miner_uid,
                    interval_id=prediction.interval_id,
                    color_score=0.8,
                    price_score=0.9,
                    confidence_weight=0.7,
                    final_score=0.85,
                    actual_color="green",
                    actual_price=52.0,
                )

        scorer = MockScorer()
        prediction = CandlePrediction(
            prediction_id=1,
            miner_uid=100,
            interval_id="test_interval",
            interval=TimeInterval.HOURLY,
            color=CandleColor.GREEN,
            price=Decimal("50.0"),
            confidence=Decimal("0.8"),
            prediction_date=datetime.now(),
        )

        result = await scorer.score_prediction(prediction)

        assert isinstance(result, ScoringResult)
        assert result.prediction_id == 1
        assert result.final_score == 0.85

    def test_base_scorer_abstract_methods(self):
        """Test that BaseScorer has the expected abstract methods."""
        assert hasattr(BaseScorer, "score_prediction")
        assert BaseScorer.__abstractmethods__ == {"score_prediction"}
