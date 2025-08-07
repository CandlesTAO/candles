from datetime import datetime, timedelta, timezone
from pytest import fixture
from candles.core.data import CandlePrediction, TimeInterval


@fixture
def mock_now():
    return datetime.now(timezone.utc)


@fixture
def mock_hour():
    return datetime.now(timezone.utc) + timedelta(hours=1)


@fixture
def mock_day():
    return datetime.now(timezone.utc) + timedelta(days=1)


@fixture
def mock_week():
    return datetime.now(timezone.utc) + timedelta(days=7)


@fixture
def mock_get_next_timestamp_by_interval():
    return datetime.now(timezone.utc) + timedelta(hours=1)


@fixture
def mock_get_next_candle_prediction_requests_hourly():
    return [
        CandlePrediction(
            interval=TimeInterval.HOURLY,
            interval_id=f"{mock_hour}::{TimeInterval.HOURLY}",
            prediction_id=mock_hour,
        ),
    ]


@fixture
def mock_get_next_candle_prediction_requests_daily():
    return [
        CandlePrediction(
            interval=TimeInterval.DAILY,
            interval_id=f"{mock_day}::{TimeInterval.DAILY}",
            prediction_id=mock_day,
        ),
    ]


@fixture
def mock_get_next_candle_prediction_requests_weekly():
    return [
        CandlePrediction(
            interval=TimeInterval.WEEKLY,
            interval_id=f"{mock_week}::{TimeInterval.WEEKLY}",
            prediction_id=mock_week,
        ),
    ]


@fixture
def mock_get_next_candle_prediction_requests_none():
    return []


@fixture
def mock_get_next_candle_prediction_requests_hourly_daily():
    return [
        CandlePrediction(
            interval=TimeInterval.HOURLY,
            interval_id=f"{mock_hour}::{TimeInterval.HOURLY}",
            prediction_id=mock_hour,
        ),
        CandlePrediction(
            interval=TimeInterval.DAILY,
            interval_id=f"{mock_day}::{TimeInterval.DAILY}",
            prediction_id=mock_day,
        ),
    ]


@fixture
def mock_get_next_candle_prediction_requests_hourly_daily_weekly():
    return [
        CandlePrediction(
            interval=TimeInterval.HOURLY,
            interval_id=f"{mock_hour}::{TimeInterval.HOURLY}",
            prediction_id=mock_hour,
        ),
        CandlePrediction(
            interval=TimeInterval.DAILY,
            interval_id=f"{mock_day}::{TimeInterval.DAILY}",
            prediction_id=mock_day,
        ),
        CandlePrediction(
            interval=TimeInterval.WEEKLY,
            interval_id=f"{mock_week}::{TimeInterval.WEEKLY}",
            prediction_id=mock_week,
        ),
    ]


@fixture
def mock_get_miner_uids():
    return [1, 2, 3]
