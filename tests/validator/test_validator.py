import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

from candles.validator.validator import Validator
from candles.core.data import CandlePrediction, TimeInterval
from candles.core.synapse import GetCandlePrediction


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.neuron.timeout = 30
    config.json_path = None
    return config


@pytest.fixture
def validator_instance(mock_config):
    with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
        with patch("candles.validator.validator.JsonValidatorStorage"):
            with patch.object(Validator, "load_state"):
                with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                    mock_config.mock = True  # Prevent background task creation
                    validator = Validator(config=mock_config)
                    validator.config = mock_config
                    validator.metagraph = MagicMock()
                    validator.dendrite = AsyncMock()
                    validator.uid = 1
                    validator.storage = MagicMock()
                    validator.blacklist = MagicMock()  # Add missing blacklist method
                    return validator


@pytest.fixture
def sample_candle_prediction():
    return CandlePrediction(
        prediction_id=1234567890,
        interval=TimeInterval.HOURLY,
        interval_id="1234567890::hourly",
        miner_uid=1,
        hotkey="test_hotkey",
    )


@pytest.fixture
def sample_get_candle_prediction(sample_candle_prediction):
    return GetCandlePrediction(candle_prediction=sample_candle_prediction)


class TestValidator:
    def test_init_creates_storage_instance(self, mock_config):
        with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
            with patch(
                "candles.validator.validator.JsonValidatorStorage"
            ) as mock_storage:
                with patch.object(Validator, "load_state"):
                    with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                        mock_config.mock = True  # Prevent background task creation
                        Validator(config=mock_config)
                        # The storage is created with the validator's config property
                        mock_storage.assert_called_once()

    def test_should_request_hourly_returns_true_when_more_than_10_minutes(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_hour = datetime(
            2023, 1, 1, 13, 15, 0, tzinfo=timezone.utc
        ).timestamp()  # 1 hour 15 minutes away

        result = Validator._should_request_hourly(now, next_hour)
        assert result is True

    def test_should_request_hourly_returns_false_when_less_than_10_minutes(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_hour = datetime(
            2023, 1, 1, 12, 5, 0, tzinfo=timezone.utc
        ).timestamp()  # 5 minutes away

        result = Validator._should_request_hourly(now, next_hour)
        assert result is False

    def test_should_request_daily_returns_true_when_more_than_1_hour(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_day = datetime(
            2023, 1, 2, 2, 0, 0, tzinfo=timezone.utc
        ).timestamp()  # 14 hours away

        result = Validator._should_request_daily(now, next_day)
        assert result is True

    def test_should_request_daily_returns_false_when_less_than_1_hour(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_day = datetime(
            2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc
        ).timestamp()  # 30 minutes away

        result = Validator._should_request_daily(now, next_day)
        assert result is False

    def test_should_request_weekly_returns_true_when_more_than_1_day(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_week = datetime(
            2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc
        ).timestamp()  # 2 days away

        result = Validator._should_request_weekly(now, next_week)
        assert result is True

    def test_should_request_weekly_returns_false_when_less_than_1_day(self):
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_week = datetime(
            2023, 1, 1, 18, 0, 0, tzinfo=timezone.utc
        ).timestamp()  # 6 hours away

        result = Validator._should_request_weekly(now, next_week)
        assert result is False

    @patch("candles.validator.validator.get_next_timestamp_by_interval")
    @patch("candles.validator.validator.datetime")
    def test_get_next_candle_prediction_requests_returns_all_intervals(
        self, mock_datetime, mock_get_next_timestamp
    ):
        # Mock current time
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        # Mock next timestamps (all far enough in the future to trigger requests)
        mock_get_next_timestamp.side_effect = [
            mock_now.timestamp() + 3600,  # 1 hour away (hourly)
            mock_now.timestamp() + 86400,  # 1 day away (daily)
            mock_now.timestamp() + 604800,  # 1 week away (weekly)
        ]

        result = Validator.get_next_candle_prediction_requests()

        assert len(result) == 3
        assert any(pred.interval == TimeInterval.HOURLY for pred in result)
        assert any(pred.interval == TimeInterval.DAILY for pred in result)
        assert any(pred.interval == TimeInterval.WEEKLY for pred in result)

    @patch("candles.validator.validator.get_next_timestamp_by_interval")
    @patch("candles.validator.validator.datetime")
    def test_get_next_candle_prediction_requests_returns_empty_when_too_close(
        self, mock_datetime, mock_get_next_timestamp
    ):
        # Mock current time
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        # Mock next timestamps (all too close to trigger requests)
        mock_get_next_timestamp.side_effect = [
            mock_now.timestamp() + 300,  # 5 minutes away (hourly)
            mock_now.timestamp() + 1800,  # 30 minutes away (daily)
            mock_now.timestamp() + 3600,  # 1 hour away (weekly)
        ]

        result = Validator.get_next_candle_prediction_requests()

        assert len(result) == 0

    def test_parse_responses_groups_by_interval_id(
        self, sample_get_candle_prediction, sample_candle_prediction
    ):
        prediction_requests = [sample_candle_prediction]
        miner_predictions = [sample_get_candle_prediction]

        result = Validator.parse_responses(miner_predictions, prediction_requests)

        assert "1234567890::hourly" in result
        assert isinstance(result["1234567890::hourly"], list)
        assert len(result["1234567890::hourly"]) == 1
        assert result["1234567890::hourly"][0].miner_uid == 1

    def test_parse_responses_handles_multiple_predictions_same_interval(
        self, sample_candle_prediction
    ):
        # Create multiple predictions for the same interval
        prediction1 = GetCandlePrediction(candle_prediction=sample_candle_prediction)
        prediction2 = GetCandlePrediction(
            candle_prediction=CandlePrediction(
                prediction_id=1234567890,
                interval=TimeInterval.HOURLY,
                interval_id="1234567890::hourly",
                miner_uid=2,
                hotkey="test_hotkey_2",
            )
        )

        prediction_requests = [sample_candle_prediction]
        miner_predictions = [prediction1, prediction2]

        result = Validator.parse_responses(miner_predictions, prediction_requests)

        assert len(result["1234567890::hourly"]) == 2
        miner_uids = [
            pred.miner_uid for pred in result["1234567890::hourly"]
        ]
        assert 1 in miner_uids
        assert 2 in miner_uids

    @pytest.mark.asyncio
    async def test_send_predictions_to_miners(
        self, validator_instance, sample_candle_prediction
    ):
        """Test sending predictions to miners via utils module."""
        from candles.validator.utils import send_predictions_to_miners
        
        # Setup mock axons
        mock_axon1 = MagicMock()
        mock_axon2 = MagicMock()
        validator_instance.metagraph.axons = [mock_axon1, mock_axon2]

        # Setup mock responses
        mock_responses = [MagicMock(), MagicMock()]
        validator_instance.dendrite.return_value = mock_responses

        miner_uids = [0, 1]
        input_synapse = GetCandlePrediction(candle_prediction=sample_candle_prediction)

        with patch('candles.validator.utils.process_miner_requests') as mock_process:
            mock_process.return_value = (mock_responses, miner_uids)
            
            responses, returned_uids = await send_predictions_to_miners(
                validator=validator_instance,
                input_synapse=input_synapse,
                batch_uids=miner_uids
            )

            # Verify the function was called
            mock_process.assert_called_once_with(validator_instance, miner_uids, input_synapse)
            assert responses == mock_responses
            assert returned_uids == miner_uids

    @pytest.mark.asyncio
    async def test_forward_no_prediction_requests(self, validator_instance):
        with patch.object(
            validator_instance, "get_next_candle_prediction_requests"
        ) as mock_get_requests:
            with patch("candles.validator.validator.get_miner_uids") as mock_get_miners:
                mock_get_requests.return_value = []
                mock_get_miners.return_value = [1, 2, 3]

                await validator_instance.forward()

                # Should return early without calling any other methods
                validator_instance.storage.save_predictions.assert_not_called()

    @pytest.mark.asyncio
    async def test_forward_no_miners(self, validator_instance):
        with patch.object(
            validator_instance, "get_next_candle_prediction_requests"
        ) as mock_get_requests:
            with patch("candles.validator.validator.get_miner_uids") as mock_get_miners:
                mock_get_requests.return_value = [sample_candle_prediction]
                mock_get_miners.return_value = []

                await validator_instance.forward()

                # Should return early without calling any other methods
                validator_instance.storage.save_predictions.assert_not_called()

    @pytest.mark.asyncio
    async def test_forward_full_flow(
        self, validator_instance, sample_candle_prediction
    ):
        with patch.object(
            validator_instance, "get_next_candle_prediction_requests"
        ) as mock_get_requests:
            with patch("candles.validator.validator.get_miner_uids") as mock_get_miners:
                with patch.object(
                    validator_instance, "_gather_predictions_from_miners"
                ) as mock_gather:
                    with patch.object(
                        validator_instance, "save"
                    ) as mock_save_blacklist:

                        mock_get_requests.return_value = [sample_candle_prediction]
                        mock_get_miners.return_value = [1, 2]
                        mock_gather.return_value = ([MagicMock()], [1, 2])

                        await validator_instance.forward()

                        mock_gather.assert_called_once_with(
                            [sample_candle_prediction], [1, 2]
                        )
                        mock_save_blacklist.assert_called_once()

    @pytest.mark.asyncio
    async def test_gather_predictions_from_miners(
        self, validator_instance, sample_candle_prediction
    ):
        """Test gathering predictions from miners via utils module."""
        
        with patch('candles.validator.validator.send_predictions_to_miners') as mock_send:
            mock_send.return_value = ([MagicMock(), MagicMock()], [1, 2])

            (
                finished_responses,
                working_miners,
            ) = await validator_instance._gather_predictions_from_miners(
                [sample_candle_prediction], [1, 2]
            )

            # Verify the utils function was called
            mock_send.assert_called_once()
            assert len(finished_responses) == 2
            assert working_miners == [1, 2]

    def test_save_and_blacklist(self, validator_instance, sample_candle_prediction):
        with patch.object(Validator, "parse_responses") as mock_parse:

            mock_parse.return_value = {"interval_1": {"predictions": []}}
            finished_responses = [MagicMock()]
            working_miner_uids = [1, 2]
            miner_uids = [1, 2, 3]  # 3 is not working

            validator_instance.save(
                finished_responses,
                working_miner_uids,
                miner_uids,
                [sample_candle_prediction],
            )

            # Verify predictions were saved
            validator_instance.storage.save_predictions.assert_called_once_with(
                {"interval_1": {"predictions": []}}
            )


    def test_save_and_blacklist_all_miners_working(
        self, validator_instance, sample_candle_prediction
    ):
        with patch.object(Validator, "parse_responses") as mock_parse:

            mock_parse.return_value = {"interval_1": {"predictions": []}}
            finished_responses = [MagicMock()]
            working_miner_uids = [1, 2, 3]
            miner_uids = [1, 2, 3]  # All miners working

            validator_instance.save(
                finished_responses,
                working_miner_uids,
                miner_uids,
                [sample_candle_prediction],
            )

            # Verify predictions were saved
            validator_instance.storage.save_predictions.assert_called_once()

            # Verify no miners were blacklisted
            validator_instance.blacklist.assert_not_called()

    @patch("candles.validator.validator.datetime")
    def test_intervals_to_score_returns_all_closed_intervals(self, mock_datetime):
        """Test that _intervals_to_score returns all closed intervals."""
        # Mock current time: Monday, Jan 2, 2023 at 10:30 AM UTC
        mock_now = datetime(2023, 1, 2, 10, 30, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        result = Validator._intervals_to_score()

        # Should return one interval of each type since we're past the start of hour/day/week
        # Filter out None values since methods return None when no intervals are closed
        non_none_results = [interval for interval in result if interval is not None]
        assert len(non_none_results) == 3
        
        # Check that we have one of each interval type
        interval_types = [interval.split("::")[-1] for interval in non_none_results]
        assert TimeInterval.HOURLY in interval_types
        assert TimeInterval.DAILY in interval_types
        assert TimeInterval.WEEKLY in interval_types

    @patch("candles.validator.validator.datetime")
    def test_intervals_to_score_empty_at_exact_boundaries(self, mock_datetime):
        """Test that _intervals_to_score returns empty list at exact time boundaries."""
        # Mock current time: exactly at start of hour/day/week
        mock_now = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)  # Monday midnight
        mock_datetime.now.return_value = mock_now

        result = Validator._intervals_to_score()

        # Should return empty (all None values) since we're exactly at the boundary
        # Filter out None values since methods return None when no intervals are closed
        non_none_results = [interval for interval in result if interval is not None]
        assert len(non_none_results) == 0

    def test_get_closed_hourly_intervals_returns_previous_hour(self):
        """Test that _get_closed_hourly_intervals returns the previous hour when past hour start."""
        now = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_hourly_intervals(now, current_timestamp)

        # Should return the 11:00-12:00 interval as a string
        expected_prev_hour = datetime(2023, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        expected_interval_id = f"{int(expected_prev_hour.timestamp())}::{TimeInterval.HOURLY}"
        assert result == expected_interval_id

    def test_get_closed_hourly_intervals_empty_at_hour_start(self):
        """Test that _get_closed_hourly_intervals returns empty at exact hour start."""
        now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_hourly_intervals(now, current_timestamp)

        assert result is None

    def test_get_closed_daily_intervals_returns_previous_day(self):
        """Test that _get_closed_daily_intervals returns the previous day when past day start."""
        now = datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_daily_intervals(now, current_timestamp)

        # Should return the previous day (Jan 1) as a string
        expected_prev_day = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        expected_interval_id = f"{int(expected_prev_day.timestamp())}::{TimeInterval.DAILY}"
        assert result == expected_interval_id

    def test_get_closed_daily_intervals_empty_at_day_start(self):
        """Test that _get_closed_daily_intervals returns empty at exact day start."""
        now = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_daily_intervals(now, current_timestamp)

        assert result is None

    def test_get_closed_weekly_intervals_returns_previous_week(self):
        """Test that _get_closed_weekly_intervals returns the previous week when past week start."""
        # Tuesday, Jan 3, 2023 (Monday was Jan 2)
        now = datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_weekly_intervals(now, current_timestamp)

        # Should return the previous week (starting Dec 26, 2022) as a string
        expected_prev_week = datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)
        expected_interval_id = f"{int(expected_prev_week.timestamp())}::{TimeInterval.WEEKLY}"
        assert result == expected_interval_id

    def test_get_closed_weekly_intervals_empty_at_week_start(self):
        """Test that _get_closed_weekly_intervals returns empty at exact week start."""
        # Monday, Jan 2, 2023 at midnight (start of week)
        now = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_weekly_intervals(now, current_timestamp)

        assert result is None

    def test_get_closed_weekly_intervals_handles_sunday_correctly(self):
        """Test that _get_closed_weekly_intervals handles Sunday correctly (weekday=6)."""
        # Sunday, Jan 8, 2023 at noon
        now = datetime(2023, 1, 8, 12, 0, 0, tzinfo=timezone.utc)
        current_timestamp = now.timestamp()

        result = Validator._get_closed_weekly_intervals(now, current_timestamp)

        # Should return the previous week (starting Dec 26, 2022) as a string
        expected_prev_week = datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)
        expected_interval_id = f"{int(expected_prev_week.timestamp())}::{TimeInterval.WEEKLY}"
        assert result == expected_interval_id

    @pytest.mark.parametrize("test_time,expected_intervals", [
        # Test various times and expected closed intervals
        (datetime(2023, 1, 2, 10, 30, 0, tzinfo=timezone.utc), 3),  # Monday 10:30 AM (all intervals closed)
        (datetime(2023, 1, 2, 0, 30, 0, tzinfo=timezone.utc), 3),   # Monday 12:30 AM (all intervals closed)
        (datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc), 3),  # Sunday 12:30 PM (all intervals closed)
        (datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc), 0),    # Monday midnight (none - exact boundary)
    ], ids=["monday_morning", "monday_past_midnight", "sunday_afternoon", "monday_midnight"])
    @patch("candles.validator.validator.datetime")
    def test_intervals_to_score_parametrized_scenarios(self, mock_datetime, test_time, expected_intervals):
        """Test _intervals_to_score with various time scenarios."""
        mock_datetime.now.return_value = test_time

        result = Validator._intervals_to_score()

        # Filter out None values since methods return None when no intervals are closed
        non_none_results = [interval for interval in result if interval is not None]
        assert len(non_none_results) == expected_intervals
