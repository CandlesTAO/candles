import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch


from candles.validator import utils
from candles.core.data import CandlePrediction, TimeInterval, CandleColor
from candles.core.synapse import GetCandlePrediction


@pytest.fixture
def mock_metagraph():
    metagraph = MagicMock()
    # Create mock UIDs that have .item() method like PyTorch tensors
    mock_uids = []
    for uid in [0, 1, 2, 3, 4]:
        mock_uid = MagicMock()
        mock_uid.item.return_value = uid
        mock_uids.append(mock_uid)
    metagraph.uids = mock_uids
    # Mock Tv (validator stakes) - make uid 1 a miner (0 stake), others validators (>0 stake)
    metagraph.Tv = {0: 100, 1: 0, 2: 100, 3: 100, 4: 100}
    return metagraph


@pytest.fixture
def mock_validator():
    validator = MagicMock()
    validator.metagraph = MagicMock()
    validator.metagraph.axons = {
        1: MagicMock(),
        2: MagicMock(),
        3: MagicMock(),
    }
    validator.dendrite = AsyncMock()
    validator.config = MagicMock()
    validator.config.neuron.batch_size = 2
    return validator


@pytest.fixture
def valid_candle_prediction():
    return CandlePrediction(
        prediction_id=int(datetime.now(timezone.utc).timestamp()),
        interval=TimeInterval.HOURLY,
        interval_id="test_interval",
        color=CandleColor.GREEN,
        price=Decimal("100.50"),
        confidence=Decimal("0.8500"),
    )


@pytest.fixture
def valid_input_synapse(valid_candle_prediction):
    return GetCandlePrediction(candle_prediction=valid_candle_prediction)


@pytest.fixture
def mock_response():
    from unittest.mock import create_autospec
    response = create_autospec(GetCandlePrediction, instance=True)
    response.candle_prediction = CandlePrediction(
        prediction_id=int(datetime.now(timezone.utc).timestamp()),
        interval=TimeInterval.HOURLY,
        interval_id="test_interval",
        color=CandleColor.GREEN,
        price=Decimal("100.50"),
        confidence=Decimal("0.8567"),
        prediction_date=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    response.axon = MagicMock()
    response.axon.hotkey = "test_hotkey"
    return response


class TestGetMinerUids:
    def test_get_miner_uids_returns_miners_from_metagraph(self, mock_metagraph):
        """Test that get_miner_uids returns miners from the metagraph."""
        result = utils.get_miner_uids(mock_metagraph, my_uid=0)
        assert result == [1]  # UID 1 is the only miner (Tv=0) and not excluded by my_uid

    def test_get_miner_uids_filters_miners_correctly(self, mock_metagraph):
        """Test that the function correctly filters miners from the metagraph."""
        # Create mock UIDs that have .item() method like PyTorch tensors
        mock_uids = []
        for uid in [5, 6, 7, 8]:
            mock_uid = MagicMock()
            mock_uid.item.return_value = uid
            mock_uids.append(mock_uid)
        mock_metagraph.uids = mock_uids
        # Mock Tv - make uid 6 a miner (0 stake), others validators (>0 stake)
        mock_metagraph.Tv = {5: 100, 6: 0, 7: 100, 8: 100}
        result = utils.get_miner_uids(mock_metagraph, my_uid=5)
        assert result == [6]  # Should return only the miner UIDs excluding my_uid

    def test_get_miner_uids_excludes_my_uid(self, mock_metagraph):
        """Test that the function excludes the validator's own UID."""
        result = utils.get_miner_uids(mock_metagraph, my_uid=1)
        assert result == []  # Should exclude UID 1 (which is a miner) since it's my_uid


class TestIsPredictionValid:
    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    def test_is_prediction_valid_success(
        self, mock_get_next_timestamp, valid_candle_prediction
    ):
        """Test that is_prediction_valid returns True for a valid prediction."""
        # Mock next timestamp to be 15 minutes from now
        future_time = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(
            minutes=15
        )
        mock_get_next_timestamp.return_value = int(future_time.timestamp())

        # Set prediction_id to be exactly at the next timestamp (valid)
        valid_candle_prediction.prediction_id = int(future_time.timestamp())

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is True

    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    def test_is_prediction_valid_future_prediction(
        self, mock_get_next_timestamp, valid_candle_prediction
    ):
        """Test that is_prediction_valid returns False for future predictions."""
        # Mock next timestamp to be now
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # Set prediction_id to be in the future (5 minutes later)
        future_time = current_time + timedelta(minutes=5)
        valid_candle_prediction.prediction_id = int(future_time.timestamp())

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is False

    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    def test_is_prediction_valid_too_old_prediction(
        self, mock_get_next_timestamp, valid_candle_prediction
    ):
        """Test that is_prediction_valid returns False for predictions that are too old."""
        # Mock next timestamp to be now
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # Set prediction_id to be in the past (15 minutes earlier)
        past_time = current_time - timedelta(minutes=15)
        valid_candle_prediction.prediction_id = int(past_time.timestamp())

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is False

    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    def test_is_prediction_valid_edge_case_exactly_10_minutes_old(
        self, mock_get_next_timestamp, valid_candle_prediction
    ):
        """Test that is_prediction_valid returns True for predictions exactly at next timestamp."""
        # Mock next timestamp to be now
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # Set prediction_id to be exactly at the next timestamp
        valid_candle_prediction.prediction_id = int(current_time.timestamp())

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is True

    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    def test_is_prediction_valid_edge_case_exactly_at_next_timestamp(
        self, mock_get_next_timestamp, valid_candle_prediction
    ):
        """Test that is_prediction_valid returns True for predictions exactly at next timestamp."""
        # Mock next timestamp to be now
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # Set prediction_id to be exactly at the next timestamp
        valid_candle_prediction.prediction_id = int(current_time.timestamp())

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is True

    def test_is_prediction_valid_none_prediction_id(self, valid_candle_prediction):
        """Test that is_prediction_valid returns False when prediction_id is None."""
        valid_candle_prediction.prediction_id = None

        is_valid = utils.is_prediction_valid(valid_candle_prediction)
        assert is_valid is False


class TestProcessSingleResponse:
    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    @patch("candles.validator.utils.datetime")
    def test_process_single_response_success(
        self, mock_datetime, mock_get_next_timestamp, mock_response
    ):
        """Test successful processing of a single response."""
        # Set up the prediction to be within the valid window
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())
        
        # Set prediction_id to be exactly at the next timestamp (valid)
        mock_response.candle_prediction.prediction_id = int(current_time.timestamp())

        # Set up datetime mocks properly
        mock_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.fromtimestamp.return_value = current_time

        # Make sure the mock object behaves like the real datetime
        mock_datetime.timezone = timezone

        result = utils.process_single_response(
            mock_response, uid=123
        )

        assert result is not None
        assert result.response == mock_response
        assert result.uid == 123
        assert result.response.candle_prediction.miner_uid == 123
        assert result.response.candle_prediction.hotkey == "test_hotkey"
        assert result.response.candle_prediction.prediction_date == mock_now

    def test_process_single_response_none_response(self):
        """Test processing when response is None."""
        with patch("bittensor.logging.debug") as mock_log:
            result = utils.process_single_response(None, uid=123)

            assert result is None
            mock_log.assert_called_with(
                "UID 123: Miner failed to respond"
            )

    def test_process_single_response_none_candle_prediction(self):
        """Test processing when candle_prediction is None."""
        response = MagicMock()
        response.candle_prediction = None
        response.axon = MagicMock()
        response.axon.hotkey = "test_hotkey"

        with patch("bittensor.logging.debug") as mock_log:
            result = utils.process_single_response(
                response, uid=123
            )

            assert result is None
            
            mock_log.assert_called_with(
                "UID 123: Miner failed to respond"
            )

    def test_process_single_response_none_axon(self, valid_candle_prediction):
        """Test processing when axon is None."""
        response = MagicMock()
        response.candle_prediction = valid_candle_prediction
        response.axon = None

        with patch("bittensor.logging.debug") as mock_log:
            result = utils.process_single_response(
                response, uid=123
            )

            assert result is None
            
            mock_log.assert_called_with(
                "UID 123: Miner failed to respond"
            )

    def test_process_single_response_none_hotkey(self, valid_candle_prediction):
        """Test processing when hotkey is None."""
        response = MagicMock()
        response.candle_prediction = valid_candle_prediction
        response.axon = MagicMock()
        response.axon.hotkey = None

        with patch("bittensor.logging.debug") as mock_log:
            result = utils.process_single_response(
                response, uid=123
            )

            assert result is None
            
            mock_log.assert_called_with(
                "UID 123: Miner failed to respond"
            )

    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    @patch("candles.validator.utils.datetime")
    def test_process_single_response_confidence_rounding(
        self, mock_datetime, mock_get_next_timestamp
    ):
        """Test that confidence is rounded to 4 decimal places."""
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        from unittest.mock import create_autospec
        response = create_autospec(GetCandlePrediction, instance=True)
        response.candle_prediction = CandlePrediction(
            prediction_id=int(current_time.timestamp()),
            interval=TimeInterval.HOURLY,
            interval_id="test_interval",
            color=CandleColor.GREEN,
            price=Decimal("100.0"),
            confidence=Decimal("0.123456789"),
        )
        response.axon = MagicMock()
        response.axon.hotkey = "test_hotkey"

        # Set up datetime mocks
        mock_now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone
        mock_datetime.fromtimestamp.return_value = current_time

        result = utils.process_single_response(response, uid=123)

        assert result.response == response
        assert result.uid == 123
        assert result.response.candle_prediction.miner_uid == 123
        assert result.response.candle_prediction.hotkey == "test_hotkey"
        assert result.response.candle_prediction.confidence == Decimal("0.1235")


class TestProcessResponses:
    @pytest.mark.asyncio
    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    @patch("candles.validator.utils.datetime")
    async def test_process_responses_success(
        self, mock_datetime, mock_get_next_timestamp, mock_validator
    ):
        """Test that process_responses succeeds with valid predictions."""
        batch_uids = [1, 2]

        # Set up valid prediction times
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # Mock responses
        response1 = GetCandlePrediction()
        response1.candle_prediction = CandlePrediction(
            prediction_id=int(current_time.timestamp()),
            interval=TimeInterval.HOURLY,
            interval_id="test_interval",
            confidence=Decimal("0.8"),
            color=CandleColor.GREEN,
        )
        axon1 = MagicMock()
        axon1.hotkey = "hotkey1"
        response1.__dict__['axon'] = axon1

        response2 = GetCandlePrediction()
        response2.candle_prediction = CandlePrediction(
            prediction_id=int(current_time.timestamp()),
            interval=TimeInterval.HOURLY,
            interval_id="test_interval",
            confidence=Decimal("0.9"),
            color=CandleColor.RED,
        )
        axon2 = MagicMock()
        axon2.hotkey = "hotkey2"
        response2.__dict__['axon'] = axon2

        mock_validator.dendrite.return_value = [response1, response2]

        # Set up datetime mocks
        mock_now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone
        mock_datetime.fromtimestamp.return_value = current_time

        input_synapse = GetCandlePrediction()

        finished_responses, working_uids = await utils.process_responses(
            batch_uids, mock_validator, input_synapse
        )

        assert len(finished_responses) == 2
        assert working_uids == [1, 2]
        assert finished_responses[0].candle_prediction.miner_uid == 1
        assert finished_responses[1].candle_prediction.miner_uid == 2

    @pytest.mark.asyncio
    @patch("candles.validator.utils.get_next_timestamp_by_interval")
    @patch("candles.validator.utils.datetime")
    async def test_process_responses_with_failures(
        self, mock_datetime, mock_get_next_timestamp, mock_validator
    ):
        """Test processing responses with some failures."""
        batch_uids = [1, 2]

        # Set up valid prediction times
        current_time = datetime.now(timezone.utc).replace(microsecond=0)
        mock_get_next_timestamp.return_value = int(current_time.timestamp())

        # First response is valid, second is None
        response1 = GetCandlePrediction()
        response1.candle_prediction = CandlePrediction(
            prediction_id=int(current_time.timestamp()),
            interval=TimeInterval.HOURLY,
            interval_id="test_interval",
            confidence=Decimal("0.8"),
            color=CandleColor.GREEN,
        )
        axon1 = MagicMock()
        axon1.hotkey = "hotkey1"
        response1.__dict__['axon'] = axon1

        response2 = None

        mock_validator.dendrite.return_value = [response1, response2]

        # Set up datetime mocks
        mock_now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone
        mock_datetime.fromtimestamp.return_value = current_time

        input_synapse = GetCandlePrediction()

        finished_responses, working_uids = await utils.process_responses(
            batch_uids, mock_validator, input_synapse
        )

        # Only the first response should be processed since the second is None
        assert len(finished_responses) == 1
        assert working_uids == [1]
        assert finished_responses[0].candle_prediction.miner_uid == 1


class TestProcessMinerRequests:
    @pytest.mark.asyncio
    async def test_process_miner_requests_single_batch(self, mock_validator):
        """Test processing miners in a single batch."""
        miner_uids = [1, 2]

        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_responses"
        ) as mock_process_responses:
            mock_process_responses.return_value = (["response1", "response2"], [1, 2])

            with patch("asyncio.sleep") as mock_sleep:
                with patch("bittensor.logging.info"):
                    all_responses, all_uids = await utils.process_miner_requests(
                        mock_validator, miner_uids, input_synapse
                    )

                    assert all_responses == ["response1", "response2"]
                    assert all_uids == [1, 2]
                    mock_process_responses.assert_called_once_with(
                        [1, 2], mock_validator, input_synapse
                    )
                    mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_miner_requests_multiple_batches(self, mock_validator):
        """Test processing miners in multiple batches."""
        miner_uids = [1, 2, 3, 4, 5]  # 5 miners with batch size 2
        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_responses"
        ) as mock_process_responses:
            # Mock responses for each batch
            mock_process_responses.side_effect = [
                (["response1", "response2"], [1, 2]),  # First batch
                (["response3", "response4"], [3, 4]),  # Second batch
                (["response5"], [5]),  # Third batch
            ]

            with patch("asyncio.sleep") as mock_sleep:
                with patch("bittensor.logging.info"):
                    all_responses, all_uids = await utils.process_miner_requests(
                        mock_validator, miner_uids, input_synapse
                    )

                    assert len(all_responses) == 5
                    assert len(all_uids) == 5
                    assert all_uids == [1, 2, 3, 4, 5]
                    assert mock_process_responses.call_count == 3
                    assert mock_sleep.call_count == 3


class TestSendPredictionsToMiners:
    @pytest.mark.asyncio
    async def test_send_predictions_to_miners_success(self, mock_validator):
        """Test successful sending of predictions to miners."""
        miner_uids = [1, 2, 3]
        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_miner_requests"
        ) as mock_process_requests:
            mock_process_requests.return_value = (["response1", "response2"], [1, 2])

            with patch("random.shuffle") as mock_shuffle:
                with patch("bittensor.logging.info") as mock_log:
                    result = await utils.send_predictions_to_miners(
                        mock_validator, input_synapse, miner_uids
                    )

                    assert result == (["response1", "response2"], [1, 2])
                    mock_shuffle.assert_called_once_with(miner_uids)
                    mock_process_requests.assert_called_once_with(
                        mock_validator, miner_uids, input_synapse
                    )
                    mock_log.assert_called_with("Received responses from 2 miners")

    @pytest.mark.asyncio
    async def test_send_predictions_to_miners_no_responses(self, mock_validator):
        """Test when no miners respond."""
        miner_uids = [1, 2, 3]
        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_miner_requests"
        ) as mock_process_requests:
            mock_process_requests.return_value = ([], [])

            with patch("random.shuffle"):
                with patch("bittensor.logging.info") as mock_log:
                    result = await utils.send_predictions_to_miners(
                        mock_validator, input_synapse, miner_uids
                    )

                    assert result == ([], [])
                    mock_log.assert_called_with("No miner responses available.")

    @pytest.mark.asyncio
    async def test_send_predictions_to_miners_exception_handling(self, mock_validator):
        """Test exception handling in send_predictions_to_miners."""
        miner_uids = [1, 2, 3]
        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_miner_requests"
        ) as mock_process_requests:
            mock_process_requests.side_effect = Exception("Test error")

            with patch("random.shuffle"):
                with patch("bittensor.logging.error") as mock_log_error:
                    result = await utils.send_predictions_to_miners(
                        mock_validator, input_synapse, miner_uids
                    )

                    assert result is None
                    mock_log_error.assert_called_once()
                    args, kwargs = mock_log_error.call_args
                    assert "Failed to send predictions to miners" in args[0]
                    assert "Test error" in args[0]

    @pytest.mark.asyncio
    async def test_send_predictions_to_miners_shuffles_uids(self, mock_validator):
        """Test that miner UIDs are shuffled before processing."""
        miner_uids = [1, 2, 3, 4, 5]
        input_synapse = GetCandlePrediction()

        with patch(
            "candles.validator.utils.process_miner_requests"
        ) as mock_process_requests:
            mock_process_requests.return_value = ([], [])

            with patch("random.shuffle") as mock_shuffle:
                with patch("bittensor.logging.info"):
                    await utils.send_predictions_to_miners(
                        mock_validator, input_synapse, miner_uids
                    )

                    mock_shuffle.assert_called_once_with(miner_uids)
                    # Verify the same list object was passed to both shuffle and process_miner_requests
                    shuffled_uids = mock_shuffle.call_args[0][0]
                    processed_uids = mock_process_requests.call_args[0][1]
                    assert shuffled_uids is processed_uids
