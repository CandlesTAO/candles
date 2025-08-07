import pytest
from unittest.mock import MagicMock, patch, call
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

from candles.miner.miner import Miner
from candles.core.synapse import GetCandlePrediction
from candles.core.data import CandlePrediction, TimeInterval, CandleColor


@pytest.fixture
def mock_config():
    """Mock configuration for the miner."""
    config = MagicMock()
    config.blacklist.allow_non_registered = False
    config.blacklist.force_validator_permit = True
    config.blacklist.validator_min_stake = 100
    config.neuron.timeout = 30
    config.netuid = 357
    config.subtensor.chain_endpoint = "test"
    config.neuron.epoch_length = 100
    return config


@pytest.fixture
def mock_metagraph():
    """Mock metagraph for testing."""
    metagraph = MagicMock()
    metagraph.hotkeys = ["test_hotkey_1", "test_hotkey_2"]
    metagraph.validator_permit = [True, False]
    metagraph.S = [Decimal("1000"), Decimal("50")]
    metagraph.last_update = [100, 100]
    return metagraph


@pytest.fixture
def mock_wallet():
    """Mock wallet for testing."""
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "test_miner_hotkey"
    return wallet


@pytest.fixture
def mock_dendrite():
    """Mock dendrite for testing."""
    dendrite = MagicMock()
    dendrite.hotkey = "test_hotkey_1"
    return dendrite


@pytest.fixture
def sample_candle_prediction():
    """Sample candle prediction for testing."""
    return CandlePrediction(
        prediction_id=1,
        interval="hourly",
        interval_id="test_interval_123"
    )


@pytest.fixture
def sample_synapse(sample_candle_prediction):
    """Sample synapse for testing."""
    synapse = GetCandlePrediction(candle_prediction=sample_candle_prediction)
    # Create a proper dendrite mock
    from bittensor import TerminalInfo
    dendrite = TerminalInfo(
        status_code=200,
        status_message="Success",
        process_time=0.1,
        ip="127.0.0.1",
        port=8080,
        version=1,
        nonce=123,
        uuid="test-uuid",
        hotkey="test_hotkey_1"
    )
    synapse.dendrite = dendrite
    return synapse


@pytest.fixture
def miner_instance(mock_config):
    """Create a miner instance with mocked dependencies."""
    with patch("candles.miner.miner.BaseMinerNeuron.__init__"):
        with patch("candles.miner.base.bittensor.axon"):
            miner = Miner(config=mock_config)
            miner.config = mock_config
            miner.uid = 0
            miner.wallet = MagicMock()
            miner.wallet.hotkey.ss58_address = "test_miner_hotkey"
            miner.metagraph = MagicMock()
            miner.axon = MagicMock()
            return miner


class TestMinerInit:
    """Test miner initialization."""

    def test_miner_init_with_config(self, mock_config):
        """Test miner initialization with config."""
        with patch("candles.miner.miner.BaseMinerNeuron.__init__") as mock_init:
            Miner(config=mock_config)
            mock_init.assert_called_once_with(config=mock_config)

    def test_miner_init_without_config(self):
        """Test miner initialization without config."""
        with patch("candles.miner.miner.BaseMinerNeuron.__init__") as mock_init:
            Miner()
            mock_init.assert_called_once_with(config=None)


class TestBlacklistFunction:
    """Test the blacklist function."""

    def test_blacklist_no_hotkey(self):
        """Test blacklist when no hotkey is provided."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = None

        mock_miner = MagicMock()
        result = Miner.blacklist(mock_miner, synapse)

        assert result == (True, "Hotkey not provided")

    def test_blacklist_unregistered_hotkey_not_allowed(self):
        """Test blacklist when unregistered hotkey is not allowed."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "unregistered_hotkey"

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["valid_hotkey", "validator_hotkey"]
        mock_miner.config.blacklist.allow_non_registered = False

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result[0] is True
        assert "Unrecognized hotkey" in result[1]

    def test_blacklist_unregistered_hotkey_allowed(self):
        """Test blacklist when unregistered hotkey is allowed."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "unregistered_hotkey"

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["valid_hotkey", "validator_hotkey"]
        mock_miner.config.blacklist.allow_non_registered = True

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result == (False, "Allowing un-registered hotkey")

    def test_blacklist_non_validator_hotkey(self):
        """Test blacklist when hotkey is not a validator."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "valid_hotkey"  # Index 0, not a validator

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["valid_hotkey", "validator_hotkey"]
        mock_miner.metagraph.validator_permit = [False, True]
        mock_miner.config.blacklist.allow_non_registered = False
        mock_miner.config.blacklist.force_validator_permit = True

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result == (True, "Non-validator hotkey")

    def test_blacklist_insufficient_stake(self):
        """Test blacklist when validator has insufficient stake."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "validator_hotkey"  # Index 1, is validator but low stake

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["valid_hotkey", "validator_hotkey"]
        mock_miner.metagraph.validator_permit = [False, True]
        # Create mock tensors that have .item() method
        mock_tensor_50 = MagicMock()
        mock_tensor_50.item.return_value = 50
        mock_tensor_50_2 = MagicMock()
        mock_tensor_50_2.item.return_value = 50
        mock_miner.metagraph.S = [mock_tensor_50, mock_tensor_50_2]  # Both below minimum
        mock_miner.config.blacklist.allow_non_registered = False
        mock_miner.config.blacklist.force_validator_permit = True
        mock_miner.config.blacklist.validator_min_stake = 100

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result == (True, "Stake below minimum")

    def test_blacklist_valid_validator(self):
        """Test blacklist allows valid validator with sufficient stake."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "validator_hotkey"  # Index 1, is validator

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["valid_hotkey", "validator_hotkey"]
        mock_miner.metagraph.validator_permit = [False, True]
        # Create mock tensors that have .item() method
        mock_tensor_50 = MagicMock()
        mock_tensor_50.item.return_value = 50
        mock_tensor_1000 = MagicMock()
        mock_tensor_1000.item.return_value = 1000
        mock_miner.metagraph.S = [mock_tensor_50, mock_tensor_1000]  # Second has sufficient stake
        mock_miner.config.blacklist.allow_non_registered = False
        mock_miner.config.blacklist.force_validator_permit = True
        mock_miner.config.blacklist.validator_min_stake = 100

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result == (False, "Hotkey recognized!")


class TestPriorityFunction:
    """Test the priority function."""

    @pytest.mark.asyncio
    async def test_priority_calculation(self, miner_instance):
        """Test priority calculation based on stake."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "test_hotkey"

        # Mock metagraph
        miner_instance.metagraph.hotkeys = ["test_hotkey", "other_hotkey"]
        miner_instance.metagraph.S = [Decimal("500"), Decimal("1000")]

        priority = await miner_instance.priority(synapse)

        assert priority == 500.0

    @pytest.mark.asyncio
    async def test_priority_with_different_stakes(self, miner_instance):
        """Test priority with different stake amounts."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "high_stake_hotkey"

        miner_instance.metagraph.hotkeys = ["low_stake_hotkey", "high_stake_hotkey"]
        miner_instance.metagraph.S = [Decimal("100"), Decimal("2000")]

        priority = await miner_instance.priority(synapse)

        assert priority == 2000.0


class TestMakeCandlePrediction:
    """Test the make_candle_prediction method."""

    @pytest.mark.asyncio
    async def test_make_candle_prediction_generates_values(self, miner_instance, sample_candle_prediction):
        """Test that make_candle_prediction generates all required values."""
        with patch.object(miner_instance, 'find_prediction_file', return_value=None):
            with patch("random.uniform") as mock_uniform:
                with patch("random.choice") as mock_choice:
                    with patch("candles.miner.miner.datetime") as mock_datetime:
                        # Set up mocks
                        mock_uniform.side_effect = [500.0, 0.75]  # price, confidence
                        mock_choice.return_value = CandleColor.GREEN
                        mock_datetime.now.return_value.timestamp.return_value = 1234567890

                        miner_instance.uid = 5

                        result = await miner_instance.make_candle_prediction(sample_candle_prediction)

                        # Verify all fields are set
                        assert result.color == CandleColor.GREEN
                        assert result.price == Decimal("500.0")
                        assert result.confidence == Decimal("0.75")
                        assert result.miner_uid == 5
                        assert result.hotkey == "test_miner_hotkey"
                        assert result.interval_id == "test_interval_123"
                        expected_datetime = datetime.fromtimestamp(1, tz=timezone.utc)
                        assert result.prediction_date == expected_datetime

    @pytest.mark.asyncio
    async def test_make_candle_prediction_random_values(self, miner_instance, sample_candle_prediction):
        """Test that make_candle_prediction uses random values in expected ranges."""
        with patch.object(miner_instance, 'find_prediction_file', return_value=None):
            with patch("random.uniform") as mock_uniform:
                with patch("random.choice") as mock_choice:
                    with patch("candles.miner.miner.datetime"):
                        mock_uniform.side_effect = [750.0, 0.9]
                        mock_choice.return_value = CandleColor.RED

                        await miner_instance.make_candle_prediction(sample_candle_prediction)

                        # Verify random.uniform was called with correct ranges
                        calls = mock_uniform.call_args_list
                        assert calls[0] == call(100, 1000)  # price range
                        assert calls[1] == call(0.5, 1.0)   # confidence range

                        # Verify random.choice was called with colors
                        mock_choice.assert_called_once_with([CandleColor.RED, CandleColor.GREEN])

    @pytest.mark.asyncio
    async def test_make_candle_prediction_uses_file_when_available(self, miner_instance, sample_candle_prediction):
        """Test that make_candle_prediction uses file predictions when available."""
        # Mock file prediction to be found
        mock_prediction = CandlePrediction(
            prediction_id=1,
            interval="hourly",
            interval_id="test_interval_123",
            color=CandleColor.GREEN,
            price=Decimal("400.0"),
            confidence=Decimal("0.9")
        )

        with patch.object(miner_instance, 'find_prediction_file', return_value='/test/file.csv'):
            with patch('candles.miner.miner.get_file_predictions', return_value=[mock_prediction]):
                result = await miner_instance.make_candle_prediction(sample_candle_prediction)

                # Should return the file prediction
                assert result == mock_prediction


class TestFindPredictionFile:
    """Test the find_prediction_file method."""

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_hourly_found(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test finding hourly prediction files."""
        # Create a mock Path object that has exists() method
        mock_candles_data_dir = MagicMock()
        mock_candles_data_dir.exists.return_value = True
        mock_candles_data_dir.__str__.return_value = '/home/user/.candles/data'
        mock_candles_data_dir.__truediv__.return_value = '/home/user/.candles/data/hourly_*.csv'

        # Mock Path.home() to return a path that when divided creates our mock
        mock_home_path = MagicMock()
        mock_home_path.__truediv__.return_value.__truediv__.return_value = mock_candles_data_dir

        with patch.object(Path, 'home', return_value=mock_home_path):
            mock_glob.return_value = ['/home/user/.candles/data/hourly_predictions.csv']

            result = miner_instance.find_prediction_file(TimeInterval.HOURLY)

            assert result == '/home/user/.candles/data/hourly_predictions.csv'

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_daily_found(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test finding daily prediction files."""
        # Create a mock Path object that has exists() method
        mock_candles_data_dir = MagicMock()
        mock_candles_data_dir.exists.return_value = True
        mock_candles_data_dir.__str__.return_value = '/home/user/.candles/data'
        mock_candles_data_dir.__truediv__.return_value = '/home/user/.candles/data/daily_*.csv'

        # Mock Path.home() to return a path that when divided creates our mock
        mock_home_path = MagicMock()
        mock_home_path.__truediv__.return_value.__truediv__.return_value = mock_candles_data_dir

        with patch.object(Path, 'home', return_value=mock_home_path):
            mock_glob.return_value = ['/home/user/.candles/data/daily_data.csv']

            result = miner_instance.find_prediction_file(TimeInterval.DAILY)

            assert result == '/home/user/.candles/data/daily_data.csv'

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_weekly_found(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test finding weekly prediction files."""
        # Create a mock Path object that has exists() method
        mock_candles_data_dir = MagicMock()
        mock_candles_data_dir.exists.return_value = True
        mock_candles_data_dir.__str__.return_value = '/home/user/.candles/data'
        mock_candles_data_dir.__truediv__.return_value = '/home/user/.candles/data/weekly_*.csv'

        # Mock Path.home() to return a path that when divided creates our mock
        mock_home_path = MagicMock()
        mock_home_path.__truediv__.return_value.__truediv__.return_value = mock_candles_data_dir

        with patch.object(Path, 'home', return_value=mock_home_path):
            mock_glob.return_value = ['/home/user/.candles/data/weekly_forecast.csv']

            result = miner_instance.find_prediction_file(TimeInterval.WEEKLY)

            assert result == '/home/user/.candles/data/weekly_forecast.csv'

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_multiple_files_returns_first(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test that when multiple files match, the first (sorted) is returned."""
        # Create a mock Path object that has exists() method
        mock_candles_data_dir = MagicMock()
        mock_candles_data_dir.exists.return_value = True
        mock_candles_data_dir.__str__.return_value = '/home/user/.candles/data'
        mock_candles_data_dir.__truediv__.return_value = '/home/user/.candles/data/hourly_*.csv'

        # Mock Path.home() to return a path that when divided creates our mock
        mock_home_path = MagicMock()
        mock_home_path.__truediv__.return_value.__truediv__.return_value = mock_candles_data_dir

        with patch.object(Path, 'home', return_value=mock_home_path):
            mock_glob.return_value = [
                '/home/user/.candles/data/hourly_z.csv',
                '/home/user/.candles/data/hourly_a.csv'
            ]

            result = miner_instance.find_prediction_file(TimeInterval.HOURLY)

            assert result == '/home/user/.candles/data/hourly_a.csv'

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_fallback_to_env_var(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test fallback to PREDICTIONS_FILE_PATH environment variable."""
        with patch.object(Path, 'home', return_value=Path('/home/user')):
            # Mock .candles/data doesn't exist, but env var file does
            mock_exists.side_effect = lambda path: path == '/custom/path/predictions.csv'
            mock_glob.return_value = []  # No files in .candles/data
            mock_getenv.return_value = '/custom/path/predictions.csv'

            result = miner_instance.find_prediction_file(TimeInterval.HOURLY)

            assert result == '/custom/path/predictions.csv'
            mock_getenv.assert_called_once_with('PREDICTIONS_FILE_PATH')

    @patch('candles.miner.miner.glob.glob')
    @patch('candles.miner.miner.os.path.exists')
    @patch('candles.miner.miner.os.getenv')
    def test_find_prediction_file_env_var_file_not_exists(self, mock_getenv, mock_exists, mock_glob, miner_instance):
        """Test behavior when env var is set but file doesn't exist."""
        with patch.object(Path, 'home', return_value=Path('/home/user')):
            mock_exists.return_value = False  # Nothing exists
            mock_glob.return_value = []  # No files in .candles/data
            mock_getenv.return_value = '/nonexistent/predictions.csv'

            result = miner_instance.find_prediction_file(TimeInterval.HOURLY)

            assert result is None

    def test_find_prediction_file_unsupported_interval(self, miner_instance):
        """Test behavior with unsupported interval type."""
        result = miner_instance.find_prediction_file(TimeInterval.MONTHLY)

        assert result is None


class TestGetCandlePrediction:
    """Test the get_candle_prediction method."""

    @pytest.mark.asyncio
    async def test_get_candle_prediction_success(self, miner_instance, sample_synapse):
        """Test successful candle prediction retrieval."""
        # Mock the make_candle_prediction method
        expected_prediction = CandlePrediction(
            prediction_id=1,
            interval="hourly",
            interval_id="test_interval_123",
            color=CandleColor.GREEN,
            price=Decimal("500.0"),
            confidence=Decimal("0.75")
        )

        with patch.object(miner_instance, 'make_candle_prediction', return_value=expected_prediction) as mock_make:
            result = await miner_instance.get_candle_prediction(sample_synapse)

            # Verify make_candle_prediction was called with the original prediction
            mock_make.assert_called_once()
            call_args = mock_make.call_args[0][0]
            assert call_args.prediction_id == 1
            assert call_args.interval_id == "test_interval_123"
            assert call_args.interval == "hourly"

            # Verify result
            assert result.candle_prediction == expected_prediction
            assert result.version == 1

    @pytest.mark.asyncio
    async def test_get_candle_prediction_preserves_synapse(self, miner_instance, sample_synapse):
        """Test that get_candle_prediction preserves original synapse properties."""
        original_dendrite = sample_synapse.dendrite

        with patch.object(miner_instance, 'make_candle_prediction', return_value=sample_synapse.candle_prediction):
            result = await miner_instance.get_candle_prediction(sample_synapse)

            # Verify original synapse properties are preserved
            assert result.dendrite == original_dendrite
            assert result.version == 1


class TestSaveState:
    """Test the save_state method."""

    def test_save_state_no_exception(self, miner_instance):
        """Test that save_state runs without exception."""
        # Should not raise any exception
        miner_instance.save_state()


class TestMinerIntegration:
    """Integration tests for the miner."""

    @pytest.mark.asyncio
    async def test_full_prediction_flow(self, miner_instance):
        """Test the full prediction flow from synapse to response."""
        # Create a complete synapse
        candle_prediction = CandlePrediction(
            prediction_id=1,
            interval=TimeInterval.HOURLY,
            interval_id="integration_test_123"
        )
        synapse = GetCandlePrediction(candle_prediction=candle_prediction)
        # Create a proper dendrite mock
        from bittensor import TerminalInfo
        dendrite = TerminalInfo(
            status_code=200,
            status_message="Success",
            process_time=0.1,
            ip="127.0.0.1",
            port=8080,
            version=1,
            nonce=123,
            uuid="test-uuid",
            hotkey="test_validator"
        )
        synapse.dendrite = dendrite

        with patch.object(miner_instance, 'find_prediction_file', return_value=None):
            with patch("random.uniform") as mock_uniform:
                with patch("random.choice") as mock_choice:
                    with patch("candles.miner.miner.datetime") as mock_datetime:
                        # Set up deterministic values for testing
                        mock_uniform.side_effect = [850.0, 0.85]
                        mock_choice.return_value = CandleColor.RED
                        mock_datetime.now.return_value.timestamp.return_value = 1234567890

                        miner_instance.uid = 10

                        result = await miner_instance.get_candle_prediction(synapse)

                    # Verify complete prediction
                    prediction = result.candle_prediction
                    assert prediction.prediction_id == 1
                    assert prediction.interval == TimeInterval.HOURLY
                    assert prediction.interval_id == "integration_test_123"
                    assert prediction.color == CandleColor.RED
                    assert prediction.price == Decimal("850.0")
                    assert prediction.confidence == Decimal("0.85")
                    assert prediction.miner_uid == 10
                    assert prediction.hotkey == "test_miner_hotkey"
                    expected_datetime = datetime.fromtimestamp(1, tz=timezone.utc)
                    assert prediction.prediction_date == expected_datetime
                    assert result.version == 1


class TestMinerParametrizedTests:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("interval", [
        TimeInterval.HOURLY,
        TimeInterval.DAILY,
        TimeInterval.WEEKLY,
        TimeInterval.MONTHLY
    ])
    @pytest.mark.asyncio
    async def test_prediction_with_different_intervals(self, miner_instance, interval):
        """Test prediction generation with different time intervals."""
        candle_prediction = CandlePrediction(
            prediction_id=1,
            interval=interval,
            interval_id=f"test_{interval}_123"
        )

        with patch.object(miner_instance, 'find_prediction_file', return_value=None):
            with patch("random.uniform") as mock_uniform:
                with patch("random.choice") as mock_choice:
                    with patch("candles.miner.miner.datetime"):
                        mock_uniform.side_effect = [600.0, 0.8]
                        mock_choice.return_value = CandleColor.GREEN

                        result = await miner_instance.make_candle_prediction(candle_prediction)

                        assert result.interval == interval
                        assert result.interval_id == f"test_{interval}_123"

    @pytest.mark.parametrize("color", [CandleColor.RED, CandleColor.GREEN])
    @pytest.mark.asyncio
    async def test_prediction_with_different_colors(self, miner_instance, sample_candle_prediction, color):
        """Test prediction generation with different colors."""
        with patch.object(miner_instance, 'find_prediction_file', return_value=None):
            with patch("random.uniform") as mock_uniform:
                with patch("random.choice") as mock_choice:
                    with patch("candles.miner.miner.datetime"):
                        mock_uniform.side_effect = [400.0, 0.6]
                        mock_choice.return_value = color

                        result = await miner_instance.make_candle_prediction(sample_candle_prediction)

                        assert result.color == color

    @pytest.mark.parametrize("stake,expected_blacklist", [
        (50, True),   # Below minimum stake
        (100, False), # At minimum stake
        (500, False), # Above minimum stake
    ])
    def test_blacklist_different_stakes(self, stake, expected_blacklist):
        """Test blacklist behavior with different stake amounts."""
        synapse = MagicMock()
        synapse.dendrite.hotkey = "validator_hotkey"

        mock_miner = MagicMock()
        mock_miner.metagraph.hotkeys = ["validator_hotkey"]
        mock_miner.metagraph.validator_permit = [True]
        # Create mock tensor that has .item() method
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = stake
        mock_miner.metagraph.S = [mock_tensor]
        mock_miner.config.blacklist.allow_non_registered = False
        mock_miner.config.blacklist.force_validator_permit = True
        mock_miner.config.blacklist.validator_min_stake = 100

        with patch("candles.miner.miner.miner", mock_miner, create=True):
            result = Miner.blacklist(mock_miner, synapse)

        assert result[0] == expected_blacklist
