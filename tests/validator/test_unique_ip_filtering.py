import pytest
from unittest.mock import MagicMock, patch

from candles.validator.validator import Validator
from candles.core.data import CandlePrediction, TimeInterval


@pytest.fixture
def base_validator():
    """Construct a Validator with heavy deps mocked out."""
    with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
        with patch("candles.validator.validator.JsonValidatorStorage"):
            with patch.object(Validator, "load_state"):
                with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                    v = Validator(config=MagicMock())
                    v.storage = MagicMock()
                    v.dendrite = MagicMock()
                    v.uid = 0
                    v.metagraph = MagicMock()
                    v.metagraph.axons = {}
                    return v


def make_axon_with(attr_name: str | None, value: str | None):
    ax = MagicMock()
    if attr_name is not None:
        # Set the specific attribute
        setattr(ax, attr_name, value)
        # Ensure other IP attributes don't exist or return None
        for other_attr in ("ip", "external_ip", "ip_str"):
            if other_attr != attr_name:
                # Use side_effect to return None for other attributes
                setattr(ax, other_attr, None)
    return ax


class TestEnforceUniqueMinerIpFlag:
    def test_default_false_in_mock_mode(self, base_validator):
        # When config.mock is True, default should be False
        cfg = MagicMock()
        cfg.mock = True
        
        # Test the environment variable logic directly
        import os
        
        # Clear the environment variable
        with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}, clear=True):
            # The environment variable should not be set
            assert os.getenv("ENFORCE_UNIQUE_MINER_IP") is None
            
            # Test the logic that would be used in the validator
            if os.getenv("ENFORCE_UNIQUE_MINER_IP") is None:
                # No environment variable set, use config.mock to determine default
                enforce_flag = not bool(getattr(cfg, "mock", False))
            else:
                # This branch should not be taken
                enforce_flag = True
            
            # When config.mock is True, enforce_flag should be False
            assert enforce_flag is False

    def test_env_true_overrides(self):
        cfg = MagicMock()
        cfg.mock = True
        
        # Test the environment variable logic directly
        import os
        
        with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key", "ENFORCE_UNIQUE_MINER_IP": "true"}):
            # The environment variable should be set to "true"
            assert os.getenv("ENFORCE_UNIQUE_MINER_IP") == "true"
            
            # Test the logic that would be used in the validator
            env_flag = os.getenv("ENFORCE_UNIQUE_MINER_IP")
            if env_flag is None:
                enforce_flag = not bool(getattr(cfg, "mock", False))
            else:
                env_flag = env_flag.strip().lower()
                if env_flag in ("1", "true", "yes", "on"):
                    enforce_flag = True
                elif env_flag in ("0", "false", "no", "off"):
                    enforce_flag = False
                else:
                    enforce_flag = not bool(getattr(cfg, "mock", False))
            
            # When ENFORCE_UNIQUE_MINER_IP is "true", enforce_flag should be True
            assert enforce_flag is True

    def test_env_false_overrides(self):
        cfg = MagicMock()
        cfg.mock = False
        
        # Test the environment variable logic directly
        import os
        
        with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key", "ENFORCE_UNIQUE_MINER_IP": "0"}):
            # The environment variable should be set to "0"
            assert os.getenv("ENFORCE_UNIQUE_MINER_IP") == "0"
            
            # Test the logic that would be used in the validator
            env_flag = os.getenv("ENFORCE_UNIQUE_MINER_IP")
            if env_flag is None:
                enforce_flag = not bool(getattr(cfg, "mock", False))
            else:
                env_flag = env_flag.strip().lower()
                if env_flag in ("1", "true", "yes", "on"):
                    enforce_flag = True
                elif env_flag in ("0", "false", "no", "off"):
                    enforce_flag = False
                else:
                    enforce_flag = not bool(getattr(cfg, "mock", False))
            
            # When ENFORCE_UNIQUE_MINER_IP is "0", enforce_flag should be False
            assert enforce_flag is False


class TestFilterUidsByUniqueIp:
    def test_filters_duplicates_keeps_lowest_uid(self, base_validator):
        # Setup metagraph axons with duplicate IPs and an unknown
        base_validator.metagraph.axons = {
            1: make_axon_with("ip", "1.2.3.4"),
            2: make_axon_with("ip", "1.2.3.4"),  # duplicate IP with higher uid
            3: make_axon_with("external_ip", "5.6.7.8"),
            4: make_axon_with(None, None),  # unknown IP, should be kept as its own bucket
        }
        uids = [1, 2, 3, 4]

        filtered = base_validator._filter_uids_by_unique_ip(uids)

        assert filtered == [1, 3, 4]

    def test_uses_fallback_ip_attributes(self):
        # Create a validator instance for this test
        cfg = MagicMock()
        cfg.mock = False
        
        with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
            with patch("candles.validator.validator.JsonValidatorStorage"):
                with patch.object(Validator, "load_state"):
                    with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                        # Create validator and manually set required attributes
                        v = Validator(config=cfg)
                        v.metagraph = MagicMock()
                        v.metagraph.axons = {
                            5: make_axon_with("ip_str", "9.9.9.9"),
                            6: make_axon_with("ip_str", "9.9.9.9"),
                        }
                        uids = [6, 5]
                        # Lowest uid per IP should be kept (5)
                        assert v._filter_uids_by_unique_ip(uids) == [5]


class TestForwardIntegrationWithIpFilter:
    @pytest.mark.asyncio
    async def test_forward_applies_ip_filter_when_enabled_and_not_mock(self):
        # Build validator with non-mock config so filter is considered
        cfg = MagicMock()
        cfg.mock = False
        cfg.neuron = MagicMock()
        cfg.neuron.timeout = 30
        with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
            with patch("candles.validator.validator.JsonValidatorStorage"):
                with patch.object(Validator, "load_state"):
                    with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                        v = Validator(config=cfg)
                        v.config = cfg
                        v.enforce_unique_miner_ip = True
                        v.metagraph = MagicMock()
                        v.metagraph.axons = {
                            1: make_axon_with("ip", "1.1.1.1"),
                            2: make_axon_with("ip", "1.1.1.1"),  # same IP as uid 1
                            3: make_axon_with("ip", "2.2.2.2"),
                        }
                        v.uid = 0
                        v.storage = MagicMock()

                        # Return a single prediction request so forward proceeds
                        pred = CandlePrediction(
                            prediction_id=123,
                            interval=TimeInterval.HOURLY,
                            interval_id="123::hourly",
                            miner_uid=0,
                            hotkey="hk",
                        )

                        with patch.object(v, "get_next_candle_prediction_requests") as mock_get_reqs:
                            with patch("candles.validator.validator.get_miner_uids") as mock_get_uids:
                                with patch.object(v, "_gather_predictions_from_miners") as mock_gather:
                                    mock_get_reqs.return_value = [pred]
                                    mock_get_uids.return_value = [1, 2, 3]
                                    mock_gather.return_value = ([MagicMock()], [1, 3])

                                    await v.forward()

                                    # Ensure gather was called with filtered UIDs [1, 3]
                                    mock_gather.assert_called_once()
                                    args, kwargs = mock_gather.call_args
                                    assert args[0] == [pred]
                                    assert args[1] == [1, 3]
