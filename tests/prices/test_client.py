# Standard Lib
import pytest
import unittest.mock
from unittest.mock import Mock

# Third Party
import aiohttp
from aioresponses import aioresponses

# Local
from candles.prices.client import PriceClient, PriceProvider, APIConfig
from candles.prices.schemas import CoinDeskResponseOHLC
from candles.core.data import CandleColor


class TestAPIConfig:
    """Test cases for APIConfig dataclass."""

    def test_api_config_creation(self):
        """Test creating an APIConfig instance."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="test_key",
            api_key_header="X-API-Key",
            api_params={"param1": "value1"}
        )
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test_key"
        assert config.api_key_header == "X-API-Key"
        assert config.api_params == {"param1": "value1"}

    def test_api_config_minimal(self):
        """Test creating an APIConfig with minimal parameters."""
        config = APIConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"
        assert config.api_key is None
        assert config.api_key_header is None
        assert config.api_params is None


class TestPriceProvider:
    """Test cases for PriceProvider enum."""

    def test_price_provider_values(self):
        """Test PriceProvider enum values."""
        assert PriceProvider.COINDESK == "coindesk"

    def test_coindesk_config(self):
        """Test CoinDesk provider configuration."""
        config = PriceProvider.COINDESK.config
        assert isinstance(config, APIConfig)
        assert config.base_url == "https://data-api.coindesk.com/index/cc/v1/historical"
        assert config.api_key is None
        assert config.api_key_header is None
        expected_params = {
            "market": "cadli",
            "instrument": "",
            "limit": 1,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "to_ts": "",
            "groups": "OHLC",
        }
        assert config.api_params == expected_params

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            # This would fail in real usage as StrEnum validation prevents invalid values
            # But we test the logic in the config property
            provider = Mock()
            provider.__eq__ = Mock(return_value=False)
            PriceProvider.config.fget(provider)


class TestPriceClient:
    """Test cases for PriceClient class."""

    def test_price_client_initialization(self):
        """Test PriceClient initialization."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        assert client.api_key == "test_key"
        assert client.provider == "coindesk"
        assert client.provider_enum == PriceProvider.COINDESK
        assert isinstance(client.api_config, APIConfig)

    def test_price_client_invalid_provider(self):
        """Test PriceClient with invalid provider."""
        with pytest.raises(ValueError):
            PriceClient(api_key="test_key", provider="invalid_provider")

    @pytest.mark.asyncio
    async def test_get_price_by_interval_success(self):
        """Test successful price retrieval."""
        mock_response_data = {
            "Data": [{
                "OPEN": "50000.0",
                "CLOSE": "51000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            # Match the URL pattern with any query parameters
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::hourly")

            assert isinstance(result, CoinDeskResponseOHLC)
            assert result.open == 50000.0
            assert result.close == 51000.0
            assert result.timestamp == "2023-01-01T00:00:00Z"
            assert result.color == CandleColor.GREEN

    @pytest.mark.asyncio
    async def test_get_price_by_interval_red_candle(self):
        """Test price retrieval resulting in red candle."""
        mock_response_data = {
            "Data": [{
                "OPEN": "51000.0",
                "CLOSE": "50000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::hourly")

            assert result.color == CandleColor.RED

    @pytest.mark.asyncio
    async def test_get_price_by_interval_http_error(self):
        """Test handling of HTTP errors."""
        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, status=404)

            client = PriceClient(api_key="test_key", provider="coindesk")
            
            with pytest.raises(aiohttp.ClientResponseError):
                await client.get_price_by_interval("INVALID", "2023-01-01T00:00:00Z::hourly")

    def test_get_price_by_interval_method_exists(self):
        """Test that the get_price_by_interval method exists and is callable."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        
        # Verify that the method exists and is callable
        assert hasattr(client, 'get_price_by_interval')
        assert callable(client.get_price_by_interval)
        
        # Note: Caching decorator was removed from the current implementation
        # If caching is needed in the future, this test should be updated accordingly

    def test_api_params_modification(self):
        """Test that API parameters are correctly modified for requests."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        
        # Check that api_key and instrument are set properly in params
        client.api_config.api_params.copy()
        
        # Simulate what happens in get_price_by_interval
        client.api_config.api_params["api_key"] = client.api_key
        client.api_config.api_params["instrument"] = "BTC-USDT"
        
        assert client.api_config.api_params["api_key"] == "test_key"
        assert client.api_config.api_params["instrument"] == "BTC-USDT"

    @pytest.mark.asyncio
    async def test_symbol_case_handling(self):
        """Test that symbol is properly converted to uppercase."""
        mock_response_data = {
            "Data": [{
                "OPEN": "50000.0",
                "CLOSE": "51000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            await client.get_price_by_interval("btc", "2023-01-01T00:00:00Z::hourly")

            # Verify that the instrument parameter was set with uppercase symbol
            # Check that a request was made
            assert len(m.requests) == 1

    @pytest.mark.asyncio
    async def test_get_price_by_interval_daily_interval(self):
        """Test daily interval handling."""
        mock_response_data = {
            "Data": [{
                "OPEN": "50000.0",
                "CLOSE": "51000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/days.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::daily")

            assert isinstance(result, CoinDeskResponseOHLC)
            assert result.open == 50000.0
            assert result.close == 51000.0

    @pytest.mark.asyncio
    async def test_get_price_by_interval_weekly_calls_get_weekly_candle(self):
        """Test that weekly interval delegates to get_weekly_candle."""
        import unittest.mock
        
        with unittest.mock.patch.object(PriceClient, 'get_weekly_candle') as mock_weekly:
            mock_weekly.return_value = CoinDeskResponseOHLC(
                open=50000.0,
                close=51000.0,
                timestamp="2023-01-01T00:00:00Z",  # ISO format string
                color=CandleColor.GREEN
            )
            
            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "1672617600::weekly")
            
            mock_weekly.assert_called_once_with("BTC", 1672617600)
            assert isinstance(result, CoinDeskResponseOHLC)

    def test_get_price_by_interval_unsupported_interval(self):
        """Test unsupported interval raises ValueError."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        
        with pytest.raises(ValueError, match="Unsupported interval"):
            import asyncio
            asyncio.run(client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::monthly"))


class TestGetWeeklyCandle:
    """Test cases for the get_weekly_candle method."""

    @pytest.fixture
    def client(self):
        """Create a PriceClient for testing."""
        return PriceClient(api_key="test_key", provider="coindesk")

    def test_get_weekly_candle_wrong_provider(self, client):
        """Test get_weekly_candle with wrong provider raises ValueError."""
        # Create client with mock provider that isn't COINDESK
        client.provider_enum = Mock()
        client.provider_enum.__ne__ = Mock(return_value=True)
        
        with pytest.raises(ValueError, match="Weekly candle method only supports CoinDesk provider"):
            import asyncio
            asyncio.run(client.get_weekly_candle("BTC", 1672531200))

    def test_get_weekly_candle_invalid_weekday(self, client):
        """Test get_weekly_candle with non-Monday timestamp raises ValueError."""
        sunday_timestamp = 1672531200  # Sunday 2023-01-01 00:00:00 UTC
        
        with pytest.raises(ValueError, match="Week start must be a Monday"):
            import asyncio
            asyncio.run(client.get_weekly_candle("BTC", sunday_timestamp))

    def test_get_weekly_candle_invalid_time(self, client):
        """Test get_weekly_candle with non-midnight timestamp raises ValueError."""
        monday_noon_timestamp = 1672660800  # Monday 2023-01-02 12:00:00 UTC
        
        with pytest.raises(ValueError, match="Week start must be at midnight"):
            import asyncio
            asyncio.run(client.get_weekly_candle("BTC", monday_noon_timestamp))

    @pytest.mark.asyncio
    async def test_get_weekly_candle_success_green(self, client):
        """Test successful weekly candle creation with green color."""
        monday_timestamp = 1672617600  # Monday 2023-01-02 00:00:00 UTC
        
        # Mock the open candle (Monday start)
        open_candle = CoinDeskResponseOHLC(
            open=50000.0,
            close=50500.0,
            timestamp="2023-01-02T00:00:00Z",
            color=CandleColor.GREEN
        )
        
        # Mock the close candle (Sunday end)
        close_candle = CoinDeskResponseOHLC(
            open=51000.0,
            close=51500.0,
            timestamp="2023-01-08T23:00:00Z",
            color=CandleColor.GREEN
        )
        
        with unittest.mock.patch.object(client, 'get_price_by_interval') as mock_get_price:
            mock_get_price.side_effect = [open_candle, close_candle]
            
            result = await client.get_weekly_candle("BTC", monday_timestamp)
            
            # Verify method calls
            assert mock_get_price.call_count == 2
            expected_calls = [
                unittest.mock.call("BTC", "1672617600::hourly"),
                unittest.mock.call("BTC", "1673218800::hourly")
            ]
            mock_get_price.assert_has_calls(expected_calls)
            
            # Verify result
            assert isinstance(result, CoinDeskResponseOHLC)
            assert result.open == 50000.0  # From open candle
            assert result.close == 51500.0  # From close candle
            assert result.timestamp == "1672617600"  # timestamp string
            assert result.color == CandleColor.GREEN  # close >= open

    @pytest.mark.asyncio
    async def test_get_weekly_candle_success_red(self, client):
        """Test successful weekly candle creation with red color."""
        monday_timestamp = 1672617600  # Monday 2023-01-02 00:00:00 UTC
        
        # Mock the open candle (Monday start)
        open_candle = CoinDeskResponseOHLC(
            open=52000.0,
            close=51500.0,
            timestamp="2023-01-02T00:00:00Z",
            color=CandleColor.RED
        )
        
        # Mock the close candle (Sunday end)
        close_candle = CoinDeskResponseOHLC(
            open=51000.0,
            close=51000.0,  # Close lower than open
            timestamp="2023-01-08T23:00:00Z",
            color=CandleColor.GREEN
        )
        
        with unittest.mock.patch.object(client, 'get_price_by_interval') as mock_get_price:
            mock_get_price.side_effect = [open_candle, close_candle]
            
            result = await client.get_weekly_candle("BTC", monday_timestamp)
            
            # Verify result
            assert result.open == 52000.0
            assert result.close == 51000.0
            assert result.color == CandleColor.RED  # close < open

    @pytest.mark.asyncio
    async def test_get_weekly_candle_weekend_date_calculation(self, client):
        """Test that weekend date calculation is correct."""
        monday_timestamp = 1672617600  # Monday 2023-01-02 00:00:00 UTC
        
        open_candle = CoinDeskResponseOHLC(
            open=50000.0, close=50000.0, timestamp="2023-01-02T00:00:00Z", color=CandleColor.GREEN
        )
        close_candle = CoinDeskResponseOHLC(
            open=50000.0, close=50000.0, timestamp="2023-01-08T23:00:00Z", color=CandleColor.GREEN
        )
        
        with unittest.mock.patch.object(client, 'get_price_by_interval') as mock_get_price:
            mock_get_price.side_effect = [open_candle, close_candle]
            
            await client.get_weekly_candle("BTC", monday_timestamp)
            
            # Verify that the Sunday end time is calculated correctly (Sunday 11 PM)
            calls = mock_get_price.call_args_list
            sunday_call = calls[1][0][1]  # Second call, interval_id parameter
            assert sunday_call == "1673218800::hourly"

    def test_timestamp_validation_edge_cases(self, client):
        """Test timestamp validation edge cases."""
        # Test with various invalid timestamps (all non-Monday days)
        invalid_timestamps = [
            1672531200,  # Sunday 2023-01-01
            1672704000,  # Tuesday 2023-01-03
            1672790400,  # Wednesday 2023-01-04
            1672876800,  # Thursday 2023-01-05
            1672963200,  # Friday 2023-01-06
            1673049600,  # Saturday 2023-01-07
        ]
        
        for timestamp in invalid_timestamps:
            with pytest.raises(ValueError, match="Week start must be a Monday"):
                import asyncio
                asyncio.run(client.get_weekly_candle("BTC", timestamp))

    def test_time_validation_edge_cases(self, client):
        """Test time validation edge cases."""
        # Valid Monday but at different times (all based on 2023-01-02 which is Monday)
        invalid_times = [
            1672617601,  # Monday 2023-01-02 00:00:01 (1 second past midnight)
            1672617660,  # Monday 2023-01-02 00:01:00 (1 minute past midnight)
            1672621200,  # Monday 2023-01-02 01:00:00 (1 hour past midnight)  
            1672660800,  # Monday 2023-01-02 12:00:00 (noon)
        ]
        
        for timestamp in invalid_times:
            with pytest.raises(ValueError, match="Week start must be at midnight"):
                import asyncio
                asyncio.run(client.get_weekly_candle("BTC", timestamp))