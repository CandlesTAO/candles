# Standard Lib
import pytest

# Third Party
from pydantic import ValidationError

# Local
from candles.prices.schemas import CoinDeskResponseOHLC
from candles.core.data import CandleColor


class TestCoinDeskResponseOHLC:
    """Test cases for CoinDeskResponseOHLC schema."""

    def test_model_creation_with_valid_data(self):
        """Test creating a CoinDeskResponseOHLC instance with valid data."""
        response = CoinDeskResponseOHLC(
            open=50000.0,
            close=51000.0,
            timestamp="2023-01-01T00:00:00Z",
            color=CandleColor.GREEN
        )
        
        assert response.open == 50000.0
        assert response.close == 51000.0
        assert response.timestamp == "2023-01-01T00:00:00Z"
        assert response.color == CandleColor.GREEN

    def test_model_validation_invalid_open(self):
        """Test validation with invalid open price."""
        with pytest.raises(ValidationError):
            CoinDeskResponseOHLC(
                open="invalid",
                close=51000.0,
                timestamp="2023-01-01T00:00:00Z",
                color=CandleColor.GREEN
            )

    def test_model_validation_invalid_close(self):
        """Test validation with invalid close price."""
        with pytest.raises(ValidationError):
            CoinDeskResponseOHLC(
                open=50000.0,
                close="invalid",
                timestamp="2023-01-01T00:00:00Z",
                color=CandleColor.GREEN
            )

    def test_model_validation_invalid_color(self):
        """Test validation with invalid color."""
        with pytest.raises(ValidationError):
            CoinDeskResponseOHLC(
                open=50000.0,
                close=51000.0,
                timestamp="2023-01-01T00:00:00Z",
                color="invalid_color"
            )

    def test_model_validation_missing_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            CoinDeskResponseOHLC()

    @pytest.mark.parametrize("open_price,close_price,expected_color", [
        (50000.0, 51000.0, CandleColor.GREEN),  # Close > Open
        (51000.0, 50000.0, CandleColor.RED),    # Close < Open
        (50000.0, 50000.0, CandleColor.GREEN),  # Close == Open
    ])
    def test_parse_response_color_logic(self, open_price: float, close_price: float, expected_color: CandleColor):
        """Test parse_response method color determination logic."""
        data = {
            "OPEN": str(open_price),
            "CLOSE": str(close_price),
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        assert result.color == expected_color

    def test_parse_response_success(self):
        """Test successful parsing of response data."""
        data = {
            "OPEN": "50000.5",
            "CLOSE": "51000.75",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == 50000.5
        assert result.close == 51000.75
        assert result.timestamp == "2023-01-01T00:00:00Z"
        assert result.color == CandleColor.GREEN

    def test_parse_response_string_numbers(self):
        """Test parsing with string number values."""
        data = {
            "OPEN": "50000",
            "CLOSE": "49000",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == 50000.0
        assert result.close == 49000.0
        assert result.color == CandleColor.RED

    def test_parse_response_invalid_open(self):
        """Test parsing with invalid open value."""
        data = {
            "OPEN": "invalid",
            "CLOSE": "51000.0",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        with pytest.raises(ValueError):
            CoinDeskResponseOHLC.parse_response(data)

    def test_parse_response_invalid_close(self):
        """Test parsing with invalid close value."""
        data = {
            "OPEN": "50000.0",
            "CLOSE": "invalid",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        with pytest.raises(ValueError):
            CoinDeskResponseOHLC.parse_response(data)

    def test_parse_response_missing_keys(self):
        """Test parsing with missing required keys."""
        data = {
            "OPEN": "50000.0",
            "CLOSE": "51000.0"
            # Missing TIMESTAMP
        }
        
        with pytest.raises(KeyError):
            CoinDeskResponseOHLC.parse_response(data)

    def test_parse_response_extra_keys(self):
        """Test parsing with extra keys (should be ignored)."""
        data = {
            "OPEN": "50000.0",
            "CLOSE": "51000.0",
            "TIMESTAMP": "2023-01-01T00:00:00Z",
            "HIGH": "52000.0",  # Extra key
            "LOW": "49000.0",   # Extra key
            "VOLUME": "1000"    # Extra key
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == 50000.0
        assert result.close == 51000.0
        assert result.timestamp == "2023-01-01T00:00:00Z"
        assert result.color == CandleColor.GREEN

    def test_parse_response_negative_prices(self):
        """Test parsing with negative prices."""
        data = {
            "OPEN": "-50000.0",
            "CLOSE": "-51000.0",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == -50000.0
        assert result.close == -51000.0
        assert result.color == CandleColor.RED  # -51000 < -50000

    def test_parse_response_zero_prices(self):
        """Test parsing with zero prices."""
        data = {
            "OPEN": "0.0",
            "CLOSE": "0.0",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == 0.0
        assert result.close == 0.0
        assert result.color == CandleColor.GREEN  # Equal prices result in GREEN

    def test_parse_response_decimal_precision(self):
        """Test parsing with high decimal precision."""
        data = {
            "OPEN": "50000.123456789",
            "CLOSE": "51000.987654321",
            "TIMESTAMP": "2023-01-01T00:00:00Z"
        }
        
        result = CoinDeskResponseOHLC.parse_response(data)
        
        assert result.open == 50000.123456789
        assert result.close == 51000.987654321
        assert result.color == CandleColor.GREEN

    def test_model_serialization(self):
        """Test model serialization to dict."""
        response = CoinDeskResponseOHLC(
            open=50000.0,
            close=51000.0,
            timestamp="2023-01-01T00:00:00Z",
            color=CandleColor.GREEN
        )
        
        result_dict = response.model_dump()
        expected = {
            "open": 50000.0,
            "close": 51000.0,
            "timestamp": "2023-01-01T00:00:00Z",
            "color": "green"
        }
        
        assert result_dict == expected

    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        response = CoinDeskResponseOHLC(
            open=50000.0,
            close=51000.0,
            timestamp="2023-01-01T00:00:00Z",
            color=CandleColor.GREEN
        )
        
        json_str = response.model_dump_json()
        expected_json = '{"open":50000.0,"close":51000.0,"timestamp":"2023-01-01T00:00:00Z","color":"green"}'
        
        assert json_str == expected_json