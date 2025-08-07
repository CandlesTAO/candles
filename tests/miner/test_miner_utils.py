import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from candles.miner.utils import get_file_predictions
from candles.core.data import TimeInterval


class TestGetFilePredictions:
    """Test the get_file_predictions function."""
    
    def test_get_file_predictions_filters_by_timestamp(self):
        """Test that get_file_predictions filters predictions by timestamp."""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Write CSV data with different timestamps using correct column names
            csv_content = """timestamp,color,confidence,price
1750701327,green,0.65,303.93
1750704927,red,0.09,301.76
1750708527,green,0.54,300.52"""
            temp_file.write(csv_content)
            temp_file.flush()
            
            try:
                # Mock get_next_timestamp_by_interval to return a specific timestamp
                next_timestamp = 1750704927  # This should match two rows (>= filter)
                
                with patch('candles.miner.utils.get_next_timestamp_by_interval', return_value=next_timestamp):
                    with patch('candles.miner.utils.build_prediction') as mock_build:
                        # Configure mock to return a simple object
                        mock_build.return_value = MagicMock()
                        
                        get_file_predictions(
                            filename=temp_file.name,
                            interval=TimeInterval.HOURLY,
                            miner_uid=1,
                            hotkey="test_hotkey"
                        )
                        
                        # Should call build_prediction twice (for two matching timestamps >= next_timestamp)
                        assert mock_build.call_count == 2
                        
                        # Verify the calls were made with the correct data
                        call_args_list = mock_build.call_args_list
                        # First call (timestamp 1750704927)
                        first_call_args = call_args_list[0][1]
                        assert first_call_args['price'] == 301.76
                        assert first_call_args['color'] == 'red'
                        assert first_call_args['confidence'] == 0.09
                        assert first_call_args['timestamp'] == 1750704927
                        assert first_call_args['interval'] == TimeInterval.HOURLY
                        assert first_call_args['miner_uid'] == 1
                        assert first_call_args['hotkey'] == "test_hotkey"
                        
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_predictions_no_matching_timestamp(self):
        """Test that get_file_predictions returns empty list when no timestamps match."""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            csv_content = """timestamp,color,confidence,price
1750701327,green,0.65,303.93
1750704927,red,0.09,301.76"""
            temp_file.write(csv_content)
            temp_file.flush()
            
            try:
                # Mock get_next_timestamp_by_interval to return a timestamp that doesn't exist
                next_timestamp = 9999999999  # This won't match any row
                
                with patch('candles.miner.utils.get_next_timestamp_by_interval', return_value=next_timestamp):
                    result = get_file_predictions(
                        filename=temp_file.name,
                        interval=TimeInterval.HOURLY
                    )
                    
                    # Should return empty list when no timestamps match
                    assert result == []
                    
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_predictions_rejects_earlier_timestamps(self):
        """Test that get_file_predictions rejects timestamps before next_timestamp."""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            csv_content = """timestamp,color,confidence,price
1750701327,green,0.65,303.93
1750704927,red,0.09,301.76
1750708527,green,0.54,300.52"""
            temp_file.write(csv_content)
            temp_file.flush()
            
            try:
                # Set next_timestamp to be after the first but equal to second timestamp
                next_timestamp = 1750704927
                
                with patch('candles.miner.utils.get_next_timestamp_by_interval', return_value=next_timestamp):
                    with patch('candles.miner.utils.build_prediction') as mock_build:
                        mock_build.return_value = MagicMock()
                        
                        get_file_predictions(
                            filename=temp_file.name,
                            interval=TimeInterval.HOURLY
                        )
                        
                        # Should call build_prediction twice (for timestamps >= next_timestamp)
                        assert mock_build.call_count == 2
                        
                        # Verify first call used the equal timestamp
                        call_args = mock_build.call_args_list[0][1]
                        assert call_args['timestamp'] == 1750704927  # Equal timestamp should be used
                        assert call_args['color'] == 'red'  # Data from the matching row
                        
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_predictions_multiple_equal_timestamps(self):
        """Test behavior when multiple rows have the same timestamp as next_timestamp."""
        # Create a temporary CSV file with duplicate timestamps
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            csv_content = """timestamp,color,confidence,price
1750704927,green,0.65,303.93
1750704927,red,0.09,301.76
1750708527,green,0.54,300.52"""
            temp_file.write(csv_content)
            temp_file.flush()
            
            try:
                next_timestamp = 1750704927
                
                with patch('candles.miner.utils.get_next_timestamp_by_interval', return_value=next_timestamp):
                    with patch('candles.miner.utils.build_prediction') as mock_build:
                        mock_build.return_value = MagicMock()
                        
                        get_file_predictions(
                            filename=temp_file.name,
                            interval=TimeInterval.HOURLY
                        )
                        
                        # Should call build_prediction three times (all three rows >= timestamp)
                        assert mock_build.call_count == 3
                        
            finally:
                os.unlink(temp_file.name)
    
    @pytest.mark.parametrize("interval", [
        TimeInterval.HOURLY,
        TimeInterval.DAILY,
        TimeInterval.WEEKLY,
        TimeInterval.MONTHLY
    ])
    def test_get_file_predictions_different_intervals(self, interval):
        """Test get_file_predictions with different time intervals."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            csv_content = """timestamp,color,confidence,price
1750704927,green,0.65,303.93"""
            temp_file.write(csv_content)
            temp_file.flush()
            
            try:
                next_timestamp = 1750704927
                
                with patch('candles.miner.utils.get_next_timestamp_by_interval', return_value=next_timestamp):
                    with patch('candles.miner.utils.build_prediction') as mock_build:
                        mock_build.return_value = MagicMock()
                        
                        get_file_predictions(
                            filename=temp_file.name,
                            interval=interval,
                            miner_uid=5,
                            hotkey="test_key"
                        )
                        
                        # Verify interval is passed correctly
                        call_args = mock_build.call_args[1]
                        assert call_args['interval'] == interval
                        assert call_args['miner_uid'] == 5
                        assert call_args['hotkey'] == "test_key"
                        
            finally:
                os.unlink(temp_file.name)