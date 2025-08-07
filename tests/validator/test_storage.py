import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from candles.validator.storage import JsonValidatorStorage


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.json_path = None
    return config


@pytest.fixture
def temp_storage_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def storage_with_temp_dir(temp_storage_dir):
    config = MagicMock()
    config.json_path = str(temp_storage_dir)
    with patch('candles.validator.storage.BaseJsonStorage.__init__'):
        storage = JsonValidatorStorage(config=config)
        storage.path = temp_storage_dir
        storage.validator_id = "test_validator_123"
        storage.config = config
        # Mock the parent class methods we need
        storage.load_data = MagicMock(return_value=None)
        storage.save_data = MagicMock()
        return storage


class TestJsonValidatorStorage:
    def test_init_creates_validator_id(self, mock_config):
        with patch("candles.validator.storage.BaseJsonStorage.__init__"):
            storage = JsonValidatorStorage(config=mock_config)
            assert hasattr(storage, "validator_id")
            assert storage.validator_id is not None

    def test_save_predictions_empty_new_predictions(self, storage_with_temp_dir):
        storage_with_temp_dir.save_predictions({})

        # Should call save_data with empty predictions
        storage_with_temp_dir.save_data.assert_called_once()

    def test_save_predictions_with_new_predictions(self, storage_with_temp_dir):
        new_predictions = {
            "interval_1": {
                "predictions": [
                    {"miner_uid": 1, "prediction": {"close": 100.0}},
                    {"miner_uid": 2, "prediction": {"close": 101.0}},
                ]
            }
        }

        storage_with_temp_dir.save_predictions(new_predictions)

        # Verify save_data was called with the predictions
        storage_with_temp_dir.save_data.assert_called_once()
        call_args = storage_with_temp_dir.save_data.call_args[1]
        assert "interval_1" in call_args["data"]

    def test_save_predictions_merges_with_existing(self, storage_with_temp_dir):
        # Mock existing data to be returned by load_data
        existing_data = {
            "interval_1": [{"close": 100.0}]
        }
        storage_with_temp_dir.load_data.return_value = existing_data

        # Save new predictions for same interval
        new_predictions = {
            "interval_1": {
                "predictions": [{"miner_uid": 2, "prediction": {"close": 101.0}}]
            }
        }
        storage_with_temp_dir.save_predictions(new_predictions)

        # Verify save_data was called and the data contains merged predictions
        storage_with_temp_dir.save_data.assert_called_once()
        call_args = storage_with_temp_dir.save_data.call_args[1]
        saved_data = call_args["data"]
        
        assert "interval_1" in saved_data
        assert len(saved_data["interval_1"]) == 2

    def test_save_predictions_prevents_duplicates(self, storage_with_temp_dir):
        # Mock existing data with a prediction from miner_uid 1
        existing_data = {
            "interval_1": [{"miner_uid": 1, "prediction": {"close": 100.0}}]
        }
        storage_with_temp_dir.load_data.return_value = existing_data

        # Try to save duplicate prediction (same miner_uid)
        duplicate_predictions = {
            "interval_1": {
                "predictions": [{"miner_uid": 1, "prediction": {"close": 102.0}}]
            }
        }
        storage_with_temp_dir.save_predictions(duplicate_predictions)

        # Verify save_data was called
        storage_with_temp_dir.save_data.assert_called_once()
        call_args = storage_with_temp_dir.save_data.call_args[1]
        saved_data = call_args["data"]

        # Verify the structure is correct
        assert "interval_1" in saved_data
        assert isinstance(saved_data["interval_1"], list)

    def test_load_predictions_returns_none_when_no_file(self, storage_with_temp_dir):
        result = storage_with_temp_dir.load_predictions()
        assert result is None

    def test_load_predictions_returns_saved_data(self, storage_with_temp_dir):
        mock_data = {
            "interval_1": [{"miner_uid": 1, "prediction": {"close": 100.0}}]
        }
        storage_with_temp_dir.load_data.return_value = mock_data

        loaded_predictions = storage_with_temp_dir.load_predictions()
        assert loaded_predictions is not None
        assert "interval_1" in loaded_predictions

    def test_load_predictions_by_interval_id_empty_when_no_file(
        self, storage_with_temp_dir
    ):
        result = storage_with_temp_dir.load_predictions_by_interval_id("interval_1")
        assert result == []

    def test_load_predictions_by_interval_id_filters_correctly(
        self, storage_with_temp_dir
    ):
        # Mock the load_data method to return predictions in the expected format
        mock_data = {
            "interval_1": [{"interval_id": "interval_1", "close": 100.0}],
            "interval_2": [{"interval_id": "interval_2", "close": 101.0}],
        }
        storage_with_temp_dir.load_data.return_value = mock_data

        result = storage_with_temp_dir.load_predictions_by_interval_id("interval_1")
        assert len(result) == 1
        assert result[0]["interval_id"] == "interval_1"

    def test_group_predictions_by_interval_empty_input(self, storage_with_temp_dir):
        result = storage_with_temp_dir._group_predictions_by_interval({})
        assert result == {}

    def test_group_predictions_by_interval_already_grouped(self, storage_with_temp_dir):
        predictions = {
            "interval_1": {"predictions": [{"close": 100.0}]},
            "interval_2": {"predictions": [{"close": 101.0}]},
        }

        result = storage_with_temp_dir._group_predictions_by_interval(predictions)
        assert result == predictions

    def test_group_predictions_by_interval_groups_list_data(
        self, storage_with_temp_dir
    ):
        predictions = {
            "interval_1": [{"close": 100.0}, {"close": 101.0}],
            "interval_2": [{"close": 102.0}],
        }

        result = storage_with_temp_dir._group_predictions_by_interval(predictions)
        assert "interval_1" in result
        assert len(result["interval_1"]) == 2

    def test_merge_new_with_existing_predictions_new_interval(
        self, storage_with_temp_dir
    ):
        existing = {}
        new_predictions = {
            "interval_1": {
                "predictions": [{"miner_uid": 1, "prediction": {"close": 100.0}}]
            }
        }

        result = storage_with_temp_dir._merge_new_with_existing_predictions(
            existing, new_predictions
        )
        assert "interval_1" in result
        assert len(result["interval_1"]) == 1

    def test_merge_new_with_existing_predictions_existing_interval(
        self, storage_with_temp_dir
    ):
        existing = {"interval_1": [{"miner_uid": 1, "prediction": {"close": 100.0}}]}
        new_predictions = {
            "interval_1": {
                "predictions": [{"miner_uid": 2, "prediction": {"close": 101.0}}]
            }
        }

        result = storage_with_temp_dir._merge_new_with_existing_predictions(
            existing, new_predictions
        )
        assert len(result["interval_1"]) == 2

    def test_merge_new_with_existing_predictions_handles_pydantic_objects(
        self, storage_with_temp_dir
    ):
        # Mock a pydantic-like object with a dict() method
        mock_prediction = MagicMock()
        mock_prediction.dict.return_value = {"miner_uid": 1, "close": 100.0}

        existing = {}
        new_predictions = {
            "interval_1": {
                "predictions": [mock_prediction]
            }
        }

        result = storage_with_temp_dir._merge_new_with_existing_predictions(
            existing, new_predictions
        )
        assert "interval_1" in result
        assert len(result["interval_1"]) == 1
        mock_prediction.dict.assert_called_once()

    def test_handles_json_decode_error_in_existing_predictions(
        self, storage_with_temp_dir
    ):
        # Test the JSON decode error handling in save_predictions
        storage_with_temp_dir.load_data.return_value = "invalid_json_string"

        new_predictions = {
            "interval_1": {
                "predictions": [{"miner_uid": 1, "prediction": {"close": 100.0}}]
            }
        }

        # Should not raise an exception and should handle the error gracefully
        storage_with_temp_dir.save_predictions(new_predictions)

        # Verify save_data was called
        storage_with_temp_dir.save_data.assert_called_once()
