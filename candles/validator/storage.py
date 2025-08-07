from typing import Optional
import json
import bittensor

from ..core.storage.json_storage import BaseJsonStorage




class JsonValidatorStorage(BaseJsonStorage):
    """Handles storage and retrieval of validator predictions in JSON format.

    This class provides methods to save and load validator predictions using a unique validator ID.
    """

    def __init__(self, config: Optional["bittensor.config"] = None):
        super().__init__(config)
        self.validator_id = self.generate_user_id(config)
        bittensor.logging.info(f"Initialized validator storage with ID: {self.validator_id}")

    def save_predictions(self, new_predictions: dict) -> None:
        """Save the validator predictions to a single JSON file.

        Args:
            new_predictions: A dictionary of predictions keyed by interval_id.
        """
        prefix = f"{self.validator_id}_predictions"

        existing_predictions = self.load_data(prefix=prefix) or {}

        # Convert existing predictions to the expected format if needed
        if isinstance(existing_predictions, str):
            try:
                existing_predictions = json.loads(existing_predictions)
            except json.JSONDecodeError:
                existing_predictions = {}

        existing_predictions_by_interval = self._group_predictions_by_interval(existing_predictions)

        # Log the number of predictions being saved
        total_new_predictions = sum(
            len(new_predictions[interval_id]['predictions'] if isinstance(new_predictions[interval_id], dict) and 'predictions' in new_predictions[interval_id] else new_predictions[interval_id])
            for interval_id in new_predictions
        )
        bittensor.logging.debug(f"Saving {total_new_predictions} new predictions across {len(new_predictions)} intervals")

        merged_predictions = self._merge_new_with_existing_predictions(existing_predictions_by_interval, new_predictions)

        # bittensor.logging.info(f"Saving merged predictions: {merged_predictions}")
        self.save_data(data=merged_predictions, prefix=prefix)

    def _group_predictions_by_interval(self, predictions: dict) -> dict:
        """Groups predictions by their interval ID.

        Organizes a dictionary of predictions into a dictionary keyed by interval ID.

        Args:
            predictions: A dictionary of predictions.

        Returns:
            A dictionary where each key is an interval ID and the value is a list of predictions for that interval.
        """
        if not predictions:
            return {}

        # If predictions is already in the correct format, return it
        if all(isinstance(v, dict) and "predictions" in v for v in predictions.values()):
            return predictions

        predictions_by_interval = {}
        for interval_id, data in predictions.items():
            if isinstance(data, dict) and "predictions" in data:
                predictions_by_interval[interval_id] = data["predictions"]
            else:
                predictions_by_interval[interval_id] = data if isinstance(data, list) else [data] if data is not None else []
        return predictions_by_interval

    def _merge_new_with_existing_predictions(self, existing_predictions_by_interval: dict, new_predictions: dict) -> dict:
        """Merges new predictions into the existing predictions grouped by interval.

        Adds new predictions to the appropriate interval group, overwriting existing predictions
        for the same miner_uid AND prediction_id combination.

        Args:
            existing_predictions_by_interval: A dictionary of existing predictions grouped by interval ID.
            new_predictions: A dictionary of new prediction dictionaries to merge.
        """
        for interval_id in new_predictions:
            new_predictions_list = new_predictions[interval_id]['predictions'] if isinstance(new_predictions[interval_id], dict) and 'predictions' in new_predictions[interval_id] else new_predictions[interval_id]

            if interval_id in existing_predictions_by_interval:
                existing_predictions = existing_predictions_by_interval[interval_id]

                # Process new predictions
                for new_prediction in new_predictions_list:
                    # Convert to dict if it's an object with .dict() method
                    if hasattr(new_prediction, "dict"):
                        prediction_dict = new_prediction.dict()
                    else:
                        prediction_dict = new_prediction

                    new_miner_uid = prediction_dict.get("miner_uid")
                    new_prediction_id = prediction_dict.get("prediction_id")

                    # Count existing predictions that will be removed (same miner_uid AND prediction_id)
                    removed_count = len([
                        pred for pred in existing_predictions
                        if pred.get("miner_uid") == new_miner_uid and pred.get("prediction_id") == new_prediction_id
                    ])

                    if removed_count > 0:
                        bittensor.logging.debug(f"Removing {removed_count} duplicate predictions for miner_uid={new_miner_uid}, prediction_id={new_prediction_id}")

                    # Remove existing prediction with same miner_uid AND prediction_id if it exists
                    existing_predictions = [
                        pred for pred in existing_predictions
                        if not (pred.get("miner_uid") == new_miner_uid and pred.get("prediction_id") == new_prediction_id)
                    ]

                    # Add the new prediction
                    existing_predictions.append(prediction_dict)

                existing_predictions_by_interval[interval_id] = existing_predictions
            else:
                # Create new interval
                predictions_list = []
                for prediction in new_predictions_list:
                    if hasattr(prediction, "dict"):
                        pred_data = prediction.dict()
                    else:
                        pred_data = prediction
                    predictions_list.append(pred_data)
                existing_predictions_by_interval[interval_id] = predictions_list

        return existing_predictions_by_interval

    def load_predictions(self) -> list:
        """Load the latest saved validator predictions.

        Returns:
            A list of predictions.
        """
        prefix = f"{self.validator_id}_predictions"
        return self.load_data(prefix=prefix)

    def load_predictions_by_interval_id(self, interval_id: str) -> list:
        """Load the latest saved validator predictions for a given interval_id.

        Args:
            interval_id: The interval_id to load predictions for.

        Returns:
            A list of predictions for the given interval_id.
        """
        prefix = f"{self.validator_id}_predictions"
        predictions = self.load_data(prefix=prefix)
        return [] if predictions is None else predictions.get(interval_id, [])

