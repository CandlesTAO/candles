from datetime import datetime, timezone
import bittensor
import pandas as pd
from ..core.data import CandlePrediction, TimeInterval
from ..core.utils import get_next_timestamp_by_interval

def get_file_predictions(filename="hourly_predictions.csv", interval=TimeInterval.HOURLY, miner_uid=None, hotkey=None) -> list[CandlePrediction]:
    """
    Reads candle predictions from a CSV file and converts them to CandlePrediction objects.

    This function loads prediction data from a CSV file, filters predictions to only include
    those with timestamps greater than or equal to the next interval timestamp, and converts
    each row into a CandlePrediction object with the specified miner information.

    Args:
        filename (str): Path to the CSV file containing prediction data.
                       Defaults to "hourly_predictions.csv"
        interval (TimeInterval): The time interval for the predictions (e.g., HOURLY, DAILY).
                                Defaults to TimeInterval.HOURLY
        miner_uid (int, optional): The unique identifier for the miner making the predictions.
                                  Defaults to None
        hotkey (str, optional): The hotkey identifier for the miner. Defaults to None

    Returns:
        list[CandlePrediction]: A list of CandlePrediction objects representing the filtered
                               and converted predictions from the CSV file

    The CSV file is expected to have columns:
        - timestamp: Unix timestamp for the prediction interval
        - color: The predicted candle color (Red/Green)
        - confidence: Confidence score for the prediction (0.0-1.0)
        - price: The predicted price at the end of the interval
    """
    # Get the next timestamp for the specified interval
    # This ensures we only return predictions for future intervals
    next_timestamp = get_next_timestamp_by_interval(interval)

    # Load the CSV file into a pandas DataFrame
    # This reads all prediction data from the specified file
    predictions: pd.DataFrame = pd.read_csv(filename)

    # Filter predictions to only include those with timestamps >= next_timestamp
    # This ensures we only return predictions for intervals that haven't started yet
    predictions: pd.DataFrame = predictions[predictions['timestamp'] >= next_timestamp]

    # Convert the filtered DataFrame to a list of dictionaries
    # Each dictionary represents one row from the CSV with column names as keys
    predictions: list[dict] = predictions.to_dict(orient="records")

    # Convert each dictionary to a CandlePrediction object
    # This uses the build_prediction helper function to create properly formatted objects
    # with the specified interval, miner_uid, and hotkey information
    predictions: list[CandlePrediction] = [
        build_prediction(
            color=prediction["color"],
            price=prediction["price"],
            confidence=prediction["confidence"],
            timestamp=prediction["timestamp"],
            interval=interval,  # Add the specified interval
            miner_uid=miner_uid,  # Add the miner UID
            hotkey=hotkey  # Add the miner hotkey
        )
        for prediction in predictions
    ]

    # Return the list of CandlePrediction objects
    # These can be used by the miner to respond to validator prediction requests
    return predictions

def get_random_prediction(interval=TimeInterval.HOURLY, miner_uid=None, hotkey=None) -> CandlePrediction:

    import random
    from decimal import Decimal
    from ..core.data import CandleColor

    bittensor.logging.debug(f"Making prediction for interval: {interval}")

    # Generate a random price between 100 and 1000
    price = Decimal(str(random.uniform(100, 1000)))
    bittensor.logging.debug(f"Generated price: {price}")

    # Randomly choose a color
    color = random.choice([CandleColor.RED, CandleColor.GREEN])
    bittensor.logging.debug(f"Generated color: {color}")

    # Generate a random confidence between 0.5 and 1.0
    confidence = Decimal(str(random.uniform(0.5, 1.0)))
    bittensor.logging.debug(f"Generated confidence: {confidence}")
    interval_id = f"{get_next_timestamp_by_interval(interval)}::{interval}"
    return build_prediction(price, color, confidence, interval, interval_id, miner_uid, hotkey)

def build_prediction(price, color, confidence, interval, timestamp, miner_uid, hotkey) -> CandlePrediction:
    """
    Builds a CandlePrediction object from the provided parameters.

    This function takes raw prediction data and constructs a properly formatted
    CandlePrediction object that can be used by the miner to respond to validator
    requests. It handles data type conversions and ensures all required fields
    are properly set.

    Args:
        price: The predicted price for the candle interval
        color: The predicted color of the candle (string representation)
        confidence: The confidence level of the prediction (0.0 to 1.0)
        interval: The time interval for the prediction (e.g., HOURLY, DAILY)
        timestamp: The timestamp for the prediction interval
        miner_uid: The unique identifier of the miner making the prediction
        hotkey: The hotkey of the miner making the prediction

    Returns:
        CandlePrediction: A fully constructed prediction object with all fields set

    Note:
        The color parameter is converted to lowercase and then to a CandleColor enum
        to ensure consistent formatting regardless of input case.
    """
    # Import the CandleColor enum from the core data module
    # This is done locally to avoid circular import issues
    from ..core.data import CandleColor

    # Convert the color string to lowercase and create a CandleColor enum instance
    # This ensures consistent color representation regardless of input case
    # (e.g., "RED", "red", "Red" all become CandleColor.RED)
    color = CandleColor(color.lower())

    # Construct and return a new CandlePrediction object with all the provided parameters
    # This creates a complete prediction that can be sent to validators
    return CandlePrediction(
        price=price,                    # The predicted price value
        color=color,                    # The normalized candle color enum
        confidence=confidence,          # The confidence level of the prediction
        prediction_id=int(timestamp),
        prediction_date=datetime.now(timezone.utc),  # Current UTC timestamp when prediction was made
        interval=interval,              # The time interval for this prediction
        interval_id=f"{timestamp}::{interval}",  # Unique identifier combining timestamp and interval
        miner_uid=miner_uid,            # The UID of the miner making this prediction
        hotkey=hotkey,                  # The hotkey of the miner making this prediction
    )
