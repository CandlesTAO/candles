# Standard Lib
import asyncio
import os
import traceback
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Third Party
import aiofiles

# Third Party
import numpy as np
from ..core.data import CandlePrediction, TimeInterval
from ..core.synapse import GetCandlePrediction


# Bittensor
import bittensor

# Local
from ..core.utils import get_next_timestamp_by_interval
from .base import BaseValidatorNeuron
from .utils import get_miner_uids, send_predictions_to_miners
from .storage import JsonValidatorStorage
from .sqlite_storage import SQLiteValidatorStorage
from ..core.scoring.batch_scorer import PredictionBatchScorer
from ..prices.client import PriceClient
from ..core.services.candlestao_client import CandleTAOClient, CandleTAOPredictionSubmission, CandleTAOScoreSubmission, CandleTAOMinerScoreSubmission
import math

class Validator(BaseValidatorNeuron):
    """
    The Validator class manages the process of requesting, collecting, and scoring predictions from miners.
    It handles the scheduling of prediction requests, scoring of miner responses, and updating of weights based on performance.

    The Validator operates as a neuron in the network, periodically querying miners for predictions, scoring their responses, and maintaining state.
    It also manages the storage of predictions and the incentive mechanism for miners.

    Args:
        config: Optional configuration object for the validator.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bittensor.logging.info("load_state()")
        self.load_state()
        self.storage = JsonValidatorStorage(config=self.config)

        # Initialize SQLite storage for scores over time
        self.sqlite_storage = SQLiteValidatorStorage(config=self.config)
        api_key = os.getenv("COINDESK_API_KEY")
        if not api_key:
            raise ValueError("COINDESK_API_KEY is not set")
        self.incentive_scoring_interval = int(os.getenv("INCENTIVE_SCORING_INTERVAL", 30))
        # Initialize scoring components
        self.price_client = PriceClient(api_key=api_key, provider="coindesk")
        self.batch_scorer = PredictionBatchScorer(self.price_client)

        # Initialize CandleTAO client
        try:
            self.candletao_client = CandleTAOClient()
            bittensor.logging.info("CandleTAO client initialized successfully")
        except Exception as e:
            bittensor.logging.warning(f"CandleTAO client initialization failed: {e}")
            self.candletao_client = None

        # Background task management
        self.scoring_task = None

    async def async_init(self):
        """
        Async initialization for validator. Must be called after __init__.
        """
        # Initialize the base validator async components
        await super().async_init()

        # Start background tasks (metagraph sync and weight setting)
        self.start_background_tasks()

        # Start background scoring task
        self.start_background_scoring()

    def start_background_scoring(self):
        """
        Start the background scoring task based on subnet-22 patterns.
        """
        if not self.config.mock:
            # Create background task for incentive scoring
            self.scoring_task = self.loop.create_task(self._incentive_scoring_and_set_weights_async())
            bittensor.logging.info("Started background scoring task")

    @classmethod
    def _intervals_to_score(cls) -> list[str]:
        """
        Determines which time intervals are ready to be scored based on the current time.

        This method calculates which hourly, daily, and weekly intervals have closed
        and are now available for scoring. An interval is considered "closed" when
        the current time has passed beyond the start of the next interval.

        Returns:
            list[str]: A list of interval IDs in the format "timestamp::interval_type"
                      that are ready for scoring. Each interval ID represents a
                      completed time period that can now be evaluated against actual
                      market data.

        Example:
            If current time is 14:30 UTC on Monday, this method would return:
            - The previous hour's interval (13:00-14:00) if it's past 14:00
            - The previous day's interval (Sunday 00:00-24:00) if it's past 00:00
            - The previous week's interval if it's past Monday 00:00
        """
        # Get current UTC time to ensure consistent timezone handling
        now = datetime.now(timezone.utc)
        current_timestamp = now.timestamp()

        return [
            cls._get_closed_hourly_intervals(now, current_timestamp),
            cls._get_closed_daily_intervals(now, current_timestamp),
            cls._get_closed_weekly_intervals(now, current_timestamp),
        ]

    @classmethod
    def _get_closed_hourly_intervals(cls, now: datetime, current_timestamp: float) -> str | None:
        """
        Determines which hourly intervals have closed and are ready for scoring.

        This method calculates the previous hour's interval that has completed
        and can now be evaluated against actual market data. An hourly interval
        is considered "closed" when the current time has passed beyond the start
        of the current hour (i.e., we're no longer in the first minute of the hour).

        Args:
            now (datetime): Current UTC datetime object
            current_timestamp (float): Current timestamp in seconds since epoch

        Returns:
            str: The interval ID of the previous hour if it's ready for scoring,
                 otherwise None. The interval ID format is "timestamp::hourly"
                 where timestamp represents the start time of the completed hour.

        Example:
            If current time is 14:30 UTC, this method would return:
            ["1704067200::hourly"] (representing the 13:00-14:00 interval)

            If current time is 14:00 UTC (exactly on the hour), this method would
            return an empty list since the 13:00-14:00 interval hasn't fully closed yet.
        """
        # Calculate the start of the current hour by zeroing out minutes, seconds, and microseconds
        # This gives us the timestamp for the beginning of the current hour
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        # Check if we've moved past the start of the current hour
        # This ensures the previous hour's interval has fully completed
        if current_timestamp > hour_start.timestamp():
            # Calculate the previous hour by subtracting 1 hour from the current hour start
            # This gives us the start time of the completed hour interval
            prev_hour = hour_start - timedelta(hours=1)

            return f"{int(prev_hour.timestamp())}::{TimeInterval.HOURLY}"

    @classmethod
    def _get_closed_daily_intervals(cls, now: datetime, current_timestamp: float) -> str | None:
        """
        Returns a list of closed daily interval IDs based on the current time.
        """
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if current_timestamp > day_start.timestamp():
            prev_day = day_start - timedelta(days=1)
            return f"{int(prev_day.timestamp())}::{TimeInterval.DAILY}"

    @classmethod
    def _get_closed_weekly_intervals(cls, now: datetime, current_timestamp: float) -> str | None:
        """
        Returns a list of closed weekly interval IDs based on the current time.
        """
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        days_since_monday = now.weekday()
        week_start = day_start - timedelta(days=days_since_monday)
        if current_timestamp > week_start.timestamp():
            prev_week = week_start - timedelta(weeks=1)
            return f"{int(prev_week.timestamp())}::{TimeInterval.WEEKLY}"

    @classmethod
    def _should_request_hourly(cls, now: datetime, next_hour: float) -> bool:
        # if the next hour is more than 10 minutes away, request the next hourly candle
        return next_hour - now.timestamp() > 600  # 10 minutes

    @classmethod
    def _should_request_daily(cls, now: datetime, next_day: float) -> bool:
        # if the next day is more than 1 hour away, request the next daily candle
        return next_day - now.timestamp() > 3600  # 1 hour

    @classmethod
    def _should_request_weekly(cls, now: datetime, next_week: float) -> bool:
        # if the next week is more than 1 day away, request the next weekly candle
        return next_week - now.timestamp() > 86400  # 1 day

    @classmethod
    def get_next_candle_prediction_requests(cls) -> list[CandlePrediction]:
        """
        Determines which candle prediction requests to make based on the current time.

        Returns:
            list[CandlePrediction]: List of CandlePrediction requests
        """
        now = datetime.now(timezone.utc)
        next_hour = get_next_timestamp_by_interval(TimeInterval.HOURLY)
        next_day = get_next_timestamp_by_interval(TimeInterval.DAILY)
        next_week = get_next_timestamp_by_interval(TimeInterval.WEEKLY)

        def make_candle_prediction(interval, next_time):
            return CandlePrediction(
                interval=interval,
                interval_id=f"{next_time}::{interval}",
                prediction_id=next_time,
            )

        prediction_requests = []
        if cls._should_request_hourly(now, next_hour):
            prediction_requests.append(make_candle_prediction(TimeInterval.HOURLY, next_hour))
        if cls._should_request_daily(now, next_day):
            prediction_requests.append(make_candle_prediction(TimeInterval.DAILY, next_day))
        if cls._should_request_weekly(now, next_week):
            prediction_requests.append(make_candle_prediction(TimeInterval.WEEKLY, next_week))

        return prediction_requests

    async def _incentive_scoring_and_set_weights_async(self):
        """
        Continuously scores closed intervals and sets weights using async patterns.

        This method runs as a background async task and periodically executes the scoring
        and weight setting workflow. It operates on a configurable interval cycle, performing
        the following operations:
        1. Checks if it's time to run scoring (based on interval)
        2. Executes the scoring algorithm on closed intervals
        3. Updates validator weights based on miner performance
        4. Handles any errors that occur during the process

        The method uses a simple time-based scheduling mechanism where:
        - Scoring occurs when the current minute is divisible by interval (e.g., 00, 30)
        - The task sleeps for 60 seconds between iterations to avoid excessive CPU usage
        """
        # Main loop that continues until the validator should exit
        # This allows for graceful shutdown of the background task
        while not self.should_exit:
            try:
                # Get current UTC time to ensure consistent timezone handling
                # This is important for accurate scheduling across different timezones
                current_time = datetime.now(timezone.utc)
                minutes = current_time.minute

                # Check if it's time to run the scoring and weight setting process
                if minutes % self.incentive_scoring_interval == 0:
                    try:
                        # Log the start of the scoring process for monitoring and debugging
                        # The asterisks make it easy to spot in logs
                        bittensor.logging.info("[orange]*** Starting incentive scoring and weight setting ***[/orange]")

                        # Execute the main scoring and weight update workflow
                        # This method handles all the complex logic of scoring miners and updating weights
                        await self._score_and_update_weights_async()

                    except Exception as e:
                        # Comprehensive error handling to prevent the background task from crashing
                        # Log both the error message and full traceback for debugging
                        bittensor.logging.error(f"Error in incentive scoring and weight setting: {e}")
                        bittensor.logging.error(f"Error details: {traceback.format_exc()}")

                # Sleep for 60 seconds before the next iteration
                # This prevents the task from consuming excessive CPU resources
                # while still providing responsive scheduling
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                bittensor.logging.info("Background scoring task cancelled")
                break
            except Exception as e:
                bittensor.logging.error(f"Unexpected error in background scoring task: {e}")
                await asyncio.sleep(60)

    async def _write_scoring_results_to_file(self, scoring_results: dict[str, list], timestamp: datetime | None = None) -> None:
        """
        Write scoring results to a JSON file with interval_ids as top-level keys.

        Args:
            scoring_results: Dictionary of scoring results grouped by interval_id
            timestamp: Optional timestamp for the scoring run (defaults to current time)
        """
        if not scoring_results:
            bittensor.logging.debug("No scoring results to write to file")
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Create the data directory if it doesn't exist
        data_dir = Path.home() / ".candles" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Use a single filename for all scoring results
        filename = "scoring_results.json"
        filepath = data_dir / filename

        try:
            # Convert ScoringResult objects to dictionaries for JSON serialization
            serializable_results = {}
            for interval_id, results in scoring_results.items():
                serializable_results[interval_id] = [
                    {
                        'prediction_id': result.prediction_id,
                        'miner_uid': result.miner_uid,
                        'interval_id': result.interval_id,
                        'color_score': result.color_score,
                        'price_score': result.price_score,
                        'confidence_weight': result.confidence_weight,
                        'final_score': result.final_score,
                        'actual_color': result.actual_color,
                        'actual_price': result.actual_price,
                        'timestamp': timestamp.isoformat()
                    }
                    for result in results
                ]

            # Read existing data if file exists
            existing_data = {}
            if filepath.exists():
                try:
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        if content.strip():
                            existing_data = json.loads(content)
                            if not isinstance(existing_data, dict):
                                # If the file contains a different format, start fresh
                                existing_data = {}
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    bittensor.logging.warning(f"Error reading existing scoring file: {e}. Starting fresh.")
                    existing_data = {}

            # Merge new results with existing data
            for interval_id, new_results in serializable_results.items():
                if interval_id in existing_data:
                    # Add new results to existing interval
                    existing_data[interval_id].extend(new_results)
                else:
                    # Create new interval entry
                    existing_data[interval_id] = new_results

            # Write back to file asynchronously
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(existing_data, indent=2))

            bittensor.logging.info(f"Scoring results appended to: {filepath}")

        except Exception as e:
            bittensor.logging.error(f"Error writing scoring results to file: {e}")

    async def _save_scoring_results_to_sqlite(self, scoring_results: dict[str, list]) -> None:
        """
        Save scoring results to SQLite database for analysis.

        Args:
            scoring_results: Dictionary of scoring results grouped by interval_id
        """
        if not scoring_results:
            bittensor.logging.debug("No scoring results to save to SQLite")
            return

        # Convert ScoringResult objects to dictionaries for SQLite storage
        sqlite_scoring_results = {}
        for interval_id, results in scoring_results.items():
            sqlite_scoring_results[interval_id] = []
            for result in results:
                sqlite_scoring_results[interval_id].append({
                    'prediction_id': result.prediction_id,
                    'miner_uid': result.miner_uid,
                    'color_score': result.color_score,
                    'price_score': result.price_score,
                    'confidence_weight': result.confidence_weight,
                    'final_score': result.final_score,
                    'actual_color': result.actual_color,
                    'actual_price': result.actual_price
                })

        # Save to SQLite
        self.sqlite_storage.save_scoring_results(sqlite_scoring_results)
        total_results = sum(len(results) for results in sqlite_scoring_results.values())
        bittensor.logging.debug(f"Saved {total_results} scoring results to SQLite")

    def _convert_predictions_to_candletao_format(self, predictions_data: dict[str, list[dict]]) -> list[CandleTAOPredictionSubmission]:
        """Convert predictions data to CandleTAO submission format."""
        submissions = []

        for interval_id, predictions in predictions_data.items():
            for prediction_dict in predictions:
                try:
                    submission = CandleTAOPredictionSubmission(
                        prediction_id=prediction_dict['prediction_id'],
                        miner_uid=prediction_dict['miner_uid'],
                        hotkey=prediction_dict['hotkey'],
                        prediction_date=datetime.fromisoformat(prediction_dict['prediction_date'].replace('Z', '+00:00')),
                        interval_id=interval_id,
                        is_closed=prediction_dict.get('is_closed', False),
                        closed_date=datetime.fromisoformat(prediction_dict['closed_date'].replace('Z', '+00:00')) if prediction_dict.get('closed_date') else None,
                        interval=prediction_dict['interval'],
                        color=prediction_dict['color'],
                        price=str(prediction_dict['price']),
                        confidence=str(prediction_dict['confidence'])
                    )
                    submissions.append(submission)
                except Exception as e:
                    bittensor.logging.error(f"Error converting prediction to CandleTAO format: {e}")
                    continue

        return submissions

    def _convert_scores_to_candletao_format(self, scoring_results: dict[str, list]) -> list[CandleTAOScoreSubmission]:
        """Convert scoring results to CandleTAO submission format."""
        submissions = []
        timestamp = datetime.now(timezone.utc)

        for interval_id, results in scoring_results.items():
            for result in results:
                try:
                    submission = CandleTAOScoreSubmission(
                        prediction_id=result.prediction_id,
                        miner_uid=result.miner_uid,
                        interval_id=interval_id,
                        color_score=result.color_score,
                        price_score=result.price_score,
                        confidence_weight=result.confidence_weight,
                        final_score=result.final_score,
                        actual_color=result.actual_color,
                        actual_price=result.actual_price,
                        timestamp=timestamp
                    )
                    submissions.append(submission)
                except Exception as e:
                    bittensor.logging.error(f"Error converting score to CandleTAO format: {e}")
                    continue

        return submissions

    async def _submit_predictions_to_candletao(self, predictions_data: dict[str, list[dict]]) -> None:
        """Submit predictions to CandleTAO API."""
        if not self.candletao_client:
            bittensor.logging.debug("CandleTAO client not available, skipping prediction submission")
            return

        try:
            submissions = self._convert_predictions_to_candletao_format(predictions_data)
            if submissions:
                bittensor.logging.info(f"Submitting {len(submissions)} predictions to CandleTAO")
                response = await self.candletao_client.submit_predictions(submissions)
                bittensor.logging.info(f"Successfully submitted predictions to CandleTAO: {response}")
            else:
                bittensor.logging.debug("No predictions to submit to CandleTAO")
        except Exception as e:
            bittensor.logging.error(f"Error submitting predictions to CandleTAO: {e}")

    async def _submit_scores_to_candletao(self, scoring_results: dict[str, list]) -> None:
        """Submit scores to CandleTAO API."""
        bittensor.logging.debug(f"Attempting to submit scores to CandleTAO. Client available: {self.candletao_client is not None}")

        if not self.candletao_client:
            bittensor.logging.warning("CandleTAO client not available, skipping score submission")
            return

        try:
            bittensor.logging.debug(f"Converting {len(scoring_results)} scoring result intervals to CandleTAO format")
            submissions = self._convert_scores_to_candletao_format(scoring_results)
            bittensor.logging.debug(f"Converted to {len(submissions) if submissions else 0} score submissions")

            if submissions:
                bittensor.logging.info(f"Submitting {len(submissions)} scores to CandleTAO API endpoint")
                response = await self.candletao_client.submit_scores(submissions)
                bittensor.logging.info(f"Successfully submitted scores to CandleTAO: {response}")
            else:
                bittensor.logging.warning("No scores to submit to CandleTAO after conversion")
        except Exception as e:
            bittensor.logging.error(f"Error submitting scores to CandleTAO: {e}")

    async def _submit_decay_adjusted_scores_to_candletao(self, scoring_results: dict[str, list], decay_adjusted_scores: dict[int, float]) -> None:
        """Submit decay-adjusted scores to CandleTAO API."""
        bittensor.logging.debug(f"Attempting to submit decay-adjusted scores to CandleTAO. Client available: {self.candletao_client is not None}")

        if not self.candletao_client:
            bittensor.logging.warning("CandleTAO client not available, skipping decay-adjusted score submission")
            return

        try:
            bittensor.logging.debug(f"Converting {len(scoring_results)} scoring result intervals to CandleTAO format with decay adjustment")
            submissions = self._convert_miner_scores_to_candletao_format(decay_adjusted_scores)
            bittensor.logging.debug(f"Converted to {len(submissions) if submissions else 0} decay-adjusted score submissions")

            if submissions:
                bittensor.logging.info(f"Submitting {len(submissions)} decay-adjusted miner scores to CandleTAO miner-scores endpoint")
                response = await self.candletao_client.submit_miner_scores(submissions)
                bittensor.logging.info(f"Successfully submitted decay-adjusted scores to CandleTAO: {response}")
            else:
                bittensor.logging.warning("No decay-adjusted miner scores to submit to CandleTAO after conversion")
        except Exception as e:
            bittensor.logging.error(f"Error submitting decay-adjusted scores to CandleTAO: {e}")

    def _convert_scores_to_candletao_format_with_decay(self, scoring_results: dict[str, list], decay_adjusted_scores: dict[int, float]) -> list[CandleTAOScoreSubmission]:
        """Convert scoring results to CandleTAO submission format with decay-adjusted final scores."""
        submissions = []
        timestamp = datetime.now(timezone.utc)

        for interval_id, results in scoring_results.items():
            for result in results:
                try:
                    # Use decay-adjusted score if available, otherwise fall back to original score
                    final_score = decay_adjusted_scores.get(result.miner_uid, result.final_score)
                    
                    submission = CandleTAOScoreSubmission(
                        prediction_id=result.prediction_id,
                        miner_uid=result.miner_uid,
                        interval_id=interval_id,
                        color_score=result.color_score,
                        price_score=result.price_score,
                        confidence_weight=result.confidence_weight,
                        final_score=final_score,  # Use decay-adjusted score here
                        actual_color=result.actual_color,
                        actual_price=result.actual_price,
                        timestamp=timestamp
                    )
                    submissions.append(submission)
                except Exception as e:
                    bittensor.logging.error(f"Error converting score to CandleTAO format with decay: {e}")
                    continue

        return submissions

    def _convert_miner_scores_to_candletao_format(self, decay_adjusted_scores: dict[int, float]) -> list[CandleTAOMinerScoreSubmission]:
        """Convert decay-adjusted miner scores to CandleTAO submission format."""
        submissions = []

        for miner_uid, score in decay_adjusted_scores.items():
            try:
                # For miner score submissions, we need the last prediction ID that contributed to this score
                # We can get this from the SQLite storage or use a placeholder
                last_prediction_id = self._get_last_prediction_id_for_miner(miner_uid)
                
                submission = CandleTAOMinerScoreSubmission(
                    miner_uid=miner_uid,
                    score=score,
                    last_scored_prediction_id=last_prediction_id
                )
                submissions.append(submission)
            except Exception as e:
                bittensor.logging.error(f"Error converting miner score to CandleTAO format: {e}")
                continue

        return submissions

    def _get_last_prediction_id_for_miner(self, miner_uid: int) -> int:
        """Get the last prediction ID for a miner that contributed to their current score."""
        try:
            # Try to get from SQLite storage first
            if hasattr(self, 'sqlite_storage') and self.sqlite_storage:
                last_prediction_id = self.sqlite_storage.get_last_prediction_id_for_miner(miner_uid)
                if last_prediction_id:
                    return last_prediction_id
            
            # Fallback: use current timestamp as prediction ID
            from datetime import datetime, timezone
            return int(datetime.now(timezone.utc).timestamp())
        except Exception as e:
            bittensor.logging.warning(f"Error getting last prediction ID for miner {miner_uid}: {e}")
            # Final fallback
            from datetime import datetime, timezone
            return int(datetime.now(timezone.utc).timestamp())

    async def _score_and_update_weights_async(self) -> None:
        """
        Async version of score_and_update_weights that scores closed intervals and updates validator weights.

        This method orchestrates the complete scoring workflow:
        1. Identifies closed intervals that need scoring
        2. Loads predictions for those intervals from storage
        3. Runs scoring algorithms to evaluate miner performance
        4. Updates validator scores based on miner performance
        5. Sets weights on the blockchain (if enabled)
        """

        if closed_intervals := self._intervals_to_score():
            bittensor.logging.info(f"Scoring {len(closed_intervals)} closed intervals: {closed_intervals}")

            # Step 2: Load predictions for the closed intervals from persistent storage
            # This retrieves all miner predictions that were made for these time periods
            if predictions_data := self._load_predictions_for_intervals(closed_intervals):
                try:
                    # Submit predictions to CandleTAO before scoring
                    await self._submit_predictions_to_candletao(predictions_data)

                    # Step 3: Execute the scoring algorithm asynchronously
                    # This evaluates how accurate each miner's predictions were
                    bittensor.logging.debug(f"Running scoring async for {len(predictions_data)} intervals")
                    scoring_results = await self.batch_scorer.score_predictions_by_interval(predictions_data)

                    # Write scoring results to file
                    await self._write_scoring_results_to_file(scoring_results)

                    # Save scoring results to SQLite for analysis
                    try:
                        await self._save_scoring_results_to_sqlite(scoring_results)
                    except Exception as e:
                        bittensor.logging.error(f"Error saving scoring results to SQLite: {e}")

                    # Step 4: Extract miner scores from the scoring results
                    # This converts the raw scoring data into a format suitable for weight updates
                    miner_scores = self.batch_scorer.get_miner_scores(scoring_results)

                    # Step 5: Update the validator's internal scoring system and get decay-adjusted scores
                    # This maintains historical performance data for each miner
                    decay_adjusted_scores = self._update_validator_scores(miner_scores)

                    # Submit decay-adjusted scores to CandleTAO after calculating decay
                    if os.getenv("CANDLETAO_BEARER_TOKEN"):
                        bittensor.logging.info("CANDLETAO_BEARER_TOKEN found, submitting decay-adjusted scores to CandleTAO")
                        await self._submit_decay_adjusted_scores_to_candletao(scoring_results, decay_adjusted_scores)
                    else:
                        bittensor.logging.warning("CandleTAO API key not set, skipping score submission")

                    # Step 6: Log the top performing miners for monitoring purposes
                    # This helps with debugging and performance analysis
                    top_miners = self.batch_scorer.get_top_miners(miner_scores, limit=5)
                    bittensor.logging.info(f":arrow_right: Top performing miners: [magenta]{top_miners}[/magenta]")

                except Exception as e:
                    # Handle any errors that occur during the scoring process
                    # This prevents the entire weight update cycle from failing
                    bittensor.logging.error(f"Error during scoring: {e}")
            else:
                # Log when no predictions are found for the closed intervals
                # This could happen if miners didn't submit predictions or storage issues
                bittensor.logging.info("No predictions found for closed intervals")
                return
        else:
            # Log when there are no intervals ready for scoring
            # This is normal behavior when intervals haven't closed yet
            bittensor.logging.debug("No closed intervals to score")

        # Step 7: Set weights on the blockchain (if enabled and not in offline mode)
        # This is the final step that actually updates the network's consensus
        if not self.config.neuron.disable_set_weights and not self.config.offline:
            bittensor.logging.info("Setting weights based on scores")
            if self.config.mock:
                self.set_weights()
            else:
                await self.set_weights_async()
            bittensor.logging.info("Successfully set weights")

    def _load_predictions_for_intervals(self, closed_intervals: list[str]) -> dict[str, list[dict]]:
        """
        Loads predictions for the closed intervals from persistent storage.

        Args:
            closed_intervals: List of closed interval IDs

        Returns:
            Dictionary of predictions for the closed intervals
        """
        predictions_data = {}
        for interval_id in closed_intervals:

            bittensor.logging.debug(f"Loading predictions for interval {interval_id}")
            try:
                if interval_predictions := self.storage.load_predictions_by_interval_id(
                    interval_id
                ):
                    predictions_data[interval_id] = interval_predictions
                    bittensor.logging.debug(f"Loaded {len(interval_predictions)} predictions for interval {interval_id}")
            except Exception as e:
                bittensor.logging.error(f"Error loading predictions for interval {interval_id}: {e}")
        return predictions_data


    def _update_validator_scores(self, miner_scores: dict[int, dict[str, float]]) -> dict[int, float]:
        """
        Update the validator's scores based on miner performance.
        Implements decay-based scoring according to the scoring diagram:
        1. Collect historical daily scores for up to 31 days
        2. Sum all historical daily final scores
        3. Divide by number of days since registration (max 31)

        Args:
            miner_scores: Dictionary of miner scores from batch scorer (today's scores)
            
        Returns:
            Dictionary mapping miner_uid to decay_adjusted_score
        """
        if not miner_scores:
            bittensor.logging.warning("No miner scores to update")
            return {}

        # Create rewards array
        rewards = np.zeros(self.metagraph.n, dtype=np.float32)
        uids = []
        decay_adjusted_scores = {}

        # Prepare data for SQLite storage
        sqlite_miner_scores = {}
        sqlite_miner_stats = {}
        sqlite_days_since_reg = {}
        sqlite_hotkeys = {}

        for miner_uid, stats in miner_scores.items():
            if 0 <= miner_uid < self.metagraph.n:
                # Current day's final score (will be saved to history)
                current_daily_score = stats.get('average_score', 0.0)

                # Calculate decay-adjusted score using historical aggregation
                decay_adjusted_score = self._calculate_historical_decay_score(miner_uid, current_daily_score)

                rewards[miner_uid] = decay_adjusted_score
                decay_adjusted_scores[miner_uid] = decay_adjusted_score
                uids.append(miner_uid)

                # Prepare data for SQLite storage (store the raw daily score, not decay-adjusted)
                sqlite_miner_scores[miner_uid] = decay_adjusted_score  # Final result for current scores table
                sqlite_miner_stats[miner_uid] = {
                    **stats,
                    'score': current_daily_score  # Store today's raw score in history
                }

                # Get hotkey if available
                if hasattr(self, 'metagraph') and self.metagraph and miner_uid < len(self.metagraph.hotkeys):
                    sqlite_hotkeys[miner_uid] = self.metagraph.hotkeys[miner_uid]

                bittensor.logging.debug(
                    f"Miner {miner_uid}: current_daily_score={current_daily_score:.4f}, "
                    f"decay_adjusted_score={decay_adjusted_score:.4f}, "
                    f"predictions={stats.get('prediction_count', 0)}, "
                    f"color_acc={stats.get('color_accuracy', 0):.3f}, "
                    f"price_acc={stats.get('price_accuracy', 0):.3f}"
                )

        if uids:
            # Update scores using the base validator's method
            self.update_scores(rewards[uids], uids)
            bittensor.logging.info(f"Updated scores for {len(uids)} miners rewards[uids]: {rewards[uids]}, uids: {uids}")

            # Save scores to SQLite for historical tracking
            try:
                self.sqlite_storage.save_miner_scores(sqlite_miner_scores, sqlite_hotkeys)
                self.sqlite_storage.save_score_history(sqlite_miner_stats, sqlite_days_since_reg)
                bittensor.logging.debug(f"Saved {len(sqlite_miner_scores)} miner scores to SQLite storage")
            except Exception as e:
                bittensor.logging.error(f"Error saving scores to SQLite: {e}")
        else:
            bittensor.logging.warning("No valid miner UIDs to update scores for")
            
        return decay_adjusted_scores

    def _calculate_historical_decay_score(self, miner_uid: int, current_daily_score: float) -> float:
        """
        Calculate decay-adjusted score based on historical score aggregation with decay factors.

        Implements the decay-and-scoring diagram workflow with proper decay factors:
        1. Get historical daily scores for the last 31 days (or since first score)
        2. Apply decay factors to historical scores (older scores get lower weights)
        3. Sum all weighted historical daily scores + today's score
        4. Divide by min(days_since_first_score, 31)

        Args:
            miner_uid: Miner UID to calculate score for
            current_daily_score: Today's final score to include in calculation

        Returns:
            Decay-adjusted score based on historical aggregation with decay factors
        """
        try:
            # Get historical daily scores from SQLite (up to 31 most recent)
            historical_scores = self.sqlite_storage.get_historical_daily_scores(miner_uid, days=31)

            # Add today's score to the historical scores for complete aggregation
            all_daily_scores = [current_daily_score] + historical_scores

            # Apply 31-day cap: take only the most recent 31 scores (current + up to 30 historical)
            if len(all_daily_scores) > 31:
                all_daily_scores = all_daily_scores[:31]

            # Apply decay factors to historical scores (current day gets full weight)
            weighted_scores = []
            for i, score in enumerate(all_daily_scores):
                if i == 0:
                    # Current day gets full weight (decay factor = 1.0)
                    decay_factor = 1.0
                else:
                    # Historical scores get decayed based on their age
                    decay_factor = self._calculate_decay_factor(i)

                weighted_score = score * decay_factor
                weighted_scores.append(weighted_score)

                bittensor.logging.debug(
                    f"Miner {miner_uid} score {i}: {score:.4f} × {decay_factor:.4f} = {weighted_score:.4f}"
                )

            # Sum all weighted daily scores
            total_weighted_sum = sum(weighted_scores)

            # The denominator should be the number of days we actually have scores for,
            # capped at 31 as per the diagram requirements
            effective_days = len(all_daily_scores)

            # Calculate final decay-adjusted score
            decay_adjusted_score = total_weighted_sum / effective_days

            bittensor.logging.debug(
                f"Miner {miner_uid} historical decay calculation: "
                f"daily_scores_count={len(all_daily_scores)}, "
                f"total_weighted_sum={total_weighted_sum:.4f}, "
                f"effective_days={effective_days}, "
                f"final_score={decay_adjusted_score:.4f}"
            )

            return decay_adjusted_score

        except Exception as e:
            # Fallback to simple calculation if SQLite history is not available
            bittensor.logging.warning(
                f"Error calculating historical decay score for miner {miner_uid}: {e}. "
                f"Falling back to simple decay calculation."
            )

            # Fallback: use registration-based days (as before)
            days_since_registration = self._get_days_since_registration(miner_uid)
            return current_daily_score / days_since_registration

    def _calculate_decay_factor(self, days_ago: int) -> float:
        """
        Calculate decay factor for a score that is 'days_ago' days old.

        Based on the decay-and-scoring table, implements a decay function where:
        - Day 1 (current): decay_factor = 1.0
        - Day 31: decay_factor ≈ 0.374
        - Day 32+: decay_factor = 0.0

        Args:
            days_ago: Number of days ago the score was recorded (1 = yesterday, 2 = day before yesterday, etc.)

        Returns:
            Decay factor between 0.0 and 1.0
        """
        # Based on the table data, implement a decay function
        # The table shows decay factors decreasing over time

        if days_ago <= 0:
            return 1.0  # Current day gets full weight

        if days_ago > 31:
            return 0.0  # Scores older than 31 days get zero weight

        # Implement decay function based on table pattern
        # From the table: Day 1=1.0, Day 31=0.374, Day 32=0.0
        # Using exponential decay: decay_factor = e^(-λ * days_ago)
        # Where λ is chosen to match the table values

        # Calculate λ to match table values
        # At day 31: 0.374 = e^(-λ * 31)
        # λ = -ln(0.374) / 31 ≈ 0.032

        lambda_val = 0.032  # Decay rate parameter
        decay_factor = math.exp(-lambda_val * days_ago)

        # Ensure the factor is within bounds
        decay_factor = max(0.0, min(1.0, decay_factor))

        return decay_factor

    def _get_days_since_registration(self, miner_uid: int) -> int:
        """
        Calculate the number of days since a miner was registered.
        Returns a value between 1 and 31 (capped at 31 as per requirements).

        Args:
            miner_uid: The UID of the miner

        Returns:
            int: Number of days since registration (1-31)
        """
        try:
            # For testing purposes, check if we have proper metagraph access
            if not hasattr(self, 'metagraph') or self.metagraph is None:
                bittensor.logging.debug(f"No metagraph available, defaulting to 1 day for miner {miner_uid}")
                return 1  # Default to 1 day for testing

            # Get current block
            current_block = getattr(self, 'block', None)
            if current_block is None and hasattr(self.metagraph, 'block'):
                current_block = self.metagraph.block

            if current_block is None:
                bittensor.logging.debug(f"No current block available, defaulting to 1 day for miner {miner_uid}")
                return 1

            # Check if miner exists and has valid registration data
            if miner_uid >= len(self.metagraph.active):
                bittensor.logging.warning(f"Miner UID {miner_uid} out of range")
                return 31  # Default to max if invalid

            # Get the miner's last update block (when they were registered/last updated)
            miner_last_update = self.metagraph.last_update[miner_uid]

            if miner_last_update == 0:
                bittensor.logging.debug(f"Miner {miner_uid} has no registration block, using max days")
                return 31

            # Calculate blocks since registration
            blocks_since_registration = current_block - miner_last_update

            # Convert blocks to days (assuming ~12 second block time = 7200 blocks per day)
            BLOCKS_PER_DAY = 7200
            days_since_registration = max(1, blocks_since_registration // BLOCKS_PER_DAY)

            # Cap at 31 days as per requirements
            capped_days = min(31, days_since_registration)

            bittensor.logging.debug(
                f"Miner {miner_uid}: current_block={current_block}, "
                f"last_update_block={miner_last_update}, "
                f"blocks_since_reg={blocks_since_registration}, "
                f"days_since_reg={days_since_registration}, "
                f"capped_days={capped_days}"
            )

            return capped_days

        except Exception as e:
            bittensor.logging.error(f"Error calculating days since registration for miner {miner_uid}: {e}")
            return 1  # Default to 1 day for testing/error cases

    async def cleanup(self):
        """Async cleanup method to stop the scoring task."""
        # Stop background scoring task
        if self.scoring_task and not self.scoring_task.done():
            bittensor.logging.info("Stopping background scoring task")
            self.scoring_task.cancel()
            try:
                await self.scoring_task
            except asyncio.CancelledError:
                pass

        # Call parent cleanup
        await super().cleanup()

    @staticmethod
    def parse_responses(miner_predictions: list[GetCandlePrediction], prediction_requests: list[CandlePrediction]) -> list[dict]:
        """
        Parses the responses from the miners. They are grouped by interval_id.
        """
        parsed_responses = {
            prediction.interval_id: []
            for prediction in prediction_requests
        }
        for miner_prediction in miner_predictions:
            parsed_responses[miner_prediction.candle_prediction.interval_id].append(miner_prediction.candle_prediction)
        return parsed_responses

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the prediction queries
        - Querying the miners for predictions
        - Getting the responses and validating predictions
        - Storing prediction responses

        The forward function is called by the validator every run step.

        It is responsible for querying the network and storing the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        """
        miner_uids = get_miner_uids(self.metagraph, self.uid)
        validator_prediction_requests = self.get_next_candle_prediction_requests()
        if len(validator_prediction_requests) == 0 or len(miner_uids) == 0:
            bittensor.logging.info("No prediction requests to send or no miners to send them to.")
            return

        finished_responses, working_miner_uids = await self._gather_predictions_from_miners(
            validator_prediction_requests, miner_uids
        )

        self.save(
            finished_responses, working_miner_uids, miner_uids, validator_prediction_requests
        )

    async def _gather_predictions_from_miners(self, validator_prediction_requests: list[CandlePrediction], miner_uids: list[int]):
        """
        Sends prediction requests to miners and gathers their responses using concurrent async patterns.

        Args:
            validator_prediction_requests: List of CandlePrediction requests.
            miner_uids: List of miner UIDs.

        Returns:
            Tuple of (finished_responses, working_miner_uids).
        """
        # Use concurrent async operations based on subnet-22 patterns
        if not validator_prediction_requests:
            return [], miner_uids

        # Create concurrent tasks for all prediction requests
        tasks = []
        for candle_prediction_request in validator_prediction_requests:
            bittensor.logging.info(
                f"Sending to [magenta]{len(miner_uids)}[/magenta] miners prediction request for interval: [blue]{candle_prediction_request.interval}[/blue]"
            )
            task = send_predictions_to_miners(
                validator=self,
                input_synapse=GetCandlePrediction(candle_prediction=candle_prediction_request),
                batch_uids=miner_uids
            )
            tasks.append(task)

        # Execute all tasks concurrently and gather results
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and filter out exceptions
            all_finished_responses = []
            working_miner_uids = miner_uids  # Default to all UIDs

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    bittensor.logging.error(f"Error in prediction request {i}: {result}")
                    continue

                finished_responses, current_working_miner_uids = result
                all_finished_responses.extend(finished_responses)
                # Use the most recent working UIDs
                working_miner_uids = current_working_miner_uids

        except Exception as e:
            bittensor.logging.error(f"Error in concurrent prediction gathering: {e}")
            return [], miner_uids

        return all_finished_responses, working_miner_uids

    def save(
        self, finished_responses: list[dict], working_miner_uids: list[int], miner_uids: list[int], validator_prediction_requests: list[CandlePrediction]
    ):
        """
        Saves predictions and blacklists miners that did not respond or had invalid responses.

        Args:
            finished_responses: The responses from miners.
            working_miner_uids: List of UIDs that responded.
            miner_uids: List of all miner UIDs.
            validator_prediction_requests: List of CandlePrediction requests.
        """
        predictions = self.parse_responses(
            miner_predictions=finished_responses,
            prediction_requests=validator_prediction_requests,
        )
        self.storage.save_predictions(predictions)

        if not_working_miner_uids := [uid for uid in miner_uids if uid not in working_miner_uids]:
            bittensor.logging.debug(
                f"Miners {not_working_miner_uids} did not respond or had invalid responses."
            )



async def main():
    """Main async entry point for the validator."""
    validator = Validator()
    try:
        await validator.async_init()
        await validator.run()
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
