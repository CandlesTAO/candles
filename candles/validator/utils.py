import asyncio
import random
import traceback
from datetime import datetime, timezone

# Bittensor
import bittensor
from pydantic import BaseModel

# Local
from ..core.data import CandlePrediction
from ..core.synapse import GetCandlePrediction
from ..core.utils import get_next_timestamp_by_interval, is_miner


class ProcessedResponse(BaseModel):
    response: GetCandlePrediction
    uid: int


def get_miner_uids(metagraph: bittensor.metagraph, my_uid: int) -> list[int]:
    """Gets the uids of all miners in the metagraph."""
    return sorted(
        [
            uid.item()
            for uid in metagraph.uids
            if is_miner(uid.item(), metagraph) and uid.item() != my_uid
        ]
    )


def is_prediction_valid(prediction: CandlePrediction) -> bool:
    """Checks if a prediction is valid."""
    next_timestamp = get_next_timestamp_by_interval(prediction.interval)
    if prediction.prediction_id != next_timestamp:
        return False
    return prediction.color is not None


def process_single_response(response: GetCandlePrediction, uid: int) -> ProcessedResponse | None:
    if (
        response is None
        or response.candle_prediction is None
        or response.axon is None
        or response.axon.hotkey is None
        or not is_prediction_valid(response.candle_prediction)
    ):
        bittensor.logging.debug(
            f"UID {uid}: Miner failed to respond"
        )
        return None

    response.candle_prediction.miner_uid = uid
    response.candle_prediction.hotkey = response.axon.hotkey
    response.candle_prediction.prediction_date = datetime.now(timezone.utc)
    # TODO: Fix round() function type annotation issue - confidence field type needs verification
    # round the confidence to 4 decimal places

    response.candle_prediction.confidence = round(
        response.candle_prediction.confidence, 4
    ) # type: ignore
    return ProcessedResponse(response=response, uid=uid)


async def process_responses(
    batch_uids: list[int], validator, input_synapse: GetCandlePrediction
) -> tuple[list[CandlePrediction], list[int]]:

    responses = await validator.dendrite(
        axons=[validator.metagraph.axons[uid] for uid in batch_uids],
        synapse=input_synapse,
        deserialize=True,
        timeout=30,
    )

    working_miner_uids = []
    finished_responses = []

    for response, uid in zip(responses, batch_uids):
        if processed_response := process_single_response(response, uid):
            finished_responses.append(processed_response.response)
            working_miner_uids.append(processed_response.uid)

    return finished_responses, working_miner_uids


async def process_miner_requests(
    validator, batch_uids: list[int], input_synapse: GetCandlePrediction
) -> tuple[list[CandlePrediction], list[int]]:
    """
    Processes prediction requests to miners in batches with rate limiting.

    This function handles the distribution of prediction requests across multiple miners
    by breaking them into smaller batches and adding delays between batches to prevent
    overwhelming the network. It collects responses from all miners and tracks which
    miners are actively responding.

    Args:
        validator: The validator instance that manages the prediction requests
        batch_uids: List of miner UIDs to send prediction requests to
        input_synapse: The synapse containing the prediction request data

    Returns:
        tuple[list[CandlePrediction], list[int]]: A tuple containing:
            - List of completed candle predictions from responding miners
            - List of UIDs of miners that successfully responded

    The function implements a batching strategy where:
        - Miners are processed in groups based on validator.config.neuron.batch_size
        - Each batch is sent with a random delay (1-30 seconds) to spread network load
        - All responses are collected and aggregated before returning
    """
    # Initialize containers to accumulate responses from all batches
    # These will hold the final results from all miner interactions
    all_finished_responses = []
    all_working_miner_uids = []

    # Process miners in batches to avoid overwhelming the network
    # The batch size is configured in validator.config.neuron.batch_size
    for i in range(0, len(batch_uids), validator.config.neuron.batch_size):
        # Extract the current batch of miner UIDs
        # This creates a slice of UIDs from the current position to the next batch boundary
        batch = batch_uids[i : i + validator.config.neuron.batch_size]


        # Send prediction requests to the current batch of miners
        # This awaits responses from all miners in the batch simultaneously
        finished_responses, working_miner_uids = await process_responses(
            batch, validator, input_synapse
        )

        # Accumulate the responses from this batch into the overall results
        # This ensures we don't lose any responses when processing multiple batches
        all_finished_responses.extend(finished_responses)
        all_working_miner_uids.extend(working_miner_uids)

        # Add a random delay between batches to spread out network requests
        # This prevents overwhelming the network and reduces the chance of rate limiting
        # The delay ranges from 1-30 seconds to provide natural distribution
        await asyncio.sleep(random.uniform(1.0, 30.0))  # 1-30 seconds

    # Log completion status with summary of working miners
    # This provides visibility into the overall success rate of the prediction requests
    bittensor.logging.info(f"Finished sending prediction requests to miners, [orange]working miners[/orange]: [magenta]{len(all_working_miner_uids)}[/magenta]")

    # Return the aggregated results from all batches
    # This includes all successful predictions and the UIDs of responding miners
    return all_finished_responses, all_working_miner_uids


async def send_predictions_to_miners(
    validator, input_synapse: GetCandlePrediction, batch_uids: list[int]
) -> tuple[list[CandlePrediction], list[int]] | None:
    try:
        random.shuffle(batch_uids)

        all_finished_responses, all_working_miner_uids = await process_miner_requests(
            validator, batch_uids, input_synapse
        )

        if not all_working_miner_uids:
            bittensor.logging.info("No miner responses available.")
            return all_finished_responses, all_working_miner_uids

        bittensor.logging.info(
            f"Received responses from {len(all_working_miner_uids)} miners"
        )

        return all_finished_responses, all_working_miner_uids

    except Exception as e:
        bittensor.logging.error(
            f"Failed to send predictions to miners and store in validator database: {str(e)}",
            traceback.format_exc(),
        )
        return None
