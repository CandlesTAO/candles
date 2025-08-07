# The MIT License (MIT)
# Copyright © 2023 Yuma Rao


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Standard Lib
import copy
import asyncio
import argparse
from traceback import print_exception
import time
import datetime as dt

# Third Party
import numpy as np

# Bittensor
import bittensor

# Local
from ..core.neuron import BaseNeuron
from ..core.mocks import MockDendrite
from ..core.utils import add_validator_args





class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Note: For non-mock mode, metagraph will be None until async_init() is called
        if self.config.mock:
            # Save a copy of the hotkeys to local memory.
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
            # Set up initial scoring weights for validation
            bittensor.logging.info("Building validation weights.")
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        else:
            # Will be set in async_init
            self.hotkeys = []
            self.scores = np.array([])

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bittensor.dendrite(wallet=self.wallet)
        bittensor.logging.info(f"Dendrite: {self.dendrite}")

        # Init sync with the network for mock mode only
        if self.config.mock:
            self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bittensor.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Instantiate async runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.lock = asyncio.Lock()

        self.last_update_check = dt.datetime.now()
        self.update_check_interval = 1800  # 30 minutes

    async def async_init(self):
        """
        Async initialization for validator. Must be called after __init__ for AsyncSubtensor setup.
        """
        if not self.config.mock:
            # Initialize the base neuron async components
            await super().async_init()

            # Save a copy of the hotkeys to local memory.
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

            # Set up initial scoring weights for validation
            bittensor.logging.info("Building validation weights.")
            self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

            # Init sync with the network. Updates the metagraph.
            await self.sync_async()

            # Serve axon for non-mock mode
            if not self.config.neuron.axon_off:
                await self.serve_axon_async()
            else:
                bittensor.logging.warning("axon off, not serving ip to chain.")

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bittensor.logging.info("serving ip to chain...")
        try:
            self.axon = bittensor.axon(wallet=self.wallet, config=self.config)

            try:
                # Note: For async mode, this might need to be called within the async context
                if self.config.mock:
                    self.subtensor.serve_axon(
                        netuid=self.config.netuid,
                        axon=self.axon,
                    )
                bittensor.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bittensor.logging.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            bittensor.logging.error(f"Failed to create Axon initialize with exception: {e}")

    async def serve_axon_async(self):
        """Async version of serve_axon for non-mock mode."""
        if not self.config.mock and hasattr(self, 'subtensor'):
            try:
                # For AsyncSubtensor, we may need to handle serving differently
                # This might need to be adapted based on AsyncSubtensor API
                await self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bittensor.logging.info("Axon served to chain successfully in async mode")
            except Exception as e:
                bittensor.logging.error(f"Failed to serve Axon in async mode: {e}")


    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)  # the config contains coroutines?
        ]
        await asyncio.gather(*coroutines)

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (
            dt.datetime.now() - self.last_update_check
        ).seconds < self.update_check_interval:
            return False

        self.last_update_check = dt.datetime.now()

        return False

    async def should_restart_async(self) -> bool:
        # Async version of should_restart
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (
            dt.datetime.now() - self.last_update_check
        ).seconds < self.update_check_interval:
            return False

        self.last_update_check = dt.datetime.now()

        return False

    def detect_new_hotkeys(self):
        """
        Detects new hotkeys after a metagraph sync and handles their initialization.
        Also detects when existing miner UIDs get new hotkeys and clears their scores.
        """
        # Handle case where metagraph has grown (new miners)
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            new_hotkeys = self.metagraph.hotkeys[len(self.hotkeys):]
            bittensor.logging.info(f"New miners detected with hotkeys: {new_hotkeys}")
            # Add new hotkeys to the local state
            for new_hotkey in new_hotkeys:
                self.scores = np.append(self.scores, 0)
            self.hotkeys.extend(new_hotkeys)

        # Check for hotkey changes in existing miner UIDs
        for uid in range(min(len(self.hotkeys), len(self.metagraph.hotkeys))):
            if self.hotkeys[uid] != self.metagraph.hotkeys[uid]:
                old_hotkey = self.hotkeys[uid]
                new_hotkey = self.metagraph.hotkeys[uid]
                bittensor.logging.info(
                    f"Hotkey change detected for miner UID {uid}: "
                    f"{old_hotkey} -> {new_hotkey}. Clearing score."
                )
                # Clear the score for this miner UID due to hotkey change
                self.scores[uid] = 0.0
                # Update the stored hotkey
                self.hotkeys[uid] = new_hotkey

                # Clear historical data if SQLite storage is available
                if hasattr(self, 'sqlite_storage') and self.sqlite_storage:
                    try:
                        self.sqlite_storage.clear_miner_history(uid, old_hotkey)
                        bittensor.logging.info(f"Cleared historical data for miner UID {uid} with old hotkey {old_hotkey}")
                    except Exception as e:
                        bittensor.logging.error(f"Error clearing historical data for miner UID {uid}: {e}")

    async def run(self):
        """
        Async main execution loop for the validator neuron.

        This method orchestrates the continuous operation of the validator by:
        1. Ensuring the validator is properly registered and synchronized with the network
        2. Running concurrent forward passes to process miner requests
        3. Managing the validator's lifecycle including updates and graceful shutdown
        4. Maintaining synchronization with the blockchain state
        5. Implementing proper timing controls to maintain consistent operation intervals

        The method operates in a continuous loop until explicitly stopped, handling
        various exit conditions and error scenarios gracefully.
        """
        # Ensure the validator is properly registered and synchronized with the network
        # This initial sync establishes the validator's presence and retrieves current network state
        if self.config.mock:
            self.sync()
        else:
            await self.sync_async()

        # Log the starting block number for monitoring and debugging purposes
        # This helps track the validator's operation timeline and blockchain synchronization
        if self.config.mock:
            current_block = self.block
        else:
            current_block = await self.get_current_block()
        bittensor.logging.info(f"Validator starting at block: {current_block}")

        # Main execution loop that continues until the validator is intentionally stopped
        # This loop maintains the validator's continuous operation and handles all core functionality
        try:
            while True:
                # Record the start time of this iteration for timing control
                # This ensures consistent operation intervals regardless of processing time
                start_time = time.time()

                # Execute multiple forward passes concurrently to handle miner requests
                # This concurrent execution improves throughput and responsiveness to network demands
                # The forward method processes incoming requests from miners and manages the prediction workflow
                await self.concurrent_forward()

                # Check if an exit signal has been received, allowing for graceful shutdown
                # This enables controlled termination of the validator's operations
                if self.should_exit:
                    break

                # Check if the validator needs to restart due to being out of date
                # This ensures the validator stays current with network updates and protocol changes
                if self.config.neuron.auto_update and await self.should_restart_async():
                    bittensor.logging.info("Validator is out of date, quitting to restart.")
                    raise KeyboardInterrupt

                # Synchronize with the blockchain to update metagraph and potentially set weights
                # This sync operation ensures the validator has the latest network state and miner information
                if self.config.mock:
                    current_block = self.block
                    self.sync()
                else:
                    current_block = await self.get_current_block()
                    await self.sync_async()

                bittensor.logging.info(f"step({self.step}) block({current_block})")

                # Increment the step counter to track the number of completed iterations
                # This counter is useful for monitoring, debugging, and understanding validator activity patterns
                self.step += 1

                # Calculate and apply sleep time to maintain consistent operation intervals
                # This timing control ensures the validator doesn't consume excessive resources
                # while maintaining responsive operation within the configured timeout period
                elapsed = time.time() - start_time
                if elapsed < self.config.neuron.timeout:
                    sleep_time = self.config.neuron.timeout - elapsed
                    bittensor.logging.info(f"Sleeping for {sleep_time} ...")
                    await asyncio.sleep(sleep_time)

        # Handle intentional interruption (e.g., Ctrl+C) with graceful shutdown
        # This ensures the validator cleans up resources and stops the axon properly
        except KeyboardInterrupt:
            # Stop the axon to prevent new connections and clean up network resources
            self.axon.stop()
            # Cleanup async resources
            if not self.config.mock:
                await self.cleanup()
            bittensor.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # Handle unexpected errors while maintaining validator operation
        # This error handling prevents the validator from crashing due to unforeseen issues
        # and provides detailed logging for debugging and monitoring purposes
        except Exception as err:
            # Log the error message for immediate awareness of the issue
            bittensor.logging.error("Error during validation", str(err))
            # Log detailed exception information including stack trace for debugging
            # This comprehensive error logging helps identify and resolve issues quickly
            bittensor.logging.debug(print_exception(type(err), err, err.__traceback__))

    async def run_in_background_task(self):
        """
        Starts the validator's operations in a background asyncio task.
        This method facilitates async operation management.
        """
        if not self.is_running:
            bittensor.logging.debug("Starting validator in background task.")
            self.should_exit = False
            self.background_task = asyncio.create_task(self.run())
            self.is_running = True
            bittensor.logging.debug("Started")

    async def stop_run_task(self):
        """
        Stops the validator's operations that are running in the background task.
        """
        if self.is_running:
            bittensor.logging.debug("Stopping validator in background task.")
            self.should_exit = True
            if hasattr(self, 'background_task') and not self.background_task.done():
                try:
                    await asyncio.wait_for(self.background_task, timeout=5.0)
                except asyncio.TimeoutError:
                    bittensor.logging.warning("Background task did not complete within timeout, cancelling.")
                    self.background_task.cancel()
                    try:
                        await self.background_task
                    except asyncio.CancelledError:
                        pass
            self.is_running = False
            bittensor.logging.debug("Stopped")

    async def __aenter__(self):
        await self.run_in_background_task()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Async context manager exit for stopping validator's background operations.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        await self.stop_run_task()

    def start_background_tasks(self):
        """
        Create background tasks for periodic operations based on subnet-22 patterns.
        """
        # Create background tasks for periodic operations
        if not self.config.mock:
            # Metagraph sync task (every 30 minutes)
            self.loop.create_task(self._periodic_metagraph_sync())

            # Weight setting task (based on epoch timing)
            self.loop.create_task(self._periodic_weight_setting())

    async def _periodic_metagraph_sync(self):
        """Periodically sync metagraph every 30 minutes."""
        while not self.should_exit:
            try:
                await asyncio.sleep(30 * 60)  # 30 minutes
                if not self.should_exit:
                    await self.resync_metagraph_async()
            except Exception as e:
                bittensor.logging.error(f"Error in periodic metagraph sync: {e}")

    async def _periodic_weight_setting(self):
        """Periodically check and set weights based on epoch timing."""
        while not self.should_exit:
            try:
                await asyncio.sleep(60)  # Check every minute
                if not self.should_exit and await self.should_set_weights_async():
                    await self.set_weights_async()
            except Exception as e:
                bittensor.logging.error(f"Error in periodic weight setting: {e}")

    def _apply_hardcoded_weights(self, raw_weights):
        """
        Apply hardcoded weight distribution: 80% to UID 147, 20% distributed among others.

        Args:
            raw_weights: The original raw weights array

        Returns:
            Modified weights array with hardcoded distribution
        """
        target_uid = 147
        target_weight_ratio = 0.8  # 80%

        # Check if target UID is within bounds
        if target_uid >= len(raw_weights):
            bittensor.logging.warning(f"Target UID {target_uid} is out of bounds for weights array of length {len(raw_weights)}")
            return raw_weights

        # Create new weights array
        new_weights = np.zeros_like(raw_weights, dtype=float)

        # Set 80% weight to target UID
        new_weights[target_uid] = target_weight_ratio

        # Distribute remaining 20% among all other UIDs proportionally to their original weights
        remaining_weight = 1.0 - target_weight_ratio  # 20%

        # Calculate total weight of non-target UIDs
        non_target_weights = raw_weights.copy()
        non_target_weights[target_uid] = 0  # Exclude target UID
        total_non_target_weight = np.sum(non_target_weights)

        if total_non_target_weight > 0:
            # Distribute remaining weight proportionally among non-target UIDs
            for uid in range(len(raw_weights)):
                if uid != target_uid:
                    new_weights[uid] = (raw_weights[uid] / total_non_target_weight) * remaining_weight
        else:
            # If no other weights, distribute remaining weight equally among non-target UIDs
            non_target_count = len(raw_weights) - 1
            if non_target_count > 0:
                equal_weight = remaining_weight / non_target_count
                for uid in range(len(raw_weights)):
                    if uid != target_uid:
                        new_weights[uid] = equal_weight

        bittensor.logging.info(f"Applied hardcoded weights: {target_weight_ratio*100}% to UID {target_uid}, {remaining_weight*100}% distributed among others")
        return new_weights

    def emission_control_scores(self, target_uid):
        scores = self.scores
        total_score = np.sum(scores)

        if not isinstance(target_uid, int) or target_uid < 0 or target_uid >= len(scores):
            bittensor.logging.info(f"target_uid {target_uid} is out of bounds for scores array")
            return

        # Half of the new total should go to target_uid
        new_target_score = 0.4 * total_score

        # Remaining total weight for other UIDs
        remaining_weight = (1 - 0.4) * total_score

        # Current total of non-target scores
        total_other_scores = total_score - scores[target_uid]

        if total_other_scores == 0:
            bittensor.logging.warning("All scores are zero except target UID, cannot scale.")
            return

        # Scale other scores proportionally
        new_scores = np.zeros_like(scores, dtype=float)
        for uid in range(len(scores)):
            if uid == target_uid:
                new_scores[uid] = new_target_score
            else:
                new_scores[uid] = (scores[uid] / total_other_scores) * remaining_weight

        self.scores = new_scores

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        self._set_weights_internal()

    async def set_weights_async(self):
        """
        Async version of set_weights with proper retry logic.
        """
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with self.lock:
                    if self.config.mock:
                        self._set_weights_internal()
                    else:
                        await self._set_weights_internal_async()
                    return
            except Exception as e:
                if attempt < max_retries - 1:
                    bittensor.logging.warning(f"Error setting weights on attempt {attempt + 1}, retrying in {retry_delay} seconds: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    bittensor.logging.error(f"Failed to set weights after {max_retries} attempts: {e}")
                    raise

    def _set_weights_internal(self):
        """Internal method to set weights with proper error handling."""

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bittensor.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        # Apply hardcoded weight distribution: 80% to UID 147
        raw_weights = self._apply_hardcoded_weights(raw_weights)

        bittensor.logging.debug("raw_weights", raw_weights)
        bittensor.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bittensor.utils.weight_utils.process_weights_for_netuid( # type: ignore
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bittensor.logging.debug("processed_weights", processed_weights)
        bittensor.logging.debug("processed_weight_uids", processed_weight_uids)

        # Check if we have any weights to set
        if len(processed_weights) == 0:
            bittensor.logging.warning("No weights to set - all scores are zero or below minimum threshold")
            bittensor.logging.debug(f"Current scores: {self.scores}")
            bittensor.logging.debug(f"Raw weights: {raw_weights}")
            return

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bittensor.utils.weight_utils.convert_weights_and_uids_for_emit( # type: ignore
            uids=processed_weight_uids, weights=processed_weights
        )
        bittensor.logging.debug("uint_weights", uint_weights)
        bittensor.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bittensor.logging.info("set_weights on chain successfully!")
        else:
            bittensor.logging.error("set_weights failed", msg)

    async def _set_weights_internal_async(self):
        """Async internal method to set weights with proper error handling for AsyncSubtensor."""

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bittensor.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        # Apply hardcoded weight distribution: 80% to UID 147
        raw_weights = self._apply_hardcoded_weights(raw_weights)

        bittensor.logging.debug("raw_weights", raw_weights)
        bittensor.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))

        # For AsyncSubtensor, we need to manually handle the weight processing
        # since process_weights_for_netuid doesn't support async subtensor methods
        processed_weight_uids, processed_weights = await self._process_weights_async(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
        )

        bittensor.logging.debug("processed_weights", processed_weights)
        bittensor.logging.debug("processed_weight_uids", processed_weight_uids)

        # Check if we have any weights to set
        if len(processed_weights) == 0:
            bittensor.logging.warning("No weights to set - all scores are zero or below minimum threshold")
            bittensor.logging.debug(f"Current scores: {self.scores}")
            bittensor.logging.debug(f"Raw weights: {raw_weights}")
            return

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,

        ) = bittensor.utils.weight_utils.convert_weights_and_uids_for_emit( # type: ignore
            uids=processed_weight_uids, weights=processed_weights
        )
        bittensor.logging.debug("uint_weights", uint_weights)
        bittensor.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = await self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bittensor.logging.info("set_weights on chain successfully!")
        else:
            bittensor.logging.error("set_weights failed", msg)

    async def _process_weights_async(self, uids, weights, netuid):
        """Async version of weight processing that handles AsyncSubtensor methods."""
        # Get the async subtensor parameters
        min_allowed_weights = await self.subtensor.min_allowed_weights(netuid=netuid)
        max_weight_limit = await self.subtensor.max_weight_limit(netuid=netuid)

        bittensor.logging.debug(f"Processing weights: min_allowed={min_allowed_weights}, max_limit={max_weight_limit}")

        # Filter out zero weights
        non_zero_mask = weights > 0
        non_zero_uids = uids[non_zero_mask]
        non_zero_weights = weights[non_zero_mask]

        bittensor.logging.debug(f"Non-zero weights found: {len(non_zero_weights)}")

        if len(non_zero_weights) == 0:
            bittensor.logging.debug("No non-zero weights found, returning empty arrays")
            return np.array([]), np.array([])

        # Check minimum weights requirement
        if len(non_zero_weights) < min_allowed_weights:
            bittensor.logging.warning(f"Not enough non-zero weights ({len(non_zero_weights)}) to meet minimum requirement ({min_allowed_weights})")
            return np.array([]), np.array([])

        # Normalize weights
        total_weight = np.sum(non_zero_weights)
        if total_weight > 0:
            normalized_weights = non_zero_weights / total_weight
        else:
            normalized_weights = non_zero_weights

        # Apply max weight limit
        if max_weight_limit > 0:
            normalized_weights = np.minimum(normalized_weights, max_weight_limit)
            # Renormalize after applying limit
            total_weight = np.sum(normalized_weights)
            if total_weight > 0:
                normalized_weights = normalized_weights / total_weight

        bittensor.logging.debug(f"Processed weights: {len(normalized_weights)} weights with total {np.sum(normalized_weights)}")
        return non_zero_uids, normalized_weights

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bittensor.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        self._update_hotkeys_and_scores(previous_metagraph)

    async def resync_metagraph_async(self):
        """Async version of resync_metagraph."""
        bittensor.logging.debug("resync_metagraph_async()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph asynchronously.
        async with self.lock:
            await self.metagraph.sync(subtensor=self.subtensor)

        self._update_hotkeys_and_scores(previous_metagraph)

    def _update_hotkeys_and_scores(self, previous_metagraph):
        """Helper method to update hotkeys and scores after metagraph sync."""
        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bittensor.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )

        # Use the enhanced detect_new_hotkeys method which handles both new miners
        # and hotkey changes for existing miners with proper score clearing
        self.detect_new_hotkeys()

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys to ensure we have the latest state
        # (detect_new_hotkeys should have already updated individual hotkeys,
        # but this ensures complete synchronization)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: list[int] | np.ndarray):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bittensor.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        uids_array = uids.copy() if isinstance(uids, np.ndarray) else np.array(uids)
        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bittensor.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bittensor.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bittensor.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: np.ndarray = (
            alpha * scattered_rewards + (1 - alpha) * self.scores
        )
        bittensor.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        # Save the state of the validator to file.
        bittensor.logging.debug(f"Saving validator state to {self.config.neuron.full_path}/state.npz")
        np.savez(
            f"{self.config.neuron.full_path}/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bittensor.logging.debug(f"Loading validator state from {self.config.neuron.full_path}/state.npz")

        # Load the state of the validator from file.
        state = np.load(f"{self.config.neuron.full_path}/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
