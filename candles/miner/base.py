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

import asyncio
import argparse
from traceback import print_exception

import bittensor
from ..core.neuron import BaseNeuron
from ..core.utils import add_miner_args


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            # bittensor.logging.warning(
            #     "You are allowing non-validators to send requests to your miner. This is a security risk."
            # )
            pass
        if self.config.blacklist.allow_non_registered:
            bittensor.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bittensor.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bittensor.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.get_candle_prediction,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bittensor.logging.info(f"Axon created: {self.axon}")

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

    async def async_init(self):
        """
        Async initialization for miner. Must be called after __init__ for AsyncSubtensor setup.
        """
        if not self.config.mock:
            # Initialize the base neuron async components
            await super().async_init()

            # Init sync with the network. Updates the metagraph.
            await self.sync_async()

    async def run(self):
        """
        Async main execution loop for the miner neuron.

        This method orchestrates the continuous operation of the miner by:
        1. Ensuring the miner is properly registered and synchronized with the network
        2. Starting the axon to serve incoming requests from validators
        3. Maintaining synchronization with the blockchain state
        4. Implementing proper timing controls to maintain consistent operation intervals

        The method operates in a continuous loop until explicitly stopped, handling
        various exit conditions and error scenarios gracefully.
        """

        # Ensure the miner is properly registered and synchronized with the network
        if self.config.mock:
            self.sync()
        else:
            await self.sync_async()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bittensor.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        # Handle axon serving differently for mock vs non-mock mode
        if self.config.mock:
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        else:
            # For AsyncSubtensor, we need to await the serve_axon call
            await self.subtensor.serve_axon(
                netuid=self.config.netuid,
                axon=self.axon,
            )

        # Start the miner's axon, making it active on the network.
        self.axon.start()

        # Log the starting block number for monitoring and debugging purposes
        if self.config.mock:
            current_block = self.block
        else:
            current_block = await self.get_current_block()
        bittensor.logging.info(f"Miner starting at block: {current_block}")

        # Main execution loop that continues until the miner is intentionally stopped
        try:
            while not self.should_exit:
                # Check if we need to sync with the network
                if self.config.mock:
                    current_block = self.block
                    should_sync = (
                        current_block - self.metagraph.last_update[self.uid]
                        >= self.config.neuron.epoch_length
                    )
                else:
                    current_block = await self.get_current_block()
                    should_sync = (
                        current_block - self.metagraph.last_update[self.uid]
                        >= self.config.neuron.epoch_length
                    )

                if should_sync:
                    # Sync metagraph and potentially set weights.
                    if self.config.mock:
                        self.sync()
                    else:
                        await self.sync_async()
                    self.step += 1

                # Wait before checking again
                await asyncio.sleep(1)

        # Handle intentional interruption (e.g., Ctrl+C) with graceful shutdown
        except KeyboardInterrupt:
            self.axon.stop()
            # Cleanup async resources
            if not self.config.mock:
                await self.cleanup()
            bittensor.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # Handle unexpected errors while maintaining miner operation
        except Exception as err:
            bittensor.logging.error("Error during mining", str(err))
            bittensor.logging.debug(print_exception(type(err), err, err.__traceback__))

    async def run_in_background_task(self):
        """
        Starts the miner's operations in a background asyncio task.
        This method facilitates async operation management.
        """
        if not self.is_running:
            bittensor.logging.debug("Starting miner in background task.")
            self.should_exit = False
            self.background_task = asyncio.create_task(self.run())
            self.is_running = True
            bittensor.logging.debug("Started")

    async def stop_run_task(self):
        """
        Stops the miner's operations that are running in the background task.
        """
        if self.is_running:
            bittensor.logging.debug("Stopping miner in background task.")
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
        """
        Async context manager entry for starting miner's background operations.
        This method facilitates the use of the miner in an 'async with' statement.
        """
        await self.run_in_background_task()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        Async context manager exit for stopping miner's background operations.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        await self.stop_run_task()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

    async def resync_metagraph_async(self):
        """Async version of resync_metagraph."""

        # Sync the metagraph asynchronously.
        async with self.lock:
            await self.metagraph.sync(subtensor=self.subtensor)
