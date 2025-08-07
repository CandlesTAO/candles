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

import copy
from typing import Protocol

import bittensor as bt
from bittensor_wallet.mock.wallet_mock import MockWallet
from .mocks import MockMetagraph, MockSubtensor
from .utils import check_config, add_args, config, ttl_get_block



class BaseNeuron(Protocol):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.AsyncSubtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = 1

    @property
    def block(self):
        return ttl_get_block(self)

    async def get_current_block(self):
        """
        Async method to get current block for AsyncSubtensor.
        """
        if self.config.mock:
            return ttl_get_block(self)
        else:
            try:
                return await self.subtensor.get_current_block()
            except Exception as e:
                bt.logging.error(f"Error getting current block: {e}, reinitializing subtensor...")
                await self.subtensor.close()
                self.subtensor = bt.AsyncSubtensor(config=self.config)
                await self.subtensor.initialize()
                self.metagraph = await self.subtensor.metagraph(self.config.netuid)
                return await self.subtensor.get_current_block()

    def __init__(self, config=None):
        # print("self.config()", self.config())
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        # bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor wallet, subtensor, and metagraph.")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            print("Mocking wallet, subtensor, and metagraph")
            self.wallet = MockWallet(config=self.config)
            self.subtensor = MockSubtensor(self.config.netuid, wallet=self.wallet)
            self.metagraph = MockMetagraph(self.config.netuid, subtensor=self.subtensor)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.AsyncSubtensor(config=self.config)
            self.metagraph = None  # Will be set in async_init

        bt.logging.debug(f"Wallet: {self.wallet}")
        bt.logging.debug(f"Subtensor: {self.subtensor}")

        # Note: metagraph and registration check will be done in async_init for non-mock mode
        if self.config.mock:
            bt.logging.debug(f"Metagraph: {self.metagraph}")
            # Check if the miner is registered on the Bittensor network before proceeding further.
            self.check_registered()
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(
                f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
            )

        self.step = 0

    async def async_init(self):
        """
        Async initialization for non-mock mode. Must be called after __init__ for AsyncSubtensor setup.
        """
        if not self.config.mock:
            # Initialize the async subtensor
            await self.subtensor.initialize()

            # Get metagraph asynchronously
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)

            bt.logging.debug(f"Metagraph: {self.metagraph}")

            # Check if the miner is registered on the Bittensor network before proceeding further.
            await self.check_registered_async()

            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(
                f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
            )

    #@abstractmethod
    #async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...


    def run(self): ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        # Turning off setting weights here as we'll run its logic in a background thread defined in neurons/validator.py
        #if self.should_set_weights():
            #self.set_weights()

        # Always save state.
        self.save_state()

    async def sync_async(self):
        """
        Async wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        await self.check_registered_async()

        if await self.should_sync_metagraph_async():
            await self.resync_metagraph_async()

        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    async def check_registered_async(self):
        # --- Check for registration (async version).
        if not await self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    async def should_sync_metagraph_async(self):
        """
        Async version: Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        current_block = await self.get_current_block()
        return (
            current_block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"  # don't set weights if you're a miner

    async def should_set_weights_async(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        current_block = await self.get_current_block()
        return (
            current_block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"  # don't set weights if you're a miner

    def save_state(self):
        bt.logging.warning(
            "save_state() not implemented for this neuron. You can implement this function to save model checkpoints or other useful data."
        )

    def load_state(self):
        bt.logging.warning(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )

    async def cleanup(self):
        """
        Cleanup method for proper resource management of AsyncSubtensor.
        Should be called when shutting down the neuron.
        """
        if not self.config.mock and hasattr(self, 'subtensor') and self.subtensor:
            try:
                await self.subtensor.close()
                bt.logging.info("AsyncSubtensor connection closed successfully.")
            except Exception as e:
                bt.logging.error(f"Error closing AsyncSubtensor: {e}")
