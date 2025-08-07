# The MIT License (MIT)
# Copyright © 2024 sportstensor

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
from typing import Optional
# Third Party
import pydantic
# Bittensor
import bittensor
# Local
from .data import CandlePrediction



class BaseSynapse(bittensor.Synapse):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    version: Optional[int] = pydantic.Field(
        description="Protocol version", default=None
    )

class GetCandlePrediction(BaseSynapse):
    """
    Protocol by which Validators can retrieve a Candle Prediction from a Miner.

    Attributes:
    - candle_prediction: A single CandlePrediction object that the Miner can serve.
    """

    candle_prediction: CandlePrediction = pydantic.Field(
        description="The CandlePrediction object being requested",
        frozen=False,
        repr=False,
        default=None,
    )

    def __str__(self):
        return f"GetCandlePrediction(candle_prediction={self.candle_prediction}, axon={self.axon})"

    __repr__ = __str__
