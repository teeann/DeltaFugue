# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Trimmed vendored fork: only DeltaFugue, GatedDeltaFugue and the baseline layers
# used by the experiments are kept (see models/fla.py in the repository root).
# Upstream fla ships many more.

from .attn import Attention
from .delta_net import DeltaNet
from .delta_fugue import DeltaFugue
from .gated_deltanet import GatedDeltaNet
from .gated_deltaproduct import GatedDeltaProduct
from .gated_deltafugue import GatedDeltaFugue
from .rwkv6 import RWKV6Attention
from .rwkv7 import RWKV7Attention

__all__ = [
    'Attention',
    'DeltaNet',
    'DeltaFugue',
    'GatedDeltaNet',
    'GatedDeltaProduct',
    'GatedDeltaFugue',
    'RWKV6Attention',
    'RWKV7Attention',
]
