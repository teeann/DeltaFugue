# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Trimmed vendored fork: the DeltaFugue / GatedDeltaFugue causal-LM wrappers, plus
# the DeltaNet, Gated DeltaNet and Transformer baselines used for pretraining.

from fla.models.delta_net import DeltaNetConfig, DeltaNetForCausalLM, DeltaNetModel
from fla.models.delta_fugue import DeltaFugueConfig, DeltaFugueForCausalLM, DeltaFugueModel
from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM, GatedDeltaNetModel
from fla.models.gated_deltafugue import GatedDeltaFugueConfig, GatedDeltaFugueForCausalLM, GatedDeltaFugueModel
from fla.models.transformer import TransformerConfig, TransformerForCausalLM, TransformerModel

__all__ = [
    'DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel',
    'DeltaFugueConfig', 'DeltaFugueForCausalLM', 'DeltaFugueModel',
    'GatedDeltaNetConfig', 'GatedDeltaNetForCausalLM', 'GatedDeltaNetModel',
    'GatedDeltaFugueConfig', 'GatedDeltaFugueForCausalLM', 'GatedDeltaFugueModel',
    'TransformerConfig', 'TransformerForCausalLM', 'TransformerModel',
]
