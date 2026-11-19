# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gated_deltafugue.configuration_gated_deltafugue import GatedDeltaFugueConfig
from fla.models.gated_deltafugue.modeling_gated_deltafugue import GatedDeltaFugueForCausalLM, GatedDeltaFugueModel

AutoConfig.register(GatedDeltaFugueConfig.model_type, GatedDeltaFugueConfig, exist_ok=True)
AutoModel.register(GatedDeltaFugueConfig, GatedDeltaFugueModel, exist_ok=True)
AutoModelForCausalLM.register(GatedDeltaFugueConfig, GatedDeltaFugueForCausalLM, exist_ok=True)

__all__ = ['GatedDeltaFugueConfig', 'GatedDeltaFugueForCausalLM', 'GatedDeltaFugueModel']
