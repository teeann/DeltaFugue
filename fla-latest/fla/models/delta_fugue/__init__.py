# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.delta_fugue.configuration_delta_fugue import DeltaFugueConfig
from fla.models.delta_fugue.modeling_delta_fugue import DeltaFugueForCausalLM, DeltaFugueModel

AutoConfig.register(DeltaFugueConfig.model_type, DeltaFugueConfig, exist_ok=True)
AutoModel.register(DeltaFugueConfig, DeltaFugueModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaFugueConfig, DeltaFugueForCausalLM, exist_ok=True)

__all__ = ['DeltaFugueConfig', 'DeltaFugueForCausalLM', 'DeltaFugueModel']
