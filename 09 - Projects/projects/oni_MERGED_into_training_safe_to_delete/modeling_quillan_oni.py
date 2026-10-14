"""Transformers adapter for the native Quillan-Ronin ONI model."""

from typing import Any, Optional

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_quillan_oni import QuillanOniConfig
    from .quillan_v5_4_oni import QuillanRoninOni as NativeQuillanRoninOni
except ImportError:
    from configuration_quillan_oni import QuillanOniConfig
    from quillan_v5_4_oni import QuillanRoninOni as NativeQuillanRoninOni


class QuillanOniForCausalLM(PreTrainedModel):
    """HF-compatible wrapper that preserves native ONI forward semantics."""

    config_class = QuillanOniConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["UnrolledTransformerBlock"]

    def __init__(self, config: QuillanOniConfig) -> None:
        super().__init__(config)
        self.model = NativeQuillanRoninOni(config.to_core_config())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del attention_mask
        native_result = self.model(
            input_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=bool(use_cache),
            **kwargs,
        )

        loss = None
        aux_loss = None
        if isinstance(native_result, tuple):
            logits = native_result[0]
            if labels is not None and len(native_result) >= 2:
                loss = native_result[1]
            if labels is not None and len(native_result) >= 3:
                aux_loss = native_result[2]
            past = native_result[1] if use_cache and labels is None and len(native_result) >= 2 else None
        else:
            logits = native_result
            past = None

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past,
        )
        if aux_loss is not None:
            output.aux_loss = aux_loss
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, value) -> None:
        self.model.wte = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def tie_weights(self) -> None:
        self.model.lm_head.weight = self.model.wte.weight
