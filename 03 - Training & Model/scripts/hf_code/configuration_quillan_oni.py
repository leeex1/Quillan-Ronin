"""Transformers configuration adapter for Quillan-Ronin v5.4 ONI."""

from dataclasses import fields
from typing import Any

from transformers import PretrainedConfig


class QuillanOniConfig(PretrainedConfig):
    """Serializable Transformers config backed by the native ONI dataclass."""

    model_type = "quillan_ronin_oni"

    def __init__(self, **kwargs: Any) -> None:
        aliases = {
            "hidden_size": "hidden_dim",
            "intermediate_size": "ffn_dim",
            "num_hidden_layers": "n_layer",
            "num_attention_heads": "n_head",
            "max_position_embeddings": "max_seq_len",
        }
        for source, target in aliases.items():
            if target not in kwargs and source in kwargs:
                kwargs[target] = kwargs[source]

        from quillan_v5_4_oni import QuillanOniConfig as CoreConfig

        core_defaults = CoreConfig()
        core_fields = {field.name for field in fields(CoreConfig)}
        core_values = {}
        for name in core_fields:
            core_values[name] = kwargs.pop(name, getattr(core_defaults, name))

        super().__init__(**kwargs)
        for name, value in core_values.items():
            setattr(self, name, value)

        self.hidden_size = self.hidden_dim
        self.intermediate_size = self.ffn_dim
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.max_position_embeddings = self.max_seq_len

    def to_core_config(self):
        """Convert to the native dataclass consumed by QuillanRoninOni."""
        from quillan_v5_4_oni import QuillanOniConfig as CoreConfig

        values = {
            field.name: getattr(self, field.name)
            for field in fields(CoreConfig)
        }
        return CoreConfig(**values)
