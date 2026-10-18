# ==============================================================================
# 👑 QUILLAN-RONIN v6.0.3 QUANTUM - CONFIGURATION BRIDGE
# Architect: CrashOverrideX | Identity: C19-VIGIL
# ==============================================================================

from transformers import PretrainedConfig

class QuillanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `QuillanRoninForCausalLM`.
    It registers the custom parameters for the 34-Expert Council, EGGROLL, and Thermodynamics.
    """
    model_type = "quillan_bitnet_moe"

    def __init__(
        self,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=20,
        vocab_size=50257,
        bits=1.58,
        num_experts=34,
        num_experts_per_tok=2,
        moe_routing_algo="top_k",
        moe_layers=None,
        quantization_config=None,
        swarm_config=None,
        eggroll_config=None,
        thermodynamics=None,
        orchestration=None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.bits = bits
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_routing_algo = moe_routing_algo
        
        # Default sparse MoE layers if not explicitly passed
        self.moe_layers = moe_layers if moe_layers is not None else [4, 8, 12, 16, 20, 24, 28, 32]
        
        self.quantization_config = quantization_config or {"quant_method": "bitnet", "bits": 1.58}
        
        # Sovereign Custom Protocols (With safe fallbacks)
        self.swarm_config = swarm_config or {
            "num_micro_subagents": 100000, 
            "swarm_embedding_dim": 128, 
            "swarm_precision": "int8", 
            "agent_pooling": True
        }
        self.eggroll_config = eggroll_config or {"rank_r": 16, "active": True}
        self.thermodynamics = thermodynamics or {"governor": "Lee-Mach-6", "e_ice_limit_ms": 100}
        self.orchestration = orchestration or {"coordination_layer": "C31-NEXUS"}
        
        # Initialize the base HF PretrainedConfig
        super().__init__(**kwargs)
