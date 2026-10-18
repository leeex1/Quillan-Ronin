from dataclasses import dataclass

@dataclass
class RLConfig:
    """Configuration for RL training"""
    learning_rate: float = 3e-4
    batch_size: int = 4
    num_trajectories: int = 4  # G in the formula
    max_trajectory_len: int = 128
    clip_epsilon: float = 0.2  # ε in the formula
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    aux_loss_coef: float = 0.01
    diffusion_loss_coef: float = 0.1
    max_grad_norm: float = 1.0
    num_epochs: int = 1000
    warmup_steps: int = 1000
    use_mixed_precision: bool = True
