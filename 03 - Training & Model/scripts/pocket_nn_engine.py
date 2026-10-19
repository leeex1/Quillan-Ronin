"""
POCKETNN ENGINE: INTEGER-ONLY TRAINING AND INFERENCE VIA DIRECT FEEDBACK ALIGNMENT
==================================================================================
Paper: PocketNN: Integer-only Training and Inference of Neural Networks via Direct
       Feedback Alignment and Pocket Activations in Pure C++ (Song & Lin, HKUST, 2022)
ArXiv: 2201.02863

Core Formulations Implemented:
1. Direct Feedback Alignment (DFA) Formulation (Section 3.2, Eq. 10):
   delta_DFA^[k] = ((y_hat - y) * R^[k]) (*) actv'^[k](h^[k])
   Fixed random feedback matrix R^[k] eliminates exponential BP error growth O(2^(e + k*w)),
   guaranteeing constant bounded delta size O(2^(e + r)) independent of network depth.
2. 8-Bit Pocket Activation Family (Section 3.3, Table 1 & Figure 2):
   - PocketSigmoid: 6-segment piecewise linear approximation of 128*Sigmoid(x/32) bounded in [1, 127].
   - PocketTanh: 6-segment piecewise linear approximation of 128*Tanh(x/64) bounded in [-127, 127].
   - PocketReLU8bit: Truncated ReLU bounded in [0, 127].
3. Integer Fractional Arithmetic (Section 3.4):
   Implements fractional slopes (1/8, 1/4, 1/2) and learning rates via integer division.
4. Integer-Only Fully Connected Network (Section 3.5 & 4):
   Supports end-to-end integer forward inference and DFA backward parameter updates.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. POCKET ACTIVATION FUNCTIONS & DERIVATIVES (TABLE 1)
# ============================================================================

def pocket_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    PocketSigmoid (8-bit) approximating 128 * Sigmoid(x / 32) in range [1, 127].
    Table 1:
      1                       (x <= -128)
      x/8 + 20                (-128 < x <= -75)
      x/2 + 48                (-75 < x <= -32)
      x + 64                  (-32 < x <= 31)
      x/2 + 80                (31 < x <= 74)
      x/8 + 108               (74 < x <= 127)
      127                     (127 < x)
    Uses integer division (truncating towards zero / integer floor).
    """
    is_int = not x.is_floating_point()
    if is_int:
        x_div8 = torch.div(x, 8, rounding_mode='trunc')
        x_div2 = torch.div(x, 2, rounding_mode='trunc')
        
        out = torch.full_like(x, 1)
        m1 = (x > -128) & (x <= -75)
        out = torch.where(m1, x_div8 + 20, out)
        m2 = (x > -75) & (x <= -32)
        out = torch.where(m2, x_div2 + 48, out)
        m3 = (x > -32) & (x <= 31)
        out = torch.where(m3, x + 64, out)
        m4 = (x > 31) & (x <= 74)
        out = torch.where(m4, x_div2 + 80, out)
        m5 = (x > 74) & (x <= 127)
        out = torch.where(m5, x_div8 + 108, out)
        m6 = (x > 127)
        out = torch.where(m6, torch.full_like(x, 127), out)
        return out
    else:
        out = torch.full_like(x, 1.0)
        m1 = (x > -128.0) & (x <= -75.0)
        out = torch.where(m1, x / 8.0 + 20.0, out)
        m2 = (x > -75.0) & (x <= -32.0)
        out = torch.where(m2, x / 2.0 + 48.0, out)
        m3 = (x > -32.0) & (x <= 31.0)
        out = torch.where(m3, x + 64.0, out)
        m4 = (x > 31.0) & (x <= 74.0)
        out = torch.where(m4, x / 2.0 + 80.0, out)
        m5 = (x > 74.0) & (x <= 127.0)
        out = torch.where(m5, x / 8.0 + 108.0, out)
        m6 = (x > 127.0)
        out = torch.where(m6, torch.full_like(x, 127.0), out)
        return out


def pocket_sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of PocketSigmoid:
      0       (x <= -128 or x > 127)
      1/8     (-128 < x <= -75 or 74 < x <= 127)
      1/2     (-75 < x <= -32 or 31 < x <= 74)
      1       (-32 < x <= 31)
    """
    if not x.is_floating_point():
        # Represent derivative as scaled integer factor or float
        x_f = x.float()
    else:
        x_f = x

    deriv = torch.zeros_like(x_f)
    m_1_8 = ((x_f > -128.0) & (x_f <= -75.0)) | ((x_f > 74.0) & (x_f <= 127.0))
    deriv = torch.where(m_1_8, torch.full_like(x_f, 0.125), deriv)
    m_1_2 = ((x_f > -75.0) & (x_f <= -32.0)) | ((x_f > 31.0) & (x_f <= 74.0))
    deriv = torch.where(m_1_2, torch.full_like(x_f, 0.5), deriv)
    m_1 = (x_f > -32.0) & (x_f <= 31.0)
    deriv = torch.where(m_1, torch.full_like(x_f, 1.0), deriv)
    return deriv


def pocket_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    PocketTanh (8-bit) approximating 128 * Tanh(x / 64) in range [-127, 127].
    Table 1:
      -127                    (x <= -128)
      x/4 - 88                (-128 < x <= -75)
      x - 32                  (-75 < x <= -32)
      2x                      (-32 < x <= 31)
      x + 32                  (31 < x <= 74)
      x/4 + 88                (74 < x <= 127)
      127                     (127 < x)
    """
    is_int = not x.is_floating_point()
    if is_int:
        x_div4 = torch.div(x, 4, rounding_mode='trunc')
        out = torch.full_like(x, -127)
        m1 = (x > -128) & (x <= -75)
        out = torch.where(m1, x_div4 - 88, out)
        m2 = (x > -75) & (x <= -32)
        out = torch.where(m2, x - 32, out)
        m3 = (x > -32) & (x <= 31)
        out = torch.where(m3, 2 * x, out)
        m4 = (x > 31) & (x <= 74)
        out = torch.where(m4, x + 32, out)
        m5 = (x > 74) & (x <= 127)
        out = torch.where(m5, x_div4 + 88, out)
        m6 = (x > 127)
        out = torch.where(m6, torch.full_like(x, 127), out)
        return out
    else:
        out = torch.full_like(x, -127.0)
        m1 = (x > -128.0) & (x <= -75.0)
        out = torch.where(m1, x / 4.0 - 88.0, out)
        m2 = (x > -75.0) & (x <= -32.0)
        out = torch.where(m2, x - 32.0, out)
        m3 = (x > -32.0) & (x <= 31.0)
        out = torch.where(m3, 2.0 * x, out)
        m4 = (x > 31.0) & (x <= 74.0)
        out = torch.where(m4, x + 32.0, out)
        m5 = (x > 74.0) & (x <= 127.0)
        out = torch.where(m5, x / 4.0 + 88.0, out)
        m6 = (x > 127.0)
        out = torch.where(m6, torch.full_like(x, 127.0), out)
        return out


def pocket_tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of PocketTanh:
      0       (x <= -128 or x > 127)
      1/4     (-128 < x <= -75 or 74 < x <= 127)
      1       (-75 < x <= -32 or 31 < x <= 74)
      2       (-32 < x <= 31)
    """
    if not x.is_floating_point():
        x_f = x.float()
    else:
        x_f = x

    deriv = torch.zeros_like(x_f)
    m_1_4 = ((x_f > -128.0) & (x_f <= -75.0)) | ((x_f > 74.0) & (x_f <= 127.0))
    deriv = torch.where(m_1_4, torch.full_like(x_f, 0.25), deriv)
    m_1 = ((x_f > -75.0) & (x_f <= -32.0)) | ((x_f > 31.0) & (x_f <= 74.0))
    deriv = torch.where(m_1, torch.full_like(x_f, 1.0), deriv)
    m_2 = (x_f > -32.0) & (x_f <= 31.0)
    deriv = torch.where(m_2, torch.full_like(x_f, 2.0), deriv)
    return deriv


def pocket_relu_8bit(x: torch.Tensor) -> torch.Tensor:
    """
    PocketReLU8bit bounded in [0, 127].
    min(max(0, x), 127)
    """
    return torch.clamp(x, min=0, max=127)


def pocket_relu_8bit_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivative of PocketReLU8bit:
      1   (0 < x < 127)
      0   otherwise
    """
    if not x.is_floating_point():
        x_f = x.float()
    else:
        x_f = x
    return torch.where((x_f > 0.0) & (x_f < 127.0), torch.ones_like(x_f), torch.zeros_like(x_f))


def apply_pocket_activation(x: torch.Tensor, actv_name: str) -> torch.Tensor:
    """Dispatch helper for pocket activation functions."""
    if actv_name == 'pocket_tanh':
        return pocket_tanh(x)
    elif actv_name == 'pocket_sigmoid':
        return pocket_sigmoid(x)
    elif actv_name == 'pocket_relu8bit':
        return pocket_relu_8bit(x)
    elif actv_name == 'identity':
        return x
    else:
        raise ValueError(f"Unknown pocket activation: {actv_name}")


def apply_pocket_derivative(x: torch.Tensor, actv_name: str) -> torch.Tensor:
    """Dispatch helper for pocket activation derivatives."""
    if actv_name == 'pocket_tanh':
        return pocket_tanh_derivative(x)
    elif actv_name == 'pocket_sigmoid':
        return pocket_sigmoid_derivative(x)
    elif actv_name == 'pocket_relu8bit':
        return pocket_relu_8bit_derivative(x)
    elif actv_name == 'identity':
        return torch.ones_like(x.float())
    else:
        raise ValueError(f"Unknown pocket activation: {actv_name}")


# ============================================================================
# 2. POCKET DENSE LAYER WITH DIRECT FEEDBACK ALIGNMENT (DFA)
# ============================================================================

class PocketLinear(nn.Module):
    """
    PocketNN Linear Layer with Direct Feedback Alignment (DFA).
    
    Attributes:
      in_features: input dimension d_{k-1}
      out_features: output dimension d_k
      final_dim: network output dimension d_n
      weights: int32 tensor of shape (in_features, out_features)
      bias: int32 tensor of shape (1, out_features)
      feedback_R: fixed random feedback matrix of shape (final_dim, out_features)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        final_dim: int,
        actv: str = 'pocket_tanh',
        feedback_bound: int = 127,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.final_dim = final_dim
        self.actv = actv

        # Initialize weights and bias as int32 zeros per paper Section 4
        self.register_buffer('weight', torch.zeros((in_features, out_features), dtype=torch.int32))
        self.register_buffer('bias', torch.zeros((1, out_features), dtype=torch.int32))

        # Fixed random feedback matrix R^[k] in range [-feedback_bound, feedback_bound]
        # Shape is (final_dim, out_features) such that (y_hat - y) @ R has shape (batch_size, out_features)
        R_init = torch.randint(-feedback_bound, feedback_bound + 1, (final_dim, out_features), dtype=torch.int32)
        # Avoid all-zero columns in R
        R_init = torch.where(R_init == 0, torch.ones_like(R_init), R_init)
        self.register_buffer('feedback_R', R_init)

        # Cache for backward pass
        self.cached_input: Optional[torch.Tensor] = None
        self.cached_pre_actv: Optional[torch.Tensor] = None
        self.cached_output: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Eqs. 1-3):
          h^[k] = a^[k-1] W^[k] + b^[k]
          a^[k] = actv^[k](h^[k])
        """
        if x.dtype != torch.int32:
            x_int = x.to(torch.int32)
        else:
            x_int = x

        self.cached_input = x_int
        # Pure integer matmul and bias addition
        h = torch.matmul(x_int, self.weight) + self.bias
        self.cached_pre_actv = h
        a = apply_pocket_activation(h, self.actv)
        self.cached_output = a
        return a

    def compute_dfa_delta(self, output_error: torch.Tensor) -> torch.Tensor:
        """
        Compute DFA delta (Eq. 10):
          delta^[k]_DFA = ((y_hat - y) * R^[k]) (*) actv'^[k](h^[k])
        where output_error = (y_hat - y) of shape (batch, final_dim).
        """
        if output_error.dtype != torch.int32:
            err_int = output_error.to(torch.int32)
        else:
            err_int = output_error

        # Matrix product with fixed random feedback matrix: (batch, final_dim) @ (final_dim, out_features)
        feedback_signal = torch.matmul(err_int, self.feedback_R)  # shape: (batch, out_features)

        # Derivative of activation at cached pre-activation
        deriv = apply_pocket_derivative(self.cached_pre_actv, self.actv)

        # Hadarmard product: multiply feedback by derivative
        # Handled cleanly via integer or fractional product
        delta_float = feedback_signal.float() * deriv
        return delta_float

    def update_parameters(self, delta: torch.Tensor, lr_inv: int) -> None:
        """
        Integer parameter update with reciprocal learning rate:
          dW = (a^[k-1])^T @ delta / lr_inv
          db = sum(delta, dim=0) / lr_inv
        """
        batch_size = delta.shape[0]
        a_in = self.cached_input.float()

        # Gradient with respect to weight: shape (in_features, out_features)
        grad_W = torch.matmul(a_in.t(), delta)
        # Gradient with respect to bias: shape (1, out_features)
        grad_b = torch.sum(delta, dim=0, keepdim=True)

        # Average over batch and apply integer lr_inv scaling
        dW = torch.div(grad_W / batch_size, lr_inv, rounding_mode='trunc').to(torch.int32)
        db = torch.div(grad_b / batch_size, lr_inv, rounding_mode='trunc').to(torch.int32)

        self.weight.sub_(dW)
        self.bias.sub_(db)


# ============================================================================
# 3. POCKET NEURAL NETWORK ARCHITECTURE (SECTIONS 3 & 4)
# ============================================================================

class PocketNN(nn.Module):
    """
    PocketNN: Full Integer-only Deep Neural Network via Direct Feedback Alignment (DFA).
    
    Supports:
    - Pure integer forward pass
    - Independent layer-wise DFA backward pass (zero recursive overflow)
    - Sum of Squared Errors (SSE) loss
    - Reciprocal integer learning rate updates
    """
    def __init__(
        self,
        layer_dims: List[int],
        hidden_actv: str = 'pocket_tanh',
        output_actv: str = 'identity',
        feedback_bound: int = 127,
    ):
        super().__init__()
        assert len(layer_dims) >= 2, "Network must have at least input and output dimensions"
        self.layer_dims = layer_dims
        self.final_dim = layer_dims[-1]
        self.layers = nn.ModuleList()

        for k in range(len(layer_dims) - 1):
            in_dim = layer_dims[k]
            out_dim = layer_dims[k + 1]
            actv = hidden_actv if (k < len(layer_dims) - 2) else output_actv
            self.layers.append(
                PocketLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    final_dim=self.final_dim,
                    actv=actv,
                    feedback_bound=feedback_bound,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential forward pass across all layers."""
        curr = x
        for layer in self.layers:
            curr = layer(curr)
        return curr

    def train_step_dfa(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr_inv: int = 100,
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Execute one complete integer training step via Direct Feedback Alignment (DFA).
        
        Args:
          x: Input batch (batch_size, in_features)
          y: Target batch (batch_size, final_dim)
          lr_inv: Reciprocal learning rate (e.g. 100 for lr=0.01)
          
        Returns:
          y_hat: Model predictions
          sse_loss: Scalar Sum of Squared Errors (SSE)
          delta_norms: L2 norm of delta vectors per layer for analysis
        """
        # 1. Forward Pass
        y_hat = self.forward(x)

        # 2. Output Error (y_hat - y) for SSE loss J = 1/2 sum (y_hat - y)^2
        err = (y_hat - y).to(torch.int32)
        sse_loss = torch.sum((err.float() ** 2) / 2.0).item()

        # 3. Independent DFA Backward Pass across each layer
        delta_norms = []
        for layer in self.layers:
            delta_k = layer.compute_dfa_delta(err)
            delta_norms.append(torch.norm(delta_k).item())
            layer.update_parameters(delta_k, lr_inv=lr_inv)

        return y_hat, int(sse_loss), delta_norms


# ============================================================================
# 4. OVERFLOW ANALYSIS COMPARATOR: BP VS DFA (SECTION 3.1 & 3.2, FIGURE 1)
# ============================================================================

def compare_bp_vs_dfa_delta_bounds(
    num_layers: int = 5,
    error_bits: int = 8,
    weight_bits: int = 8,
    feedback_bits: int = 8,
) -> Dict[str, List[float]]:
    """
    Computes theoretical bit-width bounds of error deltas for Backpropagation (BP)
    versus Direct Feedback Alignment (DFA) across network layers (Section 3.1 & 3.2).
    
    BP Delta bit-width at layer k (from output layer):
      Bound_BP(k) = error_bits + k * weight_bits (exponential growth)
    DFA Delta bit-width at layer k:
      Bound_DFA(k) = error_bits + feedback_bits (strictly constant)
    """
    bp_bounds = []
    dfa_bounds = []

    for k in range(1, num_layers + 1):
        bp_bit = error_bits + k * weight_bits
        dfa_bit = error_bits + feedback_bits
        bp_bounds.append(bp_bit)
        dfa_bounds.append(dfa_bit)

    return {
        'layer_indices': list(range(1, num_layers + 1)),
        'bp_delta_bit_bounds': bp_bounds,
        'dfa_delta_bit_bounds': dfa_bounds,
        'bp_exceeds_int32_at_layer': next((k for k, b in enumerate(bp_bounds, 1) if b > 32), None),
    }
