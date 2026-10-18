"""
QuillanFusedOptimizer — v1.0
================================
Combines the best of all three optimizer traditions:

  1. MUON  — Newton-Schulz orthogonalization for matrix weights (2D and 3D MoE).
             Prevents gradient collapse along low-variance directions.
             Faster convergence, fewer steps needed vs AdamW alone.
             Origin: modded-nanoGPT → Kimi K2 → Quillan lineage.

  2. SOPHIA-H — Diagonal Hessian EMA curvature clipping applied to the
                orthogonalized Muon direction. Prevents large steps in
                high-curvature regions. Clamps update per-element.

  3. CCRL / CouncilOptimizer concepts:
       - Gradient Centralization (zero-mean before any update)
       - PID Thermo Governor  (scale LR by training health signal)
       - Gradient Noise Scale EMA (adaptive damping)
       - BitNet Dead-Zone (snap near-zero weights to 0 post-step)
       - Nesterov look-ahead momentum

Design goals:
  - Pre-classified param groups at __init__ (NO per-step routing overhead)
  - CPU-offloaded momentum buffers (fits 4 GB Pascal VRAM)
  - Single .step() call, no closures needed
  - Drop-in .state_dict() / .load_state_dict() for checkpointing

Param groups:
  - muon_params   : dim >= 2 (weight matrices, MoE w1/w2 [B,M,N], attention proj)
                    → GC → Nesterov Muon → Sophia-H clip → PID scale → BitNet DZ
  - scalar_params : dim <  2 (biases, 1D embeddings, layer norm scales)
                    → AdamW → PID scale
"""

import math
import logging
import torch

LOG = logging.getLogger(__name__)


# ─── Gradient utilities ───────────────────────────────────────────────────────

def centralize_gradient(g: torch.Tensor) -> torch.Tensor:
    """Zero-mean gradient over all non-output dims (Gradient Centralization)."""
    if g.dim() < 2:
        return g
    reduce_dims = list(range(1, g.dim()))
    return g - g.mean(dim=reduce_dims, keepdim=True)


def newton_schulz(G: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iteration.

    Supports 2D [M, N] and batched 3D [B, M, N] (MoE expert tensors).
    Ensures M <= N before iterating (transposes if needed, restores after).
    Stable coefficients: a=1.5, b=-0.5 (singular values → 1).
    """
    if G.numel() == 0 or G.norm().item() == 0.0:
        return G

    is_3d = G.dim() == 3
    # Ensure M <= N for numerical stability
    if is_3d:
        transposed = G.shape[1] > G.shape[2]
        if transposed:
            G = G.mT             # [B, N, M]
    else:
        transposed = G.shape[0] > G.shape[1]
        if transposed:
            G = G.T              # [N, M]

    X = G / (G.norm() + 1e-8)
    a, b = 1.5, -0.5
    for _ in range(steps):
        if is_3d:
            A = X @ X.mT         # [B, M, M] — batched gram
            X = a * X + b * (A @ X)
        else:
            A = X @ X.T          # [M, M]
            X = a * X + b * (A @ X)

    if transposed:
        X = X.mT if is_3d else X.T
    return X


# ─── PID Thermo Governor (from CCRL) ─────────────────────────────────────────

class PIDGovernor:
    """PID-based LR damping.  integrity=1.0 → no damping.  integrity=0 → full stop."""

    def __init__(self, kp: float = 0.1, ki: float = 0.01, kd: float = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = 0.0

    def step(self, integrity: float = 1.0) -> float:
        error = 1.0 - max(0.0, min(1.0, integrity))
        self._integral += error
        d_term = error - self._prev_error
        self._prev_error = error
        scale = 1.0 - (self.kp * error + self.ki * self._integral + self.kd * d_term)
        return max(0.0, min(1.0, scale))


# ─── Gradient Noise Scale EMA (from CCRL) ────────────────────────────────────

class NoiseScaleEMA:
    """Tracks gradient noise via exponential moving average of gradient norms."""

    def __init__(self, decay: float = 0.99):
        self._ema   = 1.0
        self._decay = decay

    def step(self, grads) -> float:
        norms = [g.norm().item() ** 2 for g in grads if g is not None]
        if norms:
            current = float(sum(norms) ** 0.5)
            self._ema = self._decay * self._ema + (1 - self._decay) * current
        return self._ema


# ─── Main Optimizer ───────────────────────────────────────────────────────────

class QuillanFusedOptimizer:
    """
    Unified Quillan optimizer — Muon + Sophia-H + CCRL PID + AdamW.

    Args:
        model           : nn.Module whose parameters to optimize
        lr_muon         : Muon base LR for matrix weights (default 0.02)
        lr_adamw        : AdamW base LR for 1D params (default 2e-4)
        lr_min          : Cosine decay floor for both groups (default 5e-6)
        momentum        : Muon Nesterov momentum (default 0.95)
        weight_decay    : Decoupled weight decay for both groups (default 0.01)
        ns_steps        : Newton-Schulz iterations — 3 is optimal (default 3)
        sophia_rho      : Sophia-H clamp threshold (default 0.05)
        sophia_beta     : Sophia-H Hessian EMA decay (default 0.999)
        gradient_centralization : Apply GC before Muon (default True)
        bitnet_delta    : Dead-zone threshold — 0=disabled (default 0.005)
        pid_kp/ki/kd    : PID governor coefficients
        noise_decay     : Gradient noise EMA decay (default 0.99)
        warmup          : LR warmup steps (default 300)
        total_steps     : Total training steps for cosine decay (default 90000)
        start_step      : Resume step (offset for LR schedule) (default 0)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr_muon:    float = 0.02,
        lr_adamw:   float = 2e-4,
        lr_min:     float = 5e-6,
        momentum:   float = 0.95,
        weight_decay: float = 0.01,
        ns_steps:   int   = 3,
        sophia_rho:  float = 0.05,
        sophia_beta: float = 0.999,
        gradient_centralization: bool = True,
        bitnet_delta: float = 0.0,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.001,
        noise_decay: float = 0.99,
        warmup:      int   = 300,
        total_steps: int   = 90000,
        start_step:  int   = 0,
        use_sophia:  bool  = False,
    ):
        self.lr_muon     = lr_muon
        self.lr_adamw    = lr_adamw
        self.lr_min      = lr_min
        self.momentum    = momentum
        self.wd          = weight_decay
        self.ns_steps    = ns_steps
        self.sophia_rho  = sophia_rho
        self.sophia_beta = sophia_beta
        self.gc          = gradient_centralization
        self.bd          = bitnet_delta
        self.warmup      = warmup
        self.total_steps = total_steps
        self.start_step  = start_step
        self._step       = start_step
        self.use_sophia  = use_sophia

        self._pid   = PIDGovernor(pid_kp, pid_ki, pid_kd)
        self._noise = NoiseScaleEMA(noise_decay)

        # ── Pre-classify all params at init ──────────────────────────────
        self.muon_params   = []  # dim >= 2
        self.scalar_params = []  # dim <  2

        # CPU fp32 buffers — keeps VRAM footprint minimal
        self.muon_buf     = []   # Nesterov momentum  [same shape as param]
        self.muon_hess    = []   # Sophia-H diag Hessian EMA [same shape]
        self.scalar_state = []   # AdamW: {step, m1, m2}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Keep MoE weights in AdamW group to bypass expensive 3D orthogonalization
            is_moe_weight = any(x in name for x in ['moe.w1', 'moe.wgate', 'moe.w2'])
            if p.dim() >= 2 and not is_moe_weight:
                self.muon_params.append(p)
                self.muon_buf.append(
                    torch.zeros(p.shape, dtype=torch.float32, device='cpu'))
                if self.use_sophia:
                    self.muon_hess.append(
                        torch.ones(p.shape,  dtype=torch.float32, device='cpu'))
            else:
                self.scalar_params.append(p)
                self.scalar_state.append({
                    'step': 0,
                    'm1': torch.zeros(p.shape, dtype=torch.float32, device='cpu'),
                    'm2': torch.zeros(p.shape, dtype=torch.float32, device='cpu'),
                })

        muon_n   = sum(p.numel() for p in self.muon_params)
        scalar_n = sum(p.numel() for p in self.scalar_params)
        LOG.info(
            f"QuillanFusedOptimizer: "
            f"{muon_n/1e6:.1f}M Muon params (matrix), "
            f"{scalar_n/1e6:.1f}M AdamW params (scalar) | "
            f"LR Muon={lr_muon:.2e}, AdamW={lr_adamw:.2e}"
        )

    # ── LR schedule ──────────────────────────────────────────────────────────

    def _lr(self, step: int, base: float) -> float:
        """Cosine decay with warmup, floored at lr_min."""
        s = step - self.start_step
        if s < self.warmup:
            return base * max(0.01, (s + 1) / self.warmup)
        prog = (s - self.warmup) / max(1, (self.total_steps - self.start_step) - self.warmup)
        prog = min(prog, 1.0)
        return self.lr_min + 0.5 * (base - self.lr_min) * (1 + math.cos(math.pi * prog))

    def current_lr_muon(self) -> float:
        return self._lr(self._step, self.lr_muon)

    def current_lr_adamw(self) -> float:
        return self._lr(self._step, self.lr_adamw)

    # ── zero_grad ─────────────────────────────────────────────────────────────

    def zero_grad(self) -> None:
        for p in self.muon_params + self.scalar_params:
            p.grad = None

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, integrity: float = 1.0) -> dict:
        """Perform one optimizer step.

        Args:
            integrity: 0-1 training health signal for PID governor.
                       1.0 = fully healthy (no damping).
                       Lower values damp the LR proportionally.

        Returns:
            dict with 'lr_muon', 'lr_adamw', 'pid_scale', 'noise_ema'
        """
        self._step += 1

        # Noise scale from all active gradients
        all_grads = [
            p.grad for p in (self.muon_params + self.scalar_params)
            if p.grad is not None
        ]
        noise_ema = self._noise.step(all_grads)
        pid_scale = self._pid.step(integrity)

        # Governor: combined damping factor
        gov = pid_scale / (1.0 + noise_ema * 0.01)  # 0.01: soften noise effect
        gov = max(0.05, min(1.0, gov))               # clamp to [5%, 100%]

        lr_m = self._lr(self._step, self.lr_muon)  * gov
        lr_a = self._lr(self._step, self.lr_adamw) * gov

        # ── MUON GROUP ────────────────────────────────────────────────────
        # Pipeline per-param:
        #   GC → Nesterov momentum → Newton-Schulz ortho → Sophia-H clip
        #   → weight decay → param update → BitNet dead-zone
        for i, (p, buf) in enumerate(zip(self.muon_params, self.muon_buf)):
            if p.grad is None:
                continue

            # Keep gradient on GPU in float32 for fast computation
            g = p.grad.detach().float()
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. Gradient Centralization (on GPU)
            if self.gc:
                g = centralize_gradient(g)

            # 2. Nesterov momentum: temporarily move CPU buffer to GPU
            buf_gpu = buf.to(device=p.device, dtype=torch.float32)
            buf_gpu.mul_(self.momentum).add_(g)
            g_nes = g + self.momentum * buf_gpu

            # Write updated momentum back to CPU (offloaded buffer)
            buf.copy_(buf_gpu.cpu())
            del buf_gpu

            # 3. Newton-Schulz orthogonalization (on GPU - extremely fast!)
            ortho = newton_schulz(g_nes, steps=self.ns_steps)

            # 4. Sophia-H: diagonal Hessian EMA clip
            if self.use_sophia:
                # hess is on CPU, move to GPU
                hess_gpu = self.muon_hess[i].to(device=p.device, dtype=torch.float32)
                # Hessian estimate: g^2 (Gauss-Newton diagonal approximation)
                hess_gpu.mul_(self.sophia_beta).addcmul_(g, g, value=1 - self.sophia_beta)
                # Save back to CPU
                self.muon_hess[i].copy_(hess_gpu.cpu())
                # Clip orthogonalized direction by curvature
                ortho = ortho / hess_gpu.clamp(min=self.sophia_rho / 100)
                ortho = ortho.clamp(-self.sophia_rho, self.sophia_rho)
                del hess_gpu

            # 5. Weight decay (decoupled, applied to GPU param)
            if self.wd != 0.0:
                p.data.mul_(1.0 - lr_m * self.wd)

            # 6. Apply update on GPU
            p.data.add_(ortho.to(dtype=p.dtype), alpha=-lr_m)

            # 7. BitNet dead-zone: snap near-zero ternary weights to exactly 0
            if self.bd > 0.0:
                with torch.no_grad():
                    p.data[p.data.abs() < self.bd] = 0.0

        # ── ADAMW GROUP ───────────────────────────────────────────────────
        for p, st in zip(self.scalar_params, self.scalar_state):
            if p.grad is None:
                continue

            # Copy float16 gradient to CPU first to save PCIe bandwidth, then upcast to float32 on CPU
            g = p.grad.detach().cpu().float()
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

            st['step'] += 1
            t  = st['step']
            m1 = st['m1']
            m2 = st['m2']

            m1.mul_(0.9).add_(g, alpha=0.1)
            m2.mul_(0.999).addcmul_(g, g, value=0.001)

            bc1 = 1.0 - 0.9   ** t
            bc2 = 1.0 - 0.999 ** t
            m_hat = m1 / bc1
            v_hat = (m2 / bc2).sqrt().add_(1e-8)

            if self.wd != 0.0:
                p.data.mul_(1.0 - lr_a * self.wd)

            delta = (m_hat / v_hat) * lr_a
            # Downcast to parameter's dtype (float16) on CPU first to save PCIe transfer bandwidth
            p.data.add_(delta.to(dtype=p.dtype).to(device=p.device), alpha=-1.0)

        return {
            'lr_muon':   lr_m,
            'lr_adamw':  lr_a,
            'pid_scale': pid_scale,
            'noise_ema': noise_ema,
            'gov':       gov,
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        sd = {
            'step':         self._step,
            'muon_buf':     [b.clone() for b in self.muon_buf],
            'scalar_state': self.scalar_state,
            'pid_integral': self._pid._integral,
            'pid_prev_err': self._pid._prev_error,
            'noise_ema':    self._noise._ema,
        }
        if self.use_sophia:
            sd['muon_hess'] = [h.clone() for h in self.muon_hess]
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd.get('step', self.start_step)
        for buf, saved_t in zip(self.muon_buf,  sd.get('muon_buf',  [])):
            buf.copy_(saved_t)
        if self.use_sophia and 'muon_hess' in sd:
            for hess, saved_t in zip(self.muon_hess, sd.get('muon_hess', [])):
                hess.copy_(saved_t)
        for st, saved in zip(self.scalar_state, sd.get('scalar_state', [])):
            st.update(saved)
        if 'pid_integral' in sd:
            self._pid._integral   = sd['pid_integral']
            self._pid._prev_error = sd['pid_prev_err']
        if 'noise_ema' in sd:
            self._noise._ema = sd['noise_ema']
        LOG.info(f"QuillanFusedOptimizer: restored from step {self._step}")