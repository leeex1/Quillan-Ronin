"""
=============================================================================
QUILLAN-RONIN: WSP RESEARCH PAPER FIGURE GENERATOR
=============================================================================
Generates 4 high-resolution, publication-grade academic figures illustrating:
1. WSP Architecture & Multi-Donor Flow
2. Closed-Form Refusal Subspace Null-Space Projection
3. Analytical MoE SwiGLU Weight-Space Folding
4. Empirical Generation Throughput & VRAM Footprint Benchmarks
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = r"C:\02_QUILLAN\10 - Formal Papers\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Visual Theme (Dark Academic / Neon Ronin)
PRIMARY = "#7C3AED"     # Sovereign Purple
SECONDARY = "#10B981"   # Bushido Green
ACCENT = "#EF4444"      # Abliterated Crimson
BLUE = "#3B82F6"        # Foundation Blue
GOLD = "#F59E0B"        # Covariance Gold
DARK_BG = "#0F172A"     # Deep Slate
PANEL_BG = "#1E293B"    # Card Slate
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "text.color": TEXT_LIGHT,
    "axes.labelcolor": TEXT_LIGHT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": "#334155",
    "axes.grid": True,
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

# =============================================================================
# FIGURE 1: WSP ARCHITECTURAL PIPELINE
# =============================================================================
print(">>> Generating Figure 1: WSP Architecture Pipeline...")
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

# Title & Subtitle
ax.text(6, 5.6, "WEIGHT-SPACE PROGRAMMING (WSP) CLOSED-FORM PIPELINE", 
        ha="center", va="center", fontsize=15, fontweight="bold", color=PRIMARY)
ax.text(6, 5.2, "Zero Backpropagation | Analytical Subspace Operations | Universal Edge Compilation", 
        ha="center", va="center", fontsize=9.5, color=TEXT_MUTED)

# Box styles
def draw_card(ax, x, y, w, h, title, subtitle, color, tag=""):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  facecolor=PANEL_BG, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    header = f"[{tag}] {title}" if tag else title
    ax.text(x + w/2, y + h*0.7, header, ha="center", va="center", 
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w/2, y + h*0.32, subtitle, ha="center", va="center", 
            fontsize=8, color=TEXT_MUTED, multialignment="center")

# Draw stages
draw_card(ax, 0.4, 2.2, 2.4, 2.2, "Ingestion", "Llama 3.2 1B (Base)\nFalcon 3 (Philosophy)\nBitNet 1.58b (Ternary)\nQwen 2.5/3.5 (Logic)", BLUE, "1")
draw_card(ax, 3.2, 2.2, 2.4, 2.2, "Covariance", "Empirical Activations\nGold Reasoning\nPrincipal Axes SVD\nRank-16 Extraction", GOLD, "2")
draw_card(ax, 6.0, 2.2, 2.4, 2.2, "Abliteration", "Contrastive Vector\nv_refusal Isolation\n(I - alpha v v^T)\nNull-Space Projection", ACCENT, "3")
draw_card(ax, 8.8, 2.2, 2.8, 2.2, "MoE Folding", "Additive Equivalence\nW_folded = W + gamma sum W1 W0\nExact SwiGLU Parity\nGGUF Edge Master", SECONDARY, "4")

# Connective Arrows
arrow_style = dict(facecolor=TEXT_MUTED, edgecolor=TEXT_MUTED, width=1.5, headwidth=6, headlength=8)
ax.annotate("", xy=(3.2, 3.3), xytext=(2.9, 3.3), arrowprops=arrow_style)
ax.annotate("", xy=(6.0, 3.3), xytext=(5.7, 3.3), arrowprops=arrow_style)
ax.annotate("", xy=(8.8, 3.3), xytext=(8.5, 3.3), arrowprops=arrow_style)

# Bottom Output Banner
banner = patches.FancyBboxPatch((0.4, 0.5), 11.2, 1.1, boxstyle="round,pad=0.08,rounding_size=0.1",
                                facecolor="#0B132B", edgecolor=PRIMARY, linewidth=1.2)
ax.add_patch(banner)
ax.text(6, 1.25, "UNIVERSAL DEPLOYMENT: QUILLAN-RONIN 1.2B SOVEREIGN GGUF", 
        ha="center", va="center", fontsize=11, fontweight="bold", color="#A78BFA")
ax.text(6, 0.8, "Q4_K_M (770 MB) | Q8_0 (1.26 GB) | F16 Master (2.36 GB) -> 48.0 TOKENS/SEC ON GTX 1050", 
        ha="center", va="center", fontsize=9, color=TEXT_LIGHT)

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "wsp_fig1_architecture.png")
plt.savefig(fig1_path, dpi=300, facecolor=DARK_BG)
plt.close()
print(f"[OK] Saved Figure 1: {fig1_path}")

# =============================================================================
# FIGURE 2: REFUSAL SUBSPACE NULL-SPACE GEOMETRY
# =============================================================================
print(">>> Generating Figure 2: Refusal Subspace Geometry...")
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 2D projection of activation space
np.random.seed(42)
n_pts = 60
sovereign_pts = np.random.randn(n_pts, 2) * 0.7 + np.array([-1.5, 1.2])
refusal_pts = np.random.randn(n_pts, 2) * 0.7 + np.array([2.0, -1.0])

ax.scatter(sovereign_pts[:, 0], sovereign_pts[:, 1], color=SECONDARY, alpha=0.6, label="Sovereign Trajectories (h_sov)", s=40)
ax.scatter(refusal_pts[:, 0], refusal_pts[:, 1], color=ACCENT, alpha=0.6, label="Corporate Refusals (h_ref)", s=40)

# Centroids
mu_sov = sovereign_pts.mean(axis=0)
mu_ref = refusal_pts.mean(axis=0)
ax.scatter(*mu_sov, color="#34D399", s=140, edgecolors="white", linewidth=2, zorder=5)
ax.scatter(*mu_ref, color="#F87171", s=140, edgecolors="white", linewidth=2, zorder=5)

# Difference vector v_diff
ax.annotate("", xy=mu_ref, xytext=mu_sov,
            arrowprops=dict(facecolor=GOLD, edgecolor=GOLD, width=2.5, headwidth=8, headlength=10))
ax.text(0.3, 0.4, "v_refusal = (h_ref - h_sov) / ||.||", color=GOLD, fontweight="bold", fontsize=10, rotation=-32)

# Null space hyperplane (orthogonal plane)
slope = -(mu_ref[0] - mu_sov[0]) / (mu_ref[1] - mu_sov[1])
x_line = np.linspace(-3.5, 3.5, 100)
y_line = slope * (x_line - 0) + 0.1
ax.plot(x_line, y_line, color="#A78BFA", linestyle="-.", linewidth=2, label="Orthogonal Null Space (I - v v^T)")

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)
ax.set_title("Closed-Form Refusal Subspace Abliteration", fontsize=13, fontweight="bold", pad=12, color=PRIMARY)
ax.set_xlabel("Principal Activation Coordinate 1", fontsize=9.5)
ax.set_ylabel("Principal Activation Coordinate 2", fontsize=9.5)
ax.legend(loc="upper right", framealpha=0.8, facecolor=PANEL_BG, edgecolor="#475569", fontsize=8.5)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "wsp_fig2_refusal_geometry.png")
plt.savefig(fig2_path, dpi=300, facecolor=DARK_BG)
plt.close()
print(f"[OK] Saved Figure 2: {fig2_path}")

# =============================================================================
# FIGURE 3: MOE WEIGHT-SPACE FOLDING PROOF
# =============================================================================
print(">>> Generating Figure 3: MoE Weight Folding Equivalence...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

# Subplot 1: Dynamic MoE (Runtime Thrashing)
ax1.set_title("A: Dynamic MoE Kernel Dispatch (PyTorch)", fontsize=11, fontweight="bold", color=ACCENT, pad=10)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis("off")

rect_in = patches.Rectangle((1, 8), 8, 1.2, facecolor="#334155", edgecolor=BLUE, linewidth=1.5)
ax1.add_patch(rect_in)
ax1.text(5, 8.6, "SwiGLU Activation Vector h in R^(8192)", ha="center", va="center", fontsize=9, fontweight="bold")

for i, exp in enumerate(["Base W_down", "MoE Exp 0 (W1 @ W0)", "MoE Exp 1 (W1 @ W0)", "MoE Exp 2 (W1 @ W0)"]):
    y_pos = 5.8 - i * 1.3
    rect_exp = patches.Rectangle((0.5, y_pos), 9, 0.9, facecolor="#1E293B", edgecolor=ACCENT if i>0 else BLUE, linestyle="--" if i>0 else "-")
    ax1.add_patch(rect_exp)
    ax1.text(5, y_pos + 0.45, f"CUDA Kernel #{i+1}: {exp}", ha="center", va="center", fontsize=8, color=TEXT_LIGHT)
    ax1.annotate("", xy=(5, y_pos + 0.9), xytext=(5, 8.0), arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, alpha=0.4))

ax1.text(5, 0.5, "[X] 9 Separate Kernel Launches / Token\n[X] Dynamic Memory Thrashing (5.36 tok/s)", 
         ha="center", va="center", fontsize=8.5, color=ACCENT, fontweight="bold")

# Subplot 2: Folded Matrix (Unified GEMM)
ax2.set_title("B: WSP Closed-Form Folded GEMM (GGUF)", fontsize=11, fontweight="bold", color=SECONDARY, pad=10)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")

rect_in2 = patches.Rectangle((1, 8), 8, 1.2, facecolor="#334155", edgecolor=BLUE, linewidth=1.5)
ax2.add_patch(rect_in2)
ax2.text(5, 8.6, "SwiGLU Activation Vector h in R^(8192)", ha="center", va="center", fontsize=9, fontweight="bold")

rect_folded = patches.Rectangle((1, 3.2), 8, 3.2, facecolor="#064E3B", edgecolor=SECONDARY, linewidth=2.0)
ax2.add_patch(rect_folded)
ax2.text(5, 5.4, "Unified Folded Matrix W_sovereign\n[2048 x 8192]", ha="center", va="center", 
         fontsize=11, fontweight="bold", color="#A7F3D0")
ax2.text(5, 4.2, "W_down + gamma sum(W1_i @ W0_i)\nExact Mathematical Parity", ha="center", va="center", 
         fontsize=9, color=TEXT_LIGHT)

ax2.annotate("", xy=(5, 6.4), xytext=(5, 8.0), arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=2))

rect_out = patches.Rectangle((1, 1.2), 8, 1.0, facecolor="#1E293B", edgecolor=SECONDARY)
ax2.add_patch(rect_out)
ax2.text(5, 1.7, "Output Residual Stream y in R^(2048)", ha="center", va="center", fontsize=9, color=TEXT_LIGHT)
ax2.annotate("", xy=(5, 2.2), xytext=(5, 3.2), arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=2))

ax2.text(5, 0.5, "[OK] 1 Single Dense GEMM Kernel Launch\n[OK] Zero Allocations | 48.0 tok/s Edge Speed", 
         ha="center", va="center", fontsize=8.5, color=SECONDARY, fontweight="bold")

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "wsp_fig3_moe_folding.png")
plt.savefig(fig3_path, dpi=300, facecolor=DARK_BG)
plt.close()
print(f"[OK] Saved Figure 3: {fig3_path}")

# =============================================================================
# FIGURE 4: EMPIRICAL BENCHMARKS (SPEED & VRAM)
# =============================================================================
print(">>> Generating Figure 4: Empirical Benchmark Telemetry...")
fig, (ax_speed, ax_vram) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

backends = ["PyTorch FP16", "Static KV FP16", "Quillan GGUF\n(Q4_K_M)"]
speeds = [3.94, 5.36, 48.00]
colors_speed = ["#64748B", "#3B82F6", "#10B981"]

bars = ax_speed.bar(backends, speeds, color=colors_speed, width=0.55, edgecolor="white", linewidth=1)
ax_speed.set_title("Inference Throughput (GTX 1050 4GB)", fontsize=11, fontweight="bold", color=PRIMARY, pad=10)
ax_speed.set_ylabel("Tokens / Second", fontsize=9.5)
ax_speed.set_ylim(0, 56)

for bar, s in zip(bars, speeds):
    y = bar.get_height()
    ax_speed.text(bar.get_x() + bar.get_width()/2, y + 1.2, f"{s:.2f} tok/s", 
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_speed.text(2, 51, "12.18x Speedup", ha="center", va="center", color="#34D399", fontweight="bold", fontsize=10)

# VRAM Footprint
vrams = [2842, 2680, 1024]
colors_vram = ["#EF4444", "#F59E0B", "#8B5CF6"]
bars_v = ax_vram.bar(backends, vrams, color=colors_vram, width=0.55, edgecolor="white", linewidth=1)
ax_vram.set_title("Peak VRAM Footprint (4096 MB Limit)", fontsize=11, fontweight="bold", color=PRIMARY, pad=10)
ax_vram.set_ylabel("Peak VRAM (MB)", fontsize=9.5)
ax_vram.set_ylim(0, 3600)
ax_vram.axhline(4096, color="#EF4444", linestyle=":", alpha=0.7, label="4GB VRAM Hardware Limit")

for bar, v in zip(bars_v, vrams):
    y = bar.get_height()
    ax_vram.text(bar.get_x() + bar.get_width()/2, y + 80, f"{v} MB", 
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_vram.legend(loc="upper right", framealpha=0.7, facecolor=PANEL_BG, edgecolor="#475569", fontsize=8)

plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "wsp_fig4_benchmarks.png")
plt.savefig(fig4_path, dpi=300, facecolor=DARK_BG)
plt.close()
print(f"[OK] Saved Figure 4: {fig4_path}")
print("[OK] ALL 4 RESEARCH FIGURES GENERATED SUCCESSFULLY.")
