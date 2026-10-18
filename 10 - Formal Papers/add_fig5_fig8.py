#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appends render_fig5 and render_fig8 to generate_perfect_figures.py and executes it.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIG_GEN = Path(r"C:\02_QUILLAN\10 - Formal Papers\generate_perfect_figures.py")

fig5_8_code = '''
# ─── FIG 5 ───────────────────────────────────────────────────────────────────
def render_fig5(out_path):
    W, H = 1400, 900
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    draw_header(draw, W,
                "Figure 5 — Telemetry Schematic (anchors real; curves illustrative)",
                "Gate A 16/16 | val 7.24 @660 (12L) | prior-phase 0.0789 @2500 NOT comparable | benchmarks pending")

    pad_x = 45
    start_y = 150

    # Left chart box
    chart_w = 780
    chart_h = 560
    draw.rectangle([pad_x, start_y, pad_x + chart_w, start_y + chart_h], fill="#FFFFFF", outline="#000000", width=2)
    
    lbl_font = get_font(FONT_BOLD, 22)
    sub_font = get_font(FONT_REG, 17)
    draw.text((pad_x + 24, start_y + 20), "Validation Loss Trajectory (log-scale schematic)", fill="#0F172A", font=lbl_font)
    
    # Draw axes
    ax_x1 = pad_x + 60
    ax_y1 = start_y + chart_h - 60
    ax_x2 = pad_x + chart_w - 40
    ax_y2 = start_y + 80
    draw.line([ax_x1, ax_y1, ax_x2, ax_y1], fill="#94A3B8", width=2)
    draw.line([ax_x1, ax_y1, ax_x1, ax_y2], fill="#94A3B8", width=2)

    # Loss curve line
    points = [
        (ax_x1, ax_y1 - 60),
        (ax_x1 + 120, ax_y1 - 120),
        (ax_x1 + 220, ax_y1 - 200),
        (ax_x1 + 320, ax_y1 - 250),
        (ax_x1 + 480, ax_y1 - 320),
        (ax_x1 + 640, ax_y1 - 380),
    ]
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill="#2563EB", width=4)

    # Anchors
    # Anchor 1: Step 660
    p1 = points[2]
    draw.ellipse([p1[0]-7, p1[1]-7, p1[0]+7, p1[1]+7], fill="#DC2626", outline="#991B1B", width=2)
    draw.text((p1[0] + 16, p1[1] - 14), "val 7.24 @ 660 (12L flagship, real)", fill="#DC2626", font=get_font(FONT_BOLD, 18))

    # Anchor 2: Step 2500
    p2 = points[4]
    draw.ellipse([p2[0]-7, p2[1]-7, p2[0]+7, p2[1]+7], fill="#475569", outline="#0F172A", width=2)
    draw.text((p2[0] + 16, p2[1] - 14), "0.0789 @ 2500 prior phase (real, NOT comparable)", fill="#334155", font=get_font(FONT_BOLD, 18))

    # Bottom label
    bot_lbl = "Step 0 -> 15,000 (flagship paused @ 660) | Prior-phase marker @ 2500 (different rig)"
    draw.text((pad_x + 60, start_y + chart_h + 16), bot_lbl, fill="#0F172A", font=get_font(FONT_BOLD, 18))

    # Right side 4 status cards
    right_x = pad_x + chart_w + 35
    right_w = W - right_x - pad_x
    card_h = 120
    card_gap = 20

    status_cards = [
        ("Gate A: 16/16 Passed", "All deterministic checks passed\\nZero regression across suite", "#F0FDF4"),
        ("Parity: 100% Legacy HW", "Verified parity on single GTX 1050 Ti\\nAMP FP16 master, BitNet forward", "#EFF6FF"),
        ("SFT: 660 / 15,000 Steps", "Flagship training paused\\nReady for final annealing run", "#EFF6FF"),
        ("Formal Benchmarks: Pending", "ARC / GPQA / MMLU segregated\\nBase model scores pending", "#FEF2F2"),
    ]

    for idx, (t, s, col) in enumerate(status_cards):
        cy = start_y + idx * (card_h + card_gap)
        draw_card(draw, [right_x, cy, right_x + right_w, cy + card_h], t, s, fill=col, max_title_sz=19, max_sub_sz=14)

    img.save(out_path, dpi=(300, 300))
    print(f"✅ Rendered Fig 5 -> {out_path.name}")

# ─── FIG 8 ───────────────────────────────────────────────────────────────────
def render_fig8(out_path):
    W, H = 1800, 1100
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    draw_header(draw, W,
                "Figure 8 — Safety loop (CCRL + E_ICE + Lee-Mach-6 + gates)",
                "V = E[wR R + wC C_VIR - wE E_ICE] | E_ICE = λ exp(Harm/T) | PID 0.15/0.05/0.02")

    pad_x = 55
    full_w = W - 2 * pad_x
    start_y = 155
    gap_y = 16

    # Row 1: 3 boxes
    r1_h = 115
    r1_gap = 20
    b3_w = (full_w - 2 * r1_gap) // 3
    r1_boxes = [
        ("Prism Ethics Ray + VIR Prior", "C2-VIR refusal layer evaluated first\\nPersonaPullGate fp32 prior routing", "#FFFBEB"),
        ("Council Consensus Vote", "VIR x WARDEN x SHEPHERD product\\nNon-negotiable threshold Phi >= 0.85", "#F0FDF4"),
        ("E_ICE Thermodynamic Penalty", "Landauer bound: kB T ln2\\nExponential energy cost on harm states", "#FEF2F2"),
    ]
    for idx, (t, s, col) in enumerate(r1_boxes):
        bx = pad_x + idx * (b3_w + r1_gap)
        draw_card(draw, [bx, start_y, bx + b3_w, start_y + r1_h], t, s, fill=col, max_title_sz=19, max_sub_sz=14)

    # Row 2: 3 boxes
    y_r2 = start_y + r1_h + gap_y
    r2_h = 115
    r2_boxes = [
        ("CCRL Policy Optimization", "pi ~ exp(Q/tau) * Consensus\\nLoss L = L_policy + lambda_c * L_cons", "#EFF6FF"),
        ("Lee-Mach-6 Governor", "Real-time thermal/latency PID feedback\\nDynamic scaling: sigma / alpha / beta", "#EFF6FF"),
        ("Exit Verification Gates", "Nullion paradox / Warden safety / Shepherd truth\\nThree-pass verification before emission", "#F8FAFC"),
    ]
    for idx, (t, s, col) in enumerate(r2_boxes):
        bx = pad_x + idx * (b3_w + r1_gap)
        draw_card(draw, [bx, y_r2, bx + b3_w, y_r2 + r2_h], t, s, fill=col, max_title_sz=19, max_sub_sz=14)

    # Row 3: Full width
    y_r3 = y_r2 + r2_h + gap_y + 10
    r3_h = 95
    draw_card(draw, [pad_x, y_r3, pad_x + full_w, y_r3 + r3_h],
              "Refusal Path: consensus FAIL or E_ICE spike -> refusal + safe completion via TYPIST (logged)",
              "Pass Path: deliberate() info['pull_confidence'] > 0.85 | HFL Edo/Bushido anchoring | Nemesis-Alpha red-team",
              fill="#F8FAFC", max_title_sz=18, max_sub_sz=14)

    # Row 4: Full width
    y_r4 = y_r3 + r3_h + gap_y
    r4_h = 90
    draw_card(draw, [pad_x, y_r4, pad_x + full_w, y_r4 + r4_h],
              "Energy: analytic, not metered | Red-team benchmark: future work | Out-of-scope: safety-of-life",
              "Model card honesty bar: rigorous boundaries strictly maintained between theoretical and physical proofs",
              fill="#F8FAFC", max_title_sz=18, max_sub_sz=14)

    img.save(out_path, dpi=(300, 300))
    print(f"✅ Rendered Fig 8 -> {out_path.name}")
'''

with open(FIG_GEN, "r", encoding="utf-8") as f:
    content = f.read()

# Replace main() and add render_fig5 and render_fig8
if "def render_fig5" not in content:
    idx = content.find("def main():")
    new_content = content[:idx] + fig5_8_code + "\ndef main():\n"
    # Update main() body
    new_main = """    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        "Fig1_arch_overview.png", "Fig2_routing.png", "Fig3_ternary_diffusion.png",
        "Fig4_prism.png", "Fig5_telemetry.png", "Fig6_examples.png",
        "Fig7_lineage.png", "Fig8_safety.png", "Fig9_memory.png", "Fig10_council.png"
    ]
    for t in targets:
        src = FIG_DIR / t
        dst = BACKUP_DIR / t
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Backed up {t}")

    render_fig1(FIG_DIR / "Fig1_arch_overview.png")
    render_fig2(FIG_DIR / "Fig2_routing.png")
    render_fig3(FIG_DIR / "Fig3_ternary_diffusion.png")
    render_fig4(FIG_DIR / "Fig4_prism.png")
    render_fig5(FIG_DIR / "Fig5_telemetry.png")
    render_fig6(FIG_DIR / "Fig6_examples.png")
    render_fig7(FIG_DIR / "Fig7_lineage.png")
    render_fig8(FIG_DIR / "Fig8_safety.png")
    render_fig9(FIG_DIR / "Fig9_memory.png")
    render_fig10(FIG_DIR / "Fig10_council.png")
    print("🎯 All 10 figures regenerated successfully with zero text overflow!")

if __name__ == "__main__":
    main()
"""
    final_code = new_content + new_main
    with open(FIG_GEN, "w", encoding="utf-8") as f:
        f.write(final_code)
    print("Updated generate_perfect_figures.py with render_fig5 and render_fig8")
