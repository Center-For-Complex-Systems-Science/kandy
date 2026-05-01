#!/usr/bin/env python3
"""Publication schematic of the Multiband Cross-Frequency Coupled Kuramoto Model.

Shows:
  - 3 clusters with their channel counts and SOZ membership
  - 5 frequency bands per cluster (15 oscillators total)
  - Discovered coupling terms (from KANDy) with actual coefficients
  - Cross-frequency coupling (delta → beta/gamma PAC)
  - Directional synchronization (SOZ drives others)
  - Coherence modulation pathways (r₀, r₂, r₃)

Author: KANDy Researcher Agent
Date: 2026-04-10
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ============================================================
# Model parameters (all from the actual experiment)
# ============================================================
CLUSTERS = {
    "C0\nSOZ core": {"n_ch": 14, "soz_ch": [21, 28, 29, 30], "color": "#1f77b4",
                      "r_pre": 0.44, "r_sez": 0.75, "pos": (0.18, 0.55)},
    "C2\nPacemaker": {"n_ch": 3, "soz_ch": [23, 25, 26], "color": "#d62728",
                       "r_pre": 0.86, "r_sez": 0.95, "pos": (0.50, 0.55)},
    "C3\nBoundary":  {"n_ch": 8, "soz_ch": [22, 24, 27], "color": "#2ca02c",
                       "r_pre": 0.72, "r_sez": 0.95, "pos": (0.82, 0.55)},
}

BANDS = [
    (r"$\delta$ 1-4 Hz",   "#4e79a7"),
    (r"$\theta$ 4-8 Hz",   "#59a14f"),
    (r"$\alpha$ 8-13 Hz",  "#f28e2b"),
    (r"$\beta$ 13-30 Hz",  "#e15759"),
    (r"$\gamma_L$ 30-50 Hz", "#76b7b2"),
]

# Discovered equations (actual coefficients from KANDy)
EQUATIONS = {
    "delta": [
        r"$\dot{\theta}_0 = 13.28 + 28.16 r_0 + 9.92 r_3 - 20.02$",
        r"$\dot{\theta}_2 = 14.85 - 5.12 r_2 + 25.88 r_3 - 14.04$",
        r"$\dot{\theta}_3 = 13.05$",
    ],
    "theta": [
        r"$\dot{\theta}_0 = 42.04 - 16.94 r_0$",
        r"$\dot{\theta}_2 = 35.24$",
        r"$\dot{\theta}_3 = 34.68$",
    ],
    "alpha": [
        r"$\dot{\theta}_0 = 78.04 + 2.14 r_0 \sin(\theta_0 - \bar{\theta}) - 21.17 r_3$",
        r"$\dot{\theta}_2 = 71.02 + 12.10 r_0 - 18.38 r_3$",
        r"$\dot{\theta}_3 = 63.45$",
    ],
    "beta": [
        r"$\dot{\theta}_0 = 80.31 + 41.79 r_3$",
        r"$\dot{\theta}_2 = 85.18 + 38.92 r_3$",
        r"$\dot{\theta}_3 = 110.05$",
    ],
    "low_gamma": [
        r"$\dot{\theta}_0 = 234.69 + 3.53 r_2 \sin(\theta_2 - \bar{\theta})$",
        r"$\dot{\theta}_2 = 213.08 + 41.35 r_0$",
        r"$\dot{\theta}_3 = 229.43$",
    ],
}


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Figure 1: Model Architecture Schematic
# ============================================================
print("Generating model schematic...")

fig = plt.figure(figsize=(18, 22))

# ---- Top section: Cluster diagram with oscillator stacks ----
ax_top = fig.add_axes([0.02, 0.52, 0.96, 0.46])
ax_top.set_xlim(0, 1)
ax_top.set_ylim(0, 1)
ax_top.axis("off")

ax_top.text(0.5, 0.97, "Multiband Cross-Frequency Coupled Kuramoto Model",
            ha="center", va="top", fontsize=18, fontweight="bold",
            family="serif")
ax_top.text(0.5, 0.93,
            "15 phase oscillators (3 clusters × 5 bands) with coherence-modulated coupling",
            ha="center", va="top", fontsize=11, color="gray", family="serif")

# Draw each cluster as a column of 5 oscillator circles
band_colors = [b[1] for b in BANDS]
band_names = [b[0] for b in BANDS]

for cl_name, cl_info in CLUSTERS.items():
    cx, cy = cl_info["pos"]
    cl_color = cl_info["color"]

    # Cluster box
    box = FancyBboxPatch((cx - 0.12, cy - 0.42), 0.24, 0.82,
                         boxstyle="round,pad=0.02",
                         facecolor=cl_color, alpha=0.08,
                         edgecolor=cl_color, lw=2)
    ax_top.add_patch(box)

    # Cluster label
    ax_top.text(cx, cy + 0.42, cl_name, ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=cl_color, family="serif")

    # Channel info
    soz_str = ", ".join(str(c) for c in cl_info["soz_ch"])
    ax_top.text(cx, cy + 0.35,
                f"{cl_info['n_ch']} channels\nSOZ: [{soz_str}]",
                ha="center", va="bottom", fontsize=8, color="gray",
                family="serif", linespacing=1.3)

    # Coherence info
    ax_top.text(cx, cy - 0.43,
                f"$r$: {cl_info['r_pre']:.2f} → {cl_info['r_sez']:.2f}",
                ha="center", va="top", fontsize=9, color=cl_color,
                family="serif", style="italic")

    # 5 oscillator circles (one per band)
    for bi in range(5):
        oy = cy + 0.25 - bi * 0.14
        circle = plt.Circle((cx, oy), 0.035, facecolor=band_colors[bi],
                            edgecolor="k", lw=1.0, alpha=0.8, zorder=5)
        ax_top.add_patch(circle)
        # Phase symbol inside
        ax_top.text(cx, oy, f"$\\theta$", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=6)
        # Band label to the side (only for leftmost cluster)
        if cl_name.startswith("C0"):
            ax_top.text(cx - 0.14, oy, band_names[bi], ha="right", va="center",
                        fontsize=8, color=band_colors[bi], family="serif")

# ---- Coupling arrows between clusters ----

# r₃ → C0, C2 (delta, alpha, beta): strongest coupling
for cl_target, target_x in [("C0\nSOZ core", 0.18), ("C2\nPacemaker", 0.50)]:
    # Arrow from C3 to target
    ax_top.annotate("", xy=(target_x + 0.06, 0.55 + 0.25),
                    xytext=(0.82 - 0.06, 0.55 + 0.25),
                    arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                   lw=2.5, alpha=0.7,
                                   connectionstyle="arc3,rad=-0.15"))

ax_top.text(0.50, 0.87, "$r_3$ drives C0, C2\n(δ, α, β bands)",
            ha="center", va="center", fontsize=8, color="#2ca02c",
            family="serif", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ca02c", alpha=0.1))

# r₀ → C2 (alpha, gamma): SOZ coherence modulates pacemaker
ax_top.annotate("", xy=(0.50 - 0.06, 0.55 + 0.11),
                xytext=(0.18 + 0.06, 0.55 + 0.11),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                               lw=2.0, alpha=0.7,
                               connectionstyle="arc3,rad=0.2"))
ax_top.text(0.34, 0.72, "$r_0$ → C2\n(α, γ)", ha="center", va="center",
            fontsize=8, color="#1f77b4", family="serif", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1f77b4", alpha=0.1))

# Directional sync: SOZ → others (post-onset)
ax_top.annotate("", xy=(0.50 - 0.06, 0.55 - 0.05),
                xytext=(0.18 + 0.06, 0.55 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color="purple",
                               lw=1.5, alpha=0.5, ls="--",
                               connectionstyle="arc3,rad=-0.15"))
ax_top.annotate("", xy=(0.82 - 0.06, 0.55 - 0.10),
                xytext=(0.18 + 0.06, 0.55 - 0.10),
                arrowprops=dict(arrowstyle="-|>", color="purple",
                               lw=1.5, alpha=0.5, ls="--",
                               connectionstyle="arc3,rad=-0.1"))
ax_top.text(0.50, 0.42, "Directional sync\n(post-onset, K=25 rad/s)",
            ha="center", va="center", fontsize=8, color="purple",
            family="serif", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="purple", alpha=0.08))

# CFC arrow (vertical, within each cluster): delta → beta/gamma
for cx in [0.18, 0.50, 0.82]:
    ax_top.annotate("", xy=(cx + 0.05, 0.55 - 0.25),
                    xytext=(cx + 0.05, 0.55 + 0.25),
                    arrowprops=dict(arrowstyle="-|>", color="goldenrod",
                                   lw=1.5, alpha=0.6, ls=":",
                                   connectionstyle="arc3,rad=0.3"))
ax_top.text(0.93, 0.55, "PAC\nδ phase →\nβ/γ amplitude",
            ha="center", va="center", fontsize=8, color="goldenrod",
            family="serif", style="italic", rotation=0,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="goldenrod", alpha=0.1))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=band_colors[i], label=band_names[i], alpha=0.8)
    for i in range(5)
]
ax_top.legend(handles=legend_elements, loc="lower left", fontsize=8,
              title="Frequency Bands", title_fontsize=9, framealpha=0.9,
              bbox_to_anchor=(0.0, 0.0))


# ---- Bottom section: Discovered equations ----
ax_bot = fig.add_axes([0.03, 0.02, 0.94, 0.48])
ax_bot.set_xlim(0, 1)
ax_bot.set_ylim(0, 1)
ax_bot.axis("off")

ax_bot.text(0.5, 0.97, "Discovered Equations (KANDy)",
            ha="center", va="top", fontsize=16, fontweight="bold",
            family="serif")
ax_bot.text(0.5, 0.93,
            "Symbolic equations extracted from trained KAN models, per frequency band",
            ha="center", va="top", fontsize=10, color="gray", family="serif")

# Equations table
band_keys = ["delta", "theta", "alpha", "beta", "low_gamma"]
band_display = [r"$\delta$ (1-4 Hz)", r"$\theta$ (4-8 Hz)", r"$\alpha$ (8-13 Hz)",
                r"$\beta$ (13-30 Hz)", r"$\gamma_L$ (30-50 Hz)"]
cl_labels = ["C0 (SOZ core)", "C2 (Pacemaker)", "C3 (Boundary)"]

y_start = 0.88
row_h = 0.035
band_h = 0.04

for bi, band_key in enumerate(band_keys):
    y_band = y_start - bi * (3 * row_h + band_h + 0.02)

    # Band header
    ax_bot.add_patch(FancyBboxPatch((0.02, y_band - 0.01), 0.96, band_h,
                                     boxstyle="round,pad=0.005",
                                     facecolor=band_colors[bi], alpha=0.15,
                                     edgecolor=band_colors[bi], lw=1))
    ax_bot.text(0.04, y_band + band_h / 2 - 0.01, band_display[bi],
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=band_colors[bi], family="serif")

    # Equations for each cluster
    for ci in range(3):
        y_eq = y_band - (ci + 1) * row_h
        eq_text = EQUATIONS[band_key][ci]

        # Cluster label
        cl_color = list(CLUSTERS.values())[ci]["color"]
        ax_bot.text(0.06, y_eq, cl_labels[ci], ha="left", va="center",
                    fontsize=8, color=cl_color, family="serif",
                    fontweight="bold")

        # Equation
        ax_bot.text(0.25, y_eq, eq_text, ha="left", va="center",
                    fontsize=9, family="serif")

# Observation model box
y_obs = 0.02
ax_bot.add_patch(FancyBboxPatch((0.02, y_obs), 0.96, 0.08,
                                 boxstyle="round,pad=0.01",
                                 facecolor="gray", alpha=0.08,
                                 edgecolor="gray", lw=1))
ax_bot.text(0.04, y_obs + 0.06, "Observation Model:", ha="left", va="center",
            fontsize=10, fontweight="bold", color="gray", family="serif")
ax_bot.text(0.04, y_obs + 0.025,
            r"$x_{ch}(t) = \sum_b A_b(t) \cdot [1 + \eta_{OU}(t)] \cdot "
            r"[1 + \kappa_b(t) \cos(\theta_\delta - \pi)] \cdot "
            r"\cos(\theta^b_{k(ch)} + \delta\phi_{ch})$"
            r"$\quad + \quad DC_{shift}(t) \quad + \quad \mathrm{pink\ noise}$",
            ha="left", va="center", fontsize=9, family="serif")

save_fig(fig, "model_schematic")

# ============================================================
# Figure 2: Simpler overview for presentations
# ============================================================
print("Generating simple overview...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.axis("off")

ax.text(0.5, 1.02, "Multiband Kuramoto Seizure Model",
        ha="center", va="top", fontsize=18, fontweight="bold", family="serif")

# Three cluster nodes
node_r = 0.08
positions = {"C0": (0.2, 0.65), "C2": (0.5, 0.65), "C3": (0.8, 0.65)}
colors = {"C0": "#1f77b4", "C2": "#d62728", "C3": "#2ca02c"}
labels = {
    "C0": "C0\nSOZ core\n14 ch\nr: 0.44→0.75",
    "C2": "C2\nPacemaker\n3 ch\nr: 0.86→0.95",
    "C3": "C3\nBoundary\n8 ch\nr: 0.72→0.95",
}

for name, (x, y) in positions.items():
    circle = plt.Circle((x, y), node_r, facecolor=colors[name], alpha=0.2,
                        edgecolor=colors[name], lw=3, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, labels[name], ha="center", va="center",
            fontsize=9, fontweight="bold", color=colors[name],
            family="serif", linespacing=1.4, zorder=6)

# Coupling arrows with coefficients
# r₃ → C0 (delta: +9.92, alpha: -21.17, beta: +41.79)
ax.annotate("", xy=(0.2 + node_r, 0.65 + 0.02),
            xytext=(0.8 - node_r, 0.65 + 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=3, alpha=0.7,
                           connectionstyle="arc3,rad=-0.25"))
ax.text(0.50, 0.82, "$r_3$: +9.9 (δ), −21.2 (α), +41.8 (β)",
        ha="center", fontsize=8, color="#2ca02c", family="serif",
        bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.9, pad=3))

# r₃ → C2
ax.annotate("", xy=(0.5 + node_r * 0.7, 0.65 + node_r * 0.7),
            xytext=(0.8 - node_r * 0.7, 0.65 + node_r * 0.7),
            arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=2, alpha=0.5,
                           connectionstyle="arc3,rad=-0.3"))
ax.text(0.68, 0.78, "$r_3$: +25.9 (δ)\n−18.4 (α), +38.9 (β)",
        ha="center", fontsize=7, color="#2ca02c", family="serif",
        bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.8, pad=2))

# r₀ → C2
ax.annotate("", xy=(0.5 - node_r, 0.65 - 0.02),
            xytext=(0.2 + node_r, 0.65 - 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.5, alpha=0.7,
                           connectionstyle="arc3,rad=-0.2"))
ax.text(0.35, 0.52, "$r_0$: +12.1 (α), +41.4 (γ)",
        ha="center", fontsize=8, color="#1f77b4", family="serif",
        bbox=dict(facecolor="white", edgecolor="#1f77b4", alpha=0.9, pad=3))

# Directional sync (post-onset)
ax.annotate("", xy=(0.5 - node_r, 0.65 + 0.06),
            xytext=(0.2 + node_r, 0.65 + 0.06),
            arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.5,
                           alpha=0.4, ls="--"))
ax.annotate("", xy=(0.8 - node_r, 0.65 - 0.06),
            xytext=(0.2 + node_r, 0.65 - 0.06),
            arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.5,
                           alpha=0.4, ls="--"))
ax.text(0.2, 0.48, "Directional sync\n(SOZ leads post-onset)",
        ha="center", fontsize=8, color="purple", family="serif", style="italic")

# Bottom: 5 frequency bands as colored bar
band_w = 0.16
for bi, (bname, bcol) in enumerate(BANDS):
    bx = 0.1 + bi * (band_w + 0.02)
    rect = FancyBboxPatch((bx, 0.15), band_w, 0.12,
                           boxstyle="round,pad=0.01",
                           facecolor=bcol, alpha=0.3, edgecolor=bcol, lw=1.5)
    ax.add_patch(rect)
    ax.text(bx + band_w / 2, 0.21, bname, ha="center", va="center",
            fontsize=9, fontweight="bold", color=bcol, family="serif")

ax.text(0.5, 0.30, "× 3 clusters = 15 coupled phase oscillators",
        ha="center", fontsize=10, family="serif", color="gray")

# CFC label
ax.annotate("PAC: δ phase → β/γ amplitude",
            xy=(0.1 + band_w/2, 0.27), xytext=(0.1 + 3*(band_w+0.02) + band_w/2, 0.32),
            fontsize=8, color="goldenrod", family="serif",
            arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5,
                           connectionstyle="arc3,rad=0.3"))

# Key parameters box
ax.text(0.5, 0.05,
        "Onset spike (SOZ-first, staggered) → Coherence ramp → "
        "Frequency shift → Synchronization → Seizure",
        ha="center", fontsize=10, family="serif",
        bbox=dict(facecolor="lightyellow", edgecolor="orange",
                  alpha=0.8, pad=5, boxstyle="round,pad=0.5"))

save_fig(fig, "model_overview")

print("Done.")
