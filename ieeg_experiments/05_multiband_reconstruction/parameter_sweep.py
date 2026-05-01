#!/usr/bin/env python3
"""Parameter sweep of the discovered multiband Kuramoto model.

Simulates the discovered equations from all 5 frequency bands, sweeping
coherence parameters (r₀, r₂, r₃) to find:
  - Phase locking / synchronization transitions
  - Frequency entrainment
  - Seizure-like dynamics (coherence ramp)

Discovered equations (from multiband_kuramoto.py):

  DELTA (1-4 Hz):
    dθ₀/dt = 13.28 + 28.16*r₀ + 9.92*r₃ - 20.02
    dθ₂/dt = 14.85 - 5.12*r₂ + 25.88*r₃ - 14.04
    dθ₃/dt = 13.17 - 0.12

  THETA (4-8 Hz):
    dθ₀/dt = 34.70 - 16.94*r₀ + 7.34
    dθ₂/dt = 35.75 - 0.51
    dθ₃/dt = 35.00 - 0.32

  ALPHA (8-13 Hz):
    dθ₀/dt = 63.68 + 2.14*r₀*sin(θ₀-θ_mean) - 21.17*r₃ + 14.36
    dθ₂/dt = 63.42 + 12.10*r₀ - 18.38*r₃ + 7.60
    dθ₃/dt = 63.88 - 0.43

  BETA (13-30 Hz):
    dθ₀/dt = 109.99 + 41.79*r₃ - 29.68
    dθ₂/dt = 112.09 + 38.92*r₃ - 26.91
    dθ₃/dt = 111.38 - 1.33

  LOW GAMMA (30-50 Hz):
    dθ₀/dt = 233.85 + 3.53*r₂*sin(θ₂-θ_mean) + 0.84
    dθ₂/dt = 230.19 + 41.35*r₀ - 17.11
    dθ₃/dt = 229.65 - 0.22

Author: KANDy Researcher Agent
Date: 2026-03-31
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BAND_NAMES = ["delta", "theta", "alpha", "beta", "low_gamma"]
BAND_LABELS = [r"$\delta$ (1-4)", r"$\theta$ (4-8)", r"$\alpha$ (8-13)",
               r"$\beta$ (13-30)", r"$\gamma_L$ (30-50)"]
CLUSTER_NAMES = ["C0 (SOZ)", "C2 (pace)", "C3 (bound)"]
CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

# State: 15 phases (3 clusters × 5 bands)
# Index: band * 3 + cluster_idx  (cluster_idx: 0=C0, 1=C2, 2=C3)
N_BANDS = 5
N_CL = 3
N_STATE = N_BANDS * N_CL  # 15


def multiband_rhs(t, theta, r0, r2, r3):
    """RHS of the full multiband Kuramoto system.

    theta: (15,) = [delta_C0, delta_C2, delta_C3, theta_C0, ..., gamma_C3]
    r0, r2, r3: coherence parameters (can be scalar or functions of t)
    """
    r0_val = r0(t) if callable(r0) else r0
    r2_val = r2(t) if callable(r2) else r2
    r3_val = r3(t) if callable(r3) else r3

    dtheta = np.zeros(N_STATE)

    # --- DELTA (1-4 Hz): indices 0, 1, 2 ---
    dtheta[0] = 13.28 + 28.16 * r0_val + 9.92 * r3_val - 20.02
    dtheta[1] = 14.85 - 5.12 * r2_val + 25.88 * r3_val - 14.04
    dtheta[2] = 13.17 - 0.12

    # --- THETA (4-8 Hz): indices 3, 4, 5 ---
    dtheta[3] = 34.70 - 16.94 * r0_val + 7.34
    dtheta[4] = 35.75 - 0.51
    dtheta[5] = 35.00 - 0.32

    # --- ALPHA (8-13 Hz): indices 6, 7, 8 ---
    th_alpha = theta[6:9]
    th_alpha_mean = th_alpha.mean()
    dtheta[6] = 63.68 + 2.14 * r0_val * np.sin(th_alpha[0] - th_alpha_mean) \
                - 21.17 * r3_val + 14.36
    dtheta[7] = 63.42 + 12.10 * r0_val - 18.38 * r3_val + 7.60
    dtheta[8] = 63.88 - 0.43

    # --- BETA (13-30 Hz): indices 9, 10, 11 ---
    dtheta[9] = 109.99 + 41.79 * r3_val - 29.68
    dtheta[10] = 112.09 + 38.92 * r3_val - 26.91
    dtheta[11] = 111.38 - 1.33

    # --- LOW GAMMA (30-50 Hz): indices 12, 13, 14 ---
    th_gamma = theta[12:15]
    th_gamma_mean = th_gamma.mean()
    dtheta[12] = 233.85 + 3.53 * r2_val * np.sin(th_gamma[1] - th_gamma_mean) + 0.84
    dtheta[13] = 230.19 + 41.35 * r0_val - 17.11
    dtheta[14] = 229.65 - 0.22

    return dtheta


def simulate(r0, r2, r3, T_end=30.0, dt=0.001, theta0=None):
    """Simulate the multiband system. Returns t, theta (T, 15)."""
    if theta0 is None:
        theta0 = np.random.uniform(0, 2 * np.pi, N_STATE)
    t_eval = np.arange(0, T_end, dt)
    sol = solve_ivp(
        lambda t, y: multiband_rhs(t, y, r0, r2, r3),
        [0, T_end], theta0, t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    return sol.t, sol.y.T


def compute_freq_diff(theta, dt, band_idx):
    """Compute instantaneous frequency differences within a band."""
    s = band_idx * N_CL
    # Instantaneous frequencies
    omega = np.diff(theta[:, s:s+N_CL], axis=0) / dt
    # Pairwise frequency differences
    dw_01 = omega[:, 0] - omega[:, 1]  # C0 - C2
    dw_02 = omega[:, 0] - omega[:, 2]  # C0 - C3
    dw_12 = omega[:, 1] - omega[:, 2]  # C2 - C3
    return dw_01, dw_02, dw_12


def compute_phase_coherence(theta, band_idx):
    """Compute instantaneous phase coherence (Kuramoto r) within a band."""
    s = band_idx * N_CL
    z = np.exp(1j * theta[:, s:s+N_CL]).mean(axis=1)
    return np.abs(z)


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Pre-seizure coherence values (from data)
# ============================================================
R0_PRE = 0.44   # C0 SOZ core mean r (pre-seizure)
R2_PRE = 0.86   # C2 pacemaker mean r
R3_PRE = 0.72   # C3 boundary mean r

R0_SEZ = 0.75   # estimated seizure values (higher coherence)
R2_SEZ = 0.95
R3_SEZ = 0.95

np.random.seed(42)
theta0 = np.random.uniform(0, 2 * np.pi, N_STATE)

print("=" * 70)
print("PARAMETER SWEEP: MULTIBAND KURAMOTO MODEL")
print("=" * 70)


# ============================================================
# 1. SEIZURE SIMULATION: coherence ramp
# ============================================================
print("\n1. Seizure simulation (coherence ramp)...")

T_TOTAL = 40.0  # seconds
ONSET = 20.0     # seizure onset at 20s
RAMP = 3.0       # 3s ramp from pre to seizure coherence


def ramp(t, pre, sez, onset=ONSET, ramp_dur=RAMP):
    if t < onset:
        return pre
    elif t < onset + ramp_dur:
        frac = (t - onset) / ramp_dur
        return pre + frac * (sez - pre)
    else:
        return sez


r0_fn = lambda t: ramp(t, R0_PRE, R0_SEZ)
r2_fn = lambda t: ramp(t, R2_PRE, R2_SEZ)
r3_fn = lambda t: ramp(t, R3_PRE, R3_SEZ)

t_sez, theta_sez = simulate(r0_fn, r2_fn, r3_fn, T_end=T_TOTAL, theta0=theta0)
dt_sez = t_sez[1] - t_sez[0]

# Plot: per-band frequency traces + phase coherence
fig = plt.figure(figsize=(16, 18))
gs = gridspec.GridSpec(N_BANDS + 1, 2, hspace=0.45, wspace=0.3,
                       height_ratios=[1] * N_BANDS + [0.8])

for bi in range(N_BANDS):
    s = bi * N_CL
    # Instantaneous frequency (smoothed)
    omega = np.diff(theta_sez[:, s:s+N_CL], axis=0) / dt_sez / (2 * np.pi)
    # Smooth with running average
    win = min(500, len(omega) // 10)
    kernel = np.ones(win) / win

    # Left: frequencies
    ax = fig.add_subplot(gs[bi, 0])
    for ci in range(N_CL):
        freq_smooth = np.convolve(omega[:, ci], kernel, mode="same")
        ax.plot(t_sez[1:], freq_smooth, color=CLUSTER_COLORS[ci], lw=0.8,
                label=CLUSTER_NAMES[ci] if bi == 0 else None)
    ax.axvline(ONSET, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(ONSET + RAMP, color="k", ls=":", lw=0.5, alpha=0.4)
    ax.set_ylabel(f"{BAND_LABELS[bi]} Hz")
    if bi == 0:
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Instantaneous Frequency", fontsize=11)
    if bi == N_BANDS - 1:
        ax.set_xlabel("Time (s)")

    # Right: phase differences (mod 2π)
    ax = fig.add_subplot(gs[bi, 1])
    for ci in range(N_CL):
        for cj in range(ci + 1, N_CL):
            diff = np.mod(theta_sez[:, s + ci] - theta_sez[:, s + cj] + np.pi,
                         2 * np.pi) - np.pi
            diff_smooth = np.convolve(diff, kernel, mode="same")
            pair_name = f"{CLUSTER_NAMES[ci].split()[0]}-{CLUSTER_NAMES[cj].split()[0]}"
            ax.plot(t_sez, diff_smooth, lw=0.8, label=pair_name if bi == 0 else None)
    ax.axvline(ONSET, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel(r"$\Delta\theta$ (rad)")
    if bi == 0:
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Phase Differences", fontsize=11)
    if bi == N_BANDS - 1:
        ax.set_xlabel("Time (s)")

# Bottom: coherence ramp
ax = fig.add_subplot(gs[N_BANDS, :])
t_plot = np.linspace(0, T_TOTAL, 1000)
ax.plot(t_plot, [r0_fn(t) for t in t_plot], color=CLUSTER_COLORS[0], lw=1.5,
        label=f"r₀ (SOZ)")
ax.plot(t_plot, [r2_fn(t) for t in t_plot], color=CLUSTER_COLORS[1], lw=1.5,
        label=f"r₂ (pace)")
ax.plot(t_plot, [r3_fn(t) for t in t_plot], color=CLUSTER_COLORS[2], lw=1.5,
        label=f"r₃ (bound)")
ax.axvline(ONSET, color="k", ls="--", lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Coherence r")
ax.set_title("Coherence Ramp (seizure onset)")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

fig.suptitle("Seizure Simulation: Multiband Kuramoto Model\n"
             "Coherence ramp at t=20s", fontsize=14, y=0.98)
save_fig(fig, "sweep_seizure_simulation")


# ============================================================
# 2. r₃ SWEEP: boundary coherence drives SOZ + pacemaker
# ============================================================
print("\n2. r₃ sweep (boundary coherence)...")

r3_values = np.linspace(0.0, 1.0, 50)
T_SIM = 10.0

# Track: mean frequency per cluster per band at each r3
freq_vs_r3 = np.zeros((len(r3_values), N_BANDS, N_CL))

for ri, r3_val in enumerate(r3_values):
    t_sim, theta_sim = simulate(R0_PRE, R2_PRE, r3_val, T_end=T_SIM, theta0=theta0)
    dt_sim = t_sim[1] - t_sim[0]
    # Use last 50% for steady-state frequency
    half = len(t_sim) // 2
    for bi in range(N_BANDS):
        s = bi * N_CL
        for ci in range(N_CL):
            omega = np.diff(theta_sim[half:, s + ci]) / dt_sim / (2 * np.pi)
            freq_vs_r3[ri, bi, ci] = omega.mean()

fig, axes = plt.subplots(1, N_BANDS, figsize=(18, 4), sharey=False)
fig.suptitle(r"Frequency vs $r_3$ (boundary coherence), $r_0$=0.44, $r_2$=0.86",
             fontsize=13)
for bi in range(N_BANDS):
    ax = axes[bi]
    for ci in range(N_CL):
        ax.plot(r3_values, freq_vs_r3[:, bi, ci], color=CLUSTER_COLORS[ci],
                lw=1.5, label=CLUSTER_NAMES[ci])
    ax.set_xlabel(r"$r_3$")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(BAND_LABELS[bi], fontsize=10)
    if bi == 0:
        ax.legend(fontsize=7)
    # Mark where frequencies converge (locking)
    freq_spread = freq_vs_r3[:, bi, :].max(axis=1) - freq_vs_r3[:, bi, :].min(axis=1)
    lock_idx = np.where(freq_spread < 0.1)[0]
    if len(lock_idx) > 0:
        ax.axvspan(r3_values[lock_idx[0]], r3_values[lock_idx[-1]],
                  alpha=0.15, color="gold", label="Freq locked" if bi == 0 else None)
fig.tight_layout()
save_fig(fig, "sweep_r3_frequency")


# ============================================================
# 3. r₀ SWEEP: SOZ core coherence
# ============================================================
print("\n3. r₀ sweep (SOZ core coherence)...")

r0_values = np.linspace(0.0, 1.0, 50)
freq_vs_r0 = np.zeros((len(r0_values), N_BANDS, N_CL))

for ri, r0_val in enumerate(r0_values):
    t_sim, theta_sim = simulate(r0_val, R2_PRE, R3_PRE, T_end=T_SIM, theta0=theta0)
    dt_sim = t_sim[1] - t_sim[0]
    half = len(t_sim) // 2
    for bi in range(N_BANDS):
        s = bi * N_CL
        for ci in range(N_CL):
            omega = np.diff(theta_sim[half:, s + ci]) / dt_sim / (2 * np.pi)
            freq_vs_r0[ri, bi, ci] = omega.mean()

fig, axes = plt.subplots(1, N_BANDS, figsize=(18, 4), sharey=False)
fig.suptitle(r"Frequency vs $r_0$ (SOZ coherence), $r_2$=0.86, $r_3$=0.72",
             fontsize=13)
for bi in range(N_BANDS):
    ax = axes[bi]
    for ci in range(N_CL):
        ax.plot(r0_values, freq_vs_r0[:, bi, ci], color=CLUSTER_COLORS[ci],
                lw=1.5, label=CLUSTER_NAMES[ci])
    ax.set_xlabel(r"$r_0$")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(BAND_LABELS[bi], fontsize=10)
    if bi == 0:
        ax.legend(fontsize=7)
    freq_spread = freq_vs_r0[:, bi, :].max(axis=1) - freq_vs_r0[:, bi, :].min(axis=1)
    lock_idx = np.where(freq_spread < 0.1)[0]
    if len(lock_idx) > 0:
        ax.axvspan(r0_values[lock_idx[0]], r0_values[lock_idx[-1]],
                  alpha=0.15, color="gold")
fig.tight_layout()
save_fig(fig, "sweep_r0_frequency")


# ============================================================
# 4. 2D SWEEP: r₀ vs r₃ — frequency spread heatmap
# ============================================================
print("\n4. 2D sweep (r₀ vs r₃)...")

N_GRID = 30
r0_grid = np.linspace(0.0, 1.0, N_GRID)
r3_grid = np.linspace(0.0, 1.0, N_GRID)
freq_spread_2d = np.zeros((N_GRID, N_GRID, N_BANDS))

for i, r0_val in enumerate(r0_grid):
    for j, r3_val in enumerate(r3_grid):
        t_sim, theta_sim = simulate(r0_val, R2_PRE, r3_val, T_end=5.0, theta0=theta0)
        dt_sim = t_sim[1] - t_sim[0]
        half = len(t_sim) // 2
        for bi in range(N_BANDS):
            s = bi * N_CL
            freqs = []
            for ci in range(N_CL):
                omega = np.diff(theta_sim[half:, s + ci]) / dt_sim / (2 * np.pi)
                freqs.append(omega.mean())
            freq_spread_2d[i, j, bi] = max(freqs) - min(freqs)

fig, axes = plt.subplots(1, N_BANDS, figsize=(20, 4))
fig.suptitle(r"Frequency Spread (max-min across clusters) vs $r_0$, $r_3$"
             f"\nYellow = synchronized, Dark = detuned", fontsize=13)
for bi in range(N_BANDS):
    ax = axes[bi]
    im = ax.imshow(freq_spread_2d[:, :, bi].T, origin="lower",
                   extent=[0, 1, 0, 1], aspect="equal", cmap="viridis_r",
                   vmin=0, vmax=max(freq_spread_2d[:, :, bi].max(), 0.5))
    ax.set_xlabel(r"$r_0$")
    if bi == 0:
        ax.set_ylabel(r"$r_3$")
    ax.set_title(BAND_LABELS[bi], fontsize=10)
    # Mark pre-seizure and seizure points
    ax.plot(R0_PRE, R3_PRE, "wo", ms=8, mew=2, label="Pre-sez")
    ax.plot(R0_SEZ, R3_SEZ, "r*", ms=12, mew=1.5, label="Seizure")
    if bi == 0:
        ax.legend(fontsize=7, loc="lower left")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Hz spread")
fig.tight_layout()
save_fig(fig, "sweep_2d_r0_r3")


# ============================================================
# 5. VIRTUAL SURGERY: remove C0 (SOZ resection)
# ============================================================
print("\n5. Virtual surgery (remove C0)...")

# Normal seizure vs resection
fig, axes = plt.subplots(2, N_BANDS, figsize=(20, 8), sharex=True)
fig.suptitle("Virtual Surgery: Normal Seizure vs SOZ Resection\n"
             "(Top: intact, Bottom: C0 removed — set r₀=0)", fontsize=13)

for row, (label, r0_surgery) in enumerate([
    ("Intact", r0_fn),
    ("C0 Resected (r₀=0)", lambda t: 0.0),
]):
    t_sim, theta_sim = simulate(r0_surgery, r2_fn, r3_fn,
                                T_end=T_TOTAL, theta0=theta0)
    dt_sim = t_sim[1] - t_sim[0]
    win = min(500, len(t_sim) // 10)
    kernel = np.ones(win) / win

    for bi in range(N_BANDS):
        ax = axes[row, bi]
        s = bi * N_CL
        omega = np.diff(theta_sim[:, s:s+N_CL], axis=0) / dt_sim / (2 * np.pi)
        for ci in range(N_CL):
            freq_smooth = np.convolve(omega[:, ci], kernel, mode="same")
            ax.plot(t_sim[1:], freq_smooth, color=CLUSTER_COLORS[ci], lw=0.8,
                    label=CLUSTER_NAMES[ci] if bi == 0 and row == 0 else None)
        ax.axvline(ONSET, color="k", ls="--", lw=0.8, alpha=0.6)
        if row == 0:
            ax.set_title(BAND_LABELS[bi], fontsize=10)
        if bi == 0:
            ax.set_ylabel(f"{label}\nFreq (Hz)", fontsize=9)
        if row == 1:
            ax.set_xlabel("Time (s)")

axes[0, 0].legend(fontsize=7)
fig.tight_layout()
save_fig(fig, "sweep_virtual_surgery")


# ============================================================
# 6. BIFURCATION: track frequency locking as r₃ ramps slowly
# ============================================================
print("\n6. Slow r₃ ramp bifurcation diagram...")

T_RAMP = 100.0  # slow ramp over 100s
r3_slow = lambda t: t / T_RAMP  # 0 → 1 linearly
t_bif, theta_bif = simulate(R0_PRE, R2_PRE, r3_slow, T_end=T_RAMP,
                             dt=0.002, theta0=theta0)
dt_bif = t_bif[1] - t_bif[0]

fig, axes = plt.subplots(N_BANDS, 1, figsize=(12, 14), sharex=True)
fig.suptitle(r"Bifurcation Diagram: $r_3$ ramp 0→1 over 100s"
             "\nFrequency of each cluster vs time (= vs r₃)", fontsize=13)

for bi in range(N_BANDS):
    ax = axes[bi]
    s = bi * N_CL
    omega = np.diff(theta_bif[:, s:s+N_CL], axis=0) / dt_bif / (2 * np.pi)
    # Heavy smoothing
    win = 2000
    kernel = np.ones(win) / win
    for ci in range(N_CL):
        freq_smooth = np.convolve(omega[:, ci], kernel, mode="same")
        ax.plot(t_bif[1:], freq_smooth, color=CLUSTER_COLORS[ci], lw=1.0,
                label=CLUSTER_NAMES[ci] if bi == 0 else None)
    ax.set_ylabel(f"{BAND_LABELS[bi]}\nFreq (Hz)")
    # Add r₃ axis on top
    if bi == 0:
        ax.legend(fontsize=7, loc="upper left")
        ax2 = ax.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel(r"$r_3$")

axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "sweep_bifurcation_r3")


# ============================================================
# 7. CROSS-FREQUENCY PHASE COHERENCE during seizure
# ============================================================
print("\n7. Cross-frequency phase coherence during seizure...")

# Compute inter-band phase coherence during seizure simulation
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Cross-Band Phase Coherence During Seizure\n"
             "(Kuramoto r across all 5 bands, per cluster)", fontsize=13)

win = 500
kernel = np.ones(win) / win

for ci in range(N_CL):
    # Wrap phases to [0, 2π] and compute cross-band coherence
    phases_ci = np.zeros((len(t_sez), N_BANDS))
    for bi in range(N_BANDS):
        phases_ci[:, bi] = np.mod(theta_sez[:, bi * N_CL + ci], 2 * np.pi)

    # Cross-band Kuramoto r for this cluster
    z = np.exp(1j * phases_ci).mean(axis=1)
    r_cross = np.abs(z)
    r_smooth = np.convolve(r_cross, kernel, mode="same")
    ax.plot(t_sez, r_smooth, color=CLUSTER_COLORS[ci], lw=1.5,
            label=CLUSTER_NAMES[ci])

ax.axvline(ONSET, color="k", ls="--", lw=0.8, label="Onset")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cross-band r")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
save_fig(fig, "sweep_cross_frequency_coherence")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("PARAMETER SWEEP COMPLETE")
print("=" * 70)
print(f"\nPlots saved to {OUT_DIR}/:")
print("  sweep_seizure_simulation.png  — full seizure with coherence ramp")
print("  sweep_r3_frequency.png        — frequency vs r₃ (1D)")
print("  sweep_r0_frequency.png        — frequency vs r₀ (1D)")
print("  sweep_2d_r0_r3.png            — frequency spread heatmap")
print("  sweep_virtual_surgery.png     — intact vs C0 resected")
print("  sweep_bifurcation_r3.png      — slow r₃ ramp bifurcation")
print("  sweep_cross_frequency_coherence.png — cross-band coherence")
print("\nDone.")
