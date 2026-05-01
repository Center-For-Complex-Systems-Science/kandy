#!/usr/bin/env python3
"""Virtual stimulation: inject current to prevent seizure.

Tests three stimulation strategies:
  1. Desynchronizing stimulation at C3 (boundary) — reduce r₃ by injecting
     at a slightly detuned frequency to break internal coherence
  2. Frequency-pinning stimulation at C0 (SOZ) — hold delta at pre-seizure
     frequency so the coherence ramp can't speed it up
  3. Anti-phase stimulation at C0 — inject at seizure frequency but anti-phase
     to cancel the pathological oscillation

In Kuramoto terms:
  dθ_k/dt += I_stim * sin(ω_stim * t - θ_k + φ_stim)

Compares: no stimulation vs each strategy, showing whether the seizure
is suppressed in the reconstructed iEEG.

Author: KANDy Researcher Agent
Date: 2026-04-10
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

FS = 500
ONSET_TIME = 88.25
SEED = 42
np.random.seed(SEED)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "low_gamma": (30, 50)}
BAND_NAMES = list(BANDS.keys())
BAND_LABELS = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma_L$"]
N_BANDS = 5
N_CL = 3
N_STATE = N_BANDS * N_CL
CL_NAMES = ["C0 (SOZ)", "C2 (pace)", "C3 (bound)"]
CL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

RAMP_DUR = 3.0
R0_PRE, R0_SEZ = 0.44, 0.75
R2_PRE, R2_SEZ = 0.86, 0.95
R3_PRE, R3_SEZ = 0.72, 0.95

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def sigmoid_ramp(t, pre, sez, onset=ONSET_TIME, tau=RAMP_DUR):
    x = (t - onset) / (tau / 4)
    return pre + (sez - pre) / (1 + np.exp(-x))


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def multiband_rhs(t, theta, r0_fn, r2_fn, r3_fn, stim_fn=None):
    """Multiband Kuramoto RHS with optional stimulation."""
    r0 = r0_fn(t)
    r2 = r2_fn(t)
    r3 = r3_fn(t)

    dtheta = np.zeros(N_STATE)

    # DELTA
    dtheta[0] = 13.28 + 28.16 * r0 + 9.92 * r3 - 20.02
    dtheta[1] = 14.85 - 5.12 * r2 + 25.88 * r3 - 14.04
    dtheta[2] = 13.17 - 0.12

    # THETA
    dtheta[3] = 34.70 - 16.94 * r0 + 7.34
    dtheta[4] = 35.75 - 0.51
    dtheta[5] = 35.00 - 0.32

    # ALPHA
    th_a = theta[6:9]
    th_a_mean = th_a.mean()
    dtheta[6] = 63.68 + 2.14 * r0 * np.sin(th_a[0] - th_a_mean) \
                - 21.17 * r3 + 14.36
    dtheta[7] = 63.42 + 12.10 * r0 - 18.38 * r3 + 7.60
    dtheta[8] = 63.88 - 0.43

    # BETA
    dtheta[9] = 109.99 + 41.79 * r3 - 29.68
    dtheta[10] = 112.09 + 38.92 * r3 - 26.91
    dtheta[11] = 111.38 - 1.33

    # LOW GAMMA
    th_g = theta[12:15]
    th_g_mean = th_g.mean()
    dtheta[12] = 233.85 + 3.53 * r2 * np.sin(th_g[1] - th_g_mean) + 0.84
    dtheta[13] = 230.19 + 41.35 * r0 - 17.11
    dtheta[14] = 229.65 - 0.22

    # SYNCHRONIZATION (SOZ drives others)
    SYNC_STRENGTH = 25.0
    K_sync = sigmoid_ramp(t, 0.0, SYNC_STRENGTH,
                          onset=ONSET_TIME + 1.0, tau=2.0)
    if K_sync > 0.01:
        for bi in range(N_BANDS):
            s = bi * N_CL
            theta_soz = theta[s + 0]
            for ci in range(1, N_CL):
                dtheta[s + ci] += K_sync * np.sin(theta_soz - theta[s + ci])

    # STIMULATION
    if stim_fn is not None:
        stim = stim_fn(t, theta)
        dtheta += stim

    return dtheta


def simulate(r0_fn, r2_fn, r3_fn, stim_fn=None, T_start=75.0, T_end=98.0):
    theta0 = np.random.RandomState(SEED).uniform(0, 2*np.pi, N_STATE)
    dt = 1.0 / FS
    t_eval = np.arange(T_start, T_end, dt)
    sol = solve_ivp(
        lambda t, y: multiband_rhs(t, y, r0_fn, r2_fn, r3_fn, stim_fn),
        [T_start, T_end], theta0, t_eval=t_eval,
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    return sol.t, sol.y.T


def compute_band_freq(theta_sim, t_sim, band_idx, smooth_win=200):
    """Compute smoothed instantaneous frequencies for a band."""
    dt = t_sim[1] - t_sim[0]
    s = band_idx * N_CL
    omega = np.diff(theta_sim[:, s:s+N_CL], axis=0) / dt / (2*np.pi)
    kern = np.ones(smooth_win) / smooth_win
    freqs = np.zeros_like(omega)
    for ci in range(N_CL):
        freqs[:, ci] = np.convolve(omega[:, ci], kern, mode="same")
    return freqs


# ============================================================
# Define stimulation strategies
# ============================================================
print("=" * 70)
print("VIRTUAL STIMULATION: PREVENT SEIZURE")
print("=" * 70)

# Pre-seizure natural frequencies (from discovered equations at pre-seizure r values)
OMEGA_PRE = {
    "delta": [13.28 + 28.16*R0_PRE + 9.92*R3_PRE - 20.02,
              14.85 - 5.12*R2_PRE + 25.88*R3_PRE - 14.04,
              13.17 - 0.12],
    "theta": [34.70 - 16.94*R0_PRE + 7.34,
              35.75 - 0.51,
              35.00 - 0.32],
}

print("\nPre-seizure frequencies:")
for band in ["delta", "theta"]:
    print(f"  {band}: {[f'{w/(2*np.pi):.2f} Hz' for w in OMEGA_PRE[band]]}")

# Coherence functions
r0_normal = lambda t: sigmoid_ramp(t, R0_PRE, R0_SEZ)
r2_normal = lambda t: sigmoid_ramp(t, R2_PRE, R2_SEZ)
r3_normal = lambda t: sigmoid_ramp(t, R3_PRE, R3_SEZ)

# Strategy 1: Desynchronize C3 — clamp r₃ at pre-seizure level
# (equivalent to stimulating C3 channels at slightly different frequencies
#  to break their internal coherence)
r3_clamped = lambda t: R3_PRE  # r₃ never ramps up


# Strategy 2: Frequency-pin C0 delta — inject forcing at pre-seizure delta frequency
# dθ_0^delta/dt += I * sin(ω_pre * t - θ_0^delta)
STIM_AMPLITUDE = 30.0  # rad/s — must be strong enough to override the seizure drive
OMEGA_STIM_DELTA = OMEGA_PRE["delta"][0]  # pre-seizure delta frequency of C0


def stim_pin_c0_delta(t, theta):
    """Pin C0 delta to pre-seizure frequency."""
    stim = np.zeros(N_STATE)
    if t >= ONSET_TIME - 2.0:  # start stimulation 2s before expected onset
        stim[0] = STIM_AMPLITUDE * np.sin(OMEGA_STIM_DELTA * t - theta[0])
    return stim


# Strategy 3: Desync C3 + pin C0 delta (combined)
def stim_combined(t, theta):
    """Combined: pin C0 delta frequency."""
    stim = np.zeros(N_STATE)
    if t >= ONSET_TIME - 2.0:
        stim[0] = STIM_AMPLITUDE * np.sin(OMEGA_STIM_DELTA * t - theta[0])
    return stim


# Strategy 4: Low-amplitude wide-band desynchronizing pulse at SOZ
# Inject broadband noise into C0 phases to prevent coherent buildup
DESYNC_AMP = 20.0
rng_stim = np.random.RandomState(SEED + 300)


def stim_desync_soz(t, theta):
    """Broadband desynchronizing stimulation at C0 across all bands."""
    stim = np.zeros(N_STATE)
    if t >= ONSET_TIME - 2.0:
        for bi in range(N_BANDS):
            # Random phase kick — different for each band
            stim[bi * N_CL + 0] = DESYNC_AMP * np.sin(
                (50 + bi * 17.3) * t + bi * 1.7)  # incommensurate frequencies
    return stim


# ============================================================
# Run simulations
# ============================================================
scenarios = {
    "No stimulation\n(seizure)": {
        "r0": r0_normal, "r2": r2_normal, "r3": r3_normal, "stim": None,
    },
    "Clamp r₃\n(desync C3)": {
        "r0": r0_normal, "r2": r2_normal, "r3": r3_clamped, "stim": None,
    },
    "Pin C0 delta\n(frequency lock)": {
        "r0": r0_normal, "r2": r2_normal, "r3": r3_normal, "stim": stim_pin_c0_delta,
    },
    "Clamp r₃ +\npin C0 delta": {
        "r0": r0_normal, "r2": r2_normal, "r3": r3_clamped, "stim": stim_combined,
    },
    "Desync SOZ\n(broadband)": {
        "r0": r0_normal, "r2": r2_normal, "r3": r3_normal, "stim": stim_desync_soz,
    },
}

results = {}
for name, params in scenarios.items():
    print(f"\n  Simulating: {name.replace(chr(10), ' ')}...")
    t_sim, theta_sim = simulate(params["r0"], params["r2"], params["r3"],
                                 params["stim"])
    results[name] = (t_sim, theta_sim)
    # Print seizure-window delta frequency
    freqs = compute_band_freq(theta_sim, t_sim, 0)
    sez_mask = (t_sim[1:] >= ONSET_TIME + 2) & (t_sim[1:] <= ONSET_TIME + 8)
    if sez_mask.sum() > 0:
        for ci in range(N_CL):
            f_sez = freqs[sez_mask, ci].mean()
            print(f"    {CL_NAMES[ci]} delta: {f_sez:.2f} Hz during seizure")


# ============================================================
# Plot 1: Delta frequency comparison across scenarios
# ============================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3 * len(scenarios)),
                         sharex=True)
fig.suptitle("Virtual Stimulation: Delta Band Frequency\n"
             "Can we prevent the seizure frequency shift?", fontsize=14)

for ax_idx, (name, (t_sim, theta_sim)) in enumerate(results.items()):
    ax = axes[ax_idx]
    freqs = compute_band_freq(theta_sim, t_sim, 0, smooth_win=300)
    for ci in range(N_CL):
        ax.plot(t_sim[1:], freqs[:, ci], color=CL_COLORS[ci], lw=1.0,
                label=CL_NAMES[ci] if ax_idx == 0 else None)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Freq (Hz)")
    ax.set_title(name.replace("\n", " "), fontsize=10, loc="left")
    pre_f = freqs[(t_sim[1:] >= 80) & (t_sim[1:] < ONSET_TIME), 0].mean()
    sez_mask = (t_sim[1:] >= ONSET_TIME + 3) & (t_sim[1:] <= 95)
    if sez_mask.sum() > 0:
        sez_f = freqs[sez_mask, 0].mean()
        shift = sez_f - pre_f
        ax.text(0.98, 0.85, f"Δf = {shift:+.2f} Hz",
                transform=ax.transAxes, ha="right", fontsize=9,
                color="red" if abs(shift) > 0.3 else "green",
                fontweight="bold")
    if ax_idx == 0:
        ax.legend(fontsize=7, loc="upper left")

axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "stim_delta_frequency")


# ============================================================
# Plot 2: All bands frequency spread (synchronization measure)
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3 * len(scenarios)),
                         sharex=True)
fig.suptitle("Frequency Spread Across All Bands\n"
             "(lower = more synchronized = more seizure-like)", fontsize=14)

for ax_idx, (name, (t_sim, theta_sim)) in enumerate(results.items()):
    ax = axes[ax_idx]
    for bi in range(N_BANDS):
        freqs = compute_band_freq(theta_sim, t_sim, bi, smooth_win=300)
        spread = freqs.max(axis=1) - freqs.min(axis=1)
        kern = np.ones(500) / 500
        spread_smooth = np.convolve(spread, kern, mode="same")
        ax.plot(t_sim[1:], spread_smooth, lw=0.8, label=BAND_LABELS[bi] if ax_idx == 0 else None)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Spread (Hz)")
    ax.set_title(name.replace("\n", " "), fontsize=10, loc="left")
    if ax_idx == 0:
        ax.legend(fontsize=7, ncol=5, loc="upper right")

axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "stim_freq_spread")


# ============================================================
# Plot 3: Phase coherence within bands
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3 * len(scenarios)),
                         sharex=True)
fig.suptitle("Intra-Band Phase Coherence (Kuramoto r)\n"
             "(higher = more synchronized)", fontsize=14)

for ax_idx, (name, (t_sim, theta_sim)) in enumerate(results.items()):
    ax = axes[ax_idx]
    kern = np.ones(500) / 500
    for bi in range(N_BANDS):
        s = bi * N_CL
        z = np.exp(1j * theta_sim[:, s:s+N_CL]).mean(axis=1)
        r = np.abs(z)
        r_smooth = np.convolve(r, kern, mode="same")
        ax.plot(t_sim, r_smooth, lw=0.8, label=BAND_LABELS[bi] if ax_idx == 0 else None)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.8)
    ax.set_ylabel("r")
    ax.set_ylim(0, 1.05)
    ax.set_title(name.replace("\n", " "), fontsize=10, loc="left")
    if ax_idx == 0:
        ax.legend(fontsize=7, ncol=5, loc="upper right")

axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "stim_coherence")


# ============================================================
# Plot 4: Summary comparison — seizure severity metric
# ============================================================
# Seizure severity = mean frequency shift across all bands + mean coherence increase
fig, ax = plt.subplots(figsize=(12, 5))

scenario_names = []
severity_freq = []
severity_coh = []

for name, (t_sim, theta_sim) in results.items():
    scenario_names.append(name.replace("\n", " "))

    # Frequency shift: mean absolute delta freq change across clusters
    freqs_delta = compute_band_freq(theta_sim, t_sim, 0, smooth_win=300)
    pre_mask = (t_sim[1:] >= 80) & (t_sim[1:] < ONSET_TIME)
    sez_mask = (t_sim[1:] >= ONSET_TIME + 3) & (t_sim[1:] <= 95)
    if pre_mask.sum() > 0 and sez_mask.sum() > 0:
        f_pre = freqs_delta[pre_mask].mean(axis=0)
        f_sez = freqs_delta[sez_mask].mean(axis=0)
        severity_freq.append(np.mean(np.abs(f_sez - f_pre)))
    else:
        severity_freq.append(0)

    # Coherence increase: mean r increase across bands
    r_increases = []
    for bi in range(N_BANDS):
        s = bi * N_CL
        z = np.exp(1j * theta_sim[:, s:s+N_CL]).mean(axis=1)
        r = np.abs(z)
        pre_r = r[(t_sim >= 80) & (t_sim < ONSET_TIME)].mean()
        sez_r = r[(t_sim >= ONSET_TIME + 3) & (t_sim <= 95)].mean()
        r_increases.append(max(sez_r - pre_r, 0))
    severity_coh.append(np.mean(r_increases))

x = np.arange(len(scenario_names))
width = 0.35
bars1 = ax.bar(x - width/2, severity_freq, width, label="Freq shift (Hz)",
               color="tomato", edgecolor="k", lw=0.5)
bars2 = ax.bar(x + width/2, severity_coh, width, label="Coherence increase",
               color="steelblue", edgecolor="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, fontsize=8)
ax.set_ylabel("Seizure Severity")
ax.set_title("Stimulation Efficacy: Lower = Better Seizure Suppression", fontsize=12)
ax.legend(fontsize=9)

# Annotate reduction percentages
baseline_f = severity_freq[0]
baseline_c = severity_coh[0]
for i in range(1, len(scenario_names)):
    pct_f = (1 - severity_freq[i] / max(baseline_f, 1e-6)) * 100
    pct_c = (1 - severity_coh[i] / max(baseline_c, 1e-6)) * 100
    ax.text(i, max(severity_freq[i], severity_coh[i]) + 0.05,
            f"-{pct_f:.0f}%F / -{pct_c:.0f}%C",
            ha="center", fontsize=7, color="green" if pct_f > 50 else "orange")

fig.tight_layout()
save_fig(fig, "stim_efficacy_summary")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("STIMULATION RESULTS")
print("=" * 70)
for i, name in enumerate(scenario_names):
    pct_f = (1 - severity_freq[i] / max(severity_freq[0], 1e-6)) * 100 if i > 0 else 0
    pct_c = (1 - severity_coh[i] / max(severity_coh[0], 1e-6)) * 100 if i > 0 else 0
    print(f"  {name:30s}: freq_shift={severity_freq[i]:.3f} Hz, "
          f"coh_increase={severity_coh[i]:.3f}"
          f"{'  (baseline)' if i == 0 else f'  ({pct_f:+.0f}%F, {pct_c:+.0f}%C)'}")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
