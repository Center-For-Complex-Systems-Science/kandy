#!/usr/bin/env python3
"""KANDy example: Intracranial EEG — Mode 0 Duffing-ReLU oscillator.

Discovers governing equations for seizure transition dynamics from real
intracranial EEG recordings using a threshold-activated Duffing oscillator
as the target model:

    x_ddot = mu*x - x^3 - alpha*y + beta*ReLU(x - theta)
             + gamma*sin(omega*t) + delta*cos(omega*t) + eta*x^2 - kappa*x*y

where x = SVD mode 0, y = dx/dt.

This script focuses ONLY on Mode 0 (the dominant SVD mode) and runs a
hyperparameter sweep over regularization (lamb) and grid size to find
the best KAN configuration for symbolic extraction.

Strategy: 2-jet embedding of alpha-band SVD mode 0
---------------------------------------------------
1. Bandpass 8-13 Hz (alpha band) on seizure channels [21-30]
2. Hilbert envelope -> 4-second running average
3. log(A + 1) to compress dynamic range
4. SVD -> mode 0 only (the dominant driving node)
5. Downsample to 5 Hz (~500 samples/episode, ~1500 total)
6. x = mode 0, y = x_dot (Savitzky-Golay)
7. Target: x_ddot (second derivative)
8. Since x_dot = y by construction, KANDy only learns the x_ddot equation

Lift design (Duffing-ReLU, 8 features)
---------------------------------------
1. x           -- linear restoring (mu*x)
2. y           -- damping (-alpha*y)
3. x^2         -- asymmetric nonlinearity (eta*x^2)
4. x^3         -- cubic saturation (-x^3)
5. x*y         -- nonlinear damping (-kappa*x*y)
6. ReLU(x-th)  -- threshold activation at seizure onset
7. sin(omega*t) -- periodic forcing
8. cos(omega*t) -- phase flexibility for forcing

KAN: [8, 1], Mode 0 only.

Data
----
E3Data.mat: X1 (49972,120), X2 (49971,120), X3 (49971,119)
fs = 500 Hz, seizure channels [21-30]
Onset times: Ep1=80.25s, Ep2=88.25s, Ep3=87.00s
"""

import os
import numpy as np
import torch
import scipy.io
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_edge_activation,
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)
from kandy.symbolic import auto_symbolic_with_costs, POLY_LIB_CHEAP, POLY_LIB

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "results/iEEG"
os.makedirs(OUT_DIR, exist_ok=True)

# Seizure onset times (from prior analysis)
ONSET_TIMES = {1: 80.25, 2: 88.25, 3: 87.00}  # seconds

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = "data/E3Data.mat"
print(f"[DATA] Loading {DATA_PATH} ...")
mat = scipy.io.loadmat(DATA_PATH)
X1 = mat["X1"].astype(np.float64)  # (49972, 120)
X2 = mat["X2"].astype(np.float64)  # (49971, 120)
X3 = mat["X3"].astype(np.float64)  # (49971, 119)
fs = 500.0
dt_raw = 1.0 / fs
print(f"[DATA] X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}")

# Seizure-onset zone channels (0-indexed)
SEIZURE_CHS = list(range(21, 31))  # channels 21-30
N_CH = len(SEIZURE_CHS)
print(f"[DATA] Seizure-zone channels: {SEIZURE_CHS} ({N_CH} channels)")

# ---------------------------------------------------------------------------
# 2. Alpha-band filtering + Hilbert envelope + moderate smoothing + log
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 1: Alpha-band envelope with moderate smoothing")
print(f"{'='*70}")

# Alpha band: 8-13 Hz
ALPHA_LOW = 8.0
ALPHA_HIGH = 13.0
b_alpha, a_alpha = butter(4, [ALPHA_LOW / (fs / 2), ALPHA_HIGH / (fs / 2)], btype="band")

# 4-second running average
SMOOTH_WIN_S = 4.0
SMOOTH_SAMPLES = int(SMOOTH_WIN_S * fs)  # 2000
smooth_kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES

print(f"[PREPROC] Alpha band: {ALPHA_LOW}-{ALPHA_HIGH} Hz")
print(f"[PREPROC] Smoothing: {SMOOTH_WIN_S}s = {SMOOTH_SAMPLES} samples")
print(f"[PREPROC] Effective bandwidth: < {1.0/SMOOTH_WIN_S:.2f} Hz")


def preprocess_episode(X_raw, channels, n_ch_available=None):
    """Alpha bandpass -> Hilbert envelope -> running avg -> log(A+1)."""
    amps_list = []
    used_channels = []
    for ch in channels:
        if n_ch_available is not None and ch >= n_ch_available:
            continue
        sig_alpha = filtfilt(b_alpha, a_alpha, X_raw[:, ch])
        analytic = hilbert(sig_alpha)
        inst_amp = np.abs(analytic)
        amp_smooth = np.convolve(inst_amp, smooth_kernel, mode="same")
        log_amp = np.log(amp_smooth + 1.0)
        amps_list.append(log_amp)
        used_channels.append(ch)
    return np.column_stack(amps_list), used_channels


log_amps_1, chs_1 = preprocess_episode(X1, SEIZURE_CHS)
log_amps_2, chs_2 = preprocess_episode(X2, SEIZURE_CHS)
log_amps_3, chs_3 = preprocess_episode(
    X3, SEIZURE_CHS, n_ch_available=X3.shape[1]
)

print(f"[PREPROC] Episode 1: {log_amps_1.shape}, chs {chs_1}")
print(f"[PREPROC] Episode 2: {log_amps_2.shape}, chs {chs_2}")
print(f"[PREPROC] Episode 3: {log_amps_3.shape}, chs {chs_3}")

# ---------------------------------------------------------------------------
# 3. SVD -> mode 0
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 2: SVD dimensionality reduction")
print(f"{'='*70}")

N_MODES = 3  # compute 3 but only use mode 0

# Common channels across all episodes
n_ch_common = min(log_amps_1.shape[1], log_amps_2.shape[1], log_amps_3.shape[1])
la1 = log_amps_1[:, :n_ch_common]
la2 = log_amps_2[:, :n_ch_common]
la3 = log_amps_3[:, :n_ch_common]

# Joint SVD across all episodes for consistent basis
all_amps = np.vstack([la1, la2, la3])
amp_mean = all_amps.mean(axis=0)
amp_std = all_amps.std(axis=0)
amp_std[amp_std < 1e-10] = 1.0
all_centered = (all_amps - amp_mean) / amp_std

U, S, Vt = np.linalg.svd(all_centered, full_matrices=False)
cumvar = np.cumsum(S**2) / np.sum(S**2) * 100

print(f"[SVD] Singular values (top 10): {S[:10].round(2)}")
print(f"[SVD] Cumulative variance: {cumvar[:N_MODES+2].round(1)}%")
print(f"[SVD] Using Mode 0, capturing {cumvar[0]:.1f}% variance")


def project_svd(data, mean, std, Vt, n_modes):
    centered = (data - mean) / std
    return centered @ Vt[:n_modes].T


modes_1 = project_svd(la1, amp_mean, amp_std, Vt, N_MODES)
modes_2 = project_svd(la2, amp_mean, amp_std, Vt, N_MODES)
modes_3 = project_svd(la3, amp_mean, amp_std, Vt, N_MODES)

print(f"[SVD] Mode shapes: ep1={modes_1.shape}, ep2={modes_2.shape}, ep3={modes_3.shape}")

# ---------------------------------------------------------------------------
# 4. Downsample to 5 Hz and compute first + second derivatives
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 3: Downsample to 5 Hz + compute derivatives")
print(f"{'='*70}")

DS = 100
dt_ds = dt_raw * DS  # 0.2 s
fs_ds = 1.0 / dt_ds  # 5 Hz

modes_1_ds = modes_1[::DS]
modes_2_ds = modes_2[::DS]
modes_3_ds = modes_3[::DS]

print(f"[DS] Downsampled {DS}x: {fs:.0f} Hz -> {fs_ds:.0f} Hz, dt={dt_ds:.3f}s")
print(f"[DS] Mode shapes: ep1={modes_1_ds.shape}, ep2={modes_2_ds.shape}, ep3={modes_3_ds.shape}")

# Compute first and second derivatives using Savitzky-Golay filter
SG_WINDOW = 13  # must be odd
SG_ORDER = 4    # polynomial order


def compute_derivatives(modes, dt, window=SG_WINDOW, polyorder=SG_ORDER):
    """Compute x_dot and x_ddot using Savitzky-Golay filter."""
    n_samples, n_cols = modes.shape
    x_dot = np.zeros_like(modes)
    x_ddot = np.zeros_like(modes)
    for j in range(n_cols):
        x_dot[:, j] = savgol_filter(modes[:, j], window, polyorder, deriv=1, delta=dt)
        x_ddot[:, j] = savgol_filter(modes[:, j], window, polyorder, deriv=2, delta=dt)
    return x_dot, x_ddot


xdot_1, xddot_1 = compute_derivatives(modes_1_ds, dt_ds)
xdot_2, xddot_2 = compute_derivatives(modes_2_ds, dt_ds)
xdot_3, xddot_3 = compute_derivatives(modes_3_ds, dt_ds)

print(f"[DERIV] Savitzky-Golay (window={SG_WINDOW}, polyorder={SG_ORDER}, dt={dt_ds}s)")

# Signal quality for Mode 0
for ep_idx, (modes_ds, xdot, xddot) in enumerate([
    (modes_1_ds, xdot_1, xddot_1),
    (modes_2_ds, xdot_2, xddot_2),
    (modes_3_ds, xdot_3, xddot_3),
], 1):
    sig_std = np.std(modes_ds[:, 0])
    dot_std = np.std(xdot[:, 0])
    ddot_std = np.std(xddot[:, 0])
    print(f"  Ep{ep_idx} mode 0: x_std={sig_std:.4f}, xdot_std={dot_std:.4f}, "
          f"xddot_std={ddot_std:.4f}, xdot/x={dot_std/max(sig_std,1e-12):.2f}")

# ---------------------------------------------------------------------------
# 5. Estimate theta (ReLU threshold) and omega (forcing frequency)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 4: Estimate ReLU threshold (theta) and forcing freq (omega)")
print(f"{'='*70}")

onset_vals = []
for ep_idx, (modes_ds, onset_s) in enumerate([
    (modes_1_ds, ONSET_TIMES[1]),
    (modes_2_ds, ONSET_TIMES[2]),
    (modes_3_ds, ONSET_TIMES[3]),
], 1):
    t_ep = np.arange(modes_ds.shape[0]) * dt_ds
    onset_idx = np.argmin(np.abs(t_ep - onset_s))
    onset_val = modes_ds[onset_idx, 0]
    onset_vals.append(onset_val)
    print(f"  Episode {ep_idx}: onset={onset_s}s, idx={onset_idx}, mode0 at onset={onset_val:.4f}")

THETA_EST = np.mean(onset_vals)
print(f"  Mean onset threshold (theta): {THETA_EST:.4f}")

# Omega: dominant frequency in pre-seizure signal
pre_seizure_modes = []
for modes_ds in [modes_1_ds, modes_2_ds, modes_3_ds]:
    t_ep = np.arange(modes_ds.shape[0]) * dt_ds
    pre_mask = t_ep < 70.0
    pre_seizure_modes.append(modes_ds[pre_mask, 0])

pre_sig = pre_seizure_modes[0]
n_fft = len(pre_sig)
freqs = np.fft.rfftfreq(n_fft, d=dt_ds)
fft_vals = np.abs(np.fft.rfft(pre_sig - pre_sig.mean()))

freq_mask = (freqs > 0.01) & (freqs < 2.0)
if np.any(freq_mask):
    peak_idx = np.argmax(fft_vals[freq_mask])
    peak_freq = freqs[freq_mask][peak_idx]
    OMEGA_EST = 2.0 * np.pi * peak_freq
    print(f"  Dominant pre-seizure frequency: {peak_freq:.4f} Hz")
    print(f"  Estimated omega: {OMEGA_EST:.4f} rad/s")
else:
    OMEGA_EST = 2.0 * np.pi * 0.1
    print(f"  No clear peak found, using fallback omega: {OMEGA_EST:.4f} rad/s")

# ---------------------------------------------------------------------------
# 6. Build training data: Mode 0 ONLY, 2-jet embedding
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 5: Build Mode 0 features (2-jet + Duffing-ReLU)")
print(f"{'='*70}")

TRIM = max(SG_WINDOW, 10)

FEATURE_NAMES = [
    "x", "y", "x^2", "x^3", "x*y",
    "ReLU(x-theta)", "sin(omega*t)", "cos(omega*t)",
]
N_FEAT = len(FEATURE_NAMES)


def build_mode0_features(modes_ds, xdot, xddot, dt, trim, theta, omega):
    """Build Duffing-ReLU feature matrix for Mode 0 only.

    Returns
    -------
    features : ndarray (N, 8) feature matrix
    targets : ndarray (N,) x_ddot targets
    t_arr : ndarray (N,) time values
    states : ndarray (N, 2) [x, y] state
    """
    T = modes_ds.shape[0]
    t_arr = np.arange(T) * dt
    valid = slice(trim, T - trim)

    x = modes_ds[valid, 0]
    y = xdot[valid, 0]
    xdd = xddot[valid, 0]
    t = t_arr[valid]

    features = np.column_stack([
        x,
        y,
        x**2,
        x**3,
        x * y,
        np.maximum(x - theta, 0.0),
        np.sin(omega * t),
        np.cos(omega * t),
    ])

    return features, xdd, t, np.column_stack([x, y])


# Build features for each episode (Mode 0 only)
feats_1, tgt_1, t_1, st_1 = build_mode0_features(
    modes_1_ds, xdot_1, xddot_1, dt_ds, TRIM, THETA_EST, OMEGA_EST
)
feats_2, tgt_2, t_2, st_2 = build_mode0_features(
    modes_2_ds, xdot_2, xddot_2, dt_ds, TRIM, THETA_EST, OMEGA_EST
)
feats_3, tgt_3, t_3, st_3 = build_mode0_features(
    modes_3_ds, xdot_3, xddot_3, dt_ds, TRIM, THETA_EST, OMEGA_EST
)

# Stack all episodes together
features_all = np.vstack([feats_1, feats_2, feats_3])
targets_all = np.concatenate([tgt_1, tgt_2, tgt_3])
t_all = np.concatenate([t_1, t_2, t_3])
states_all = np.vstack([st_1, st_2, st_3])

# Episode boundary indices
ep_boundaries = [0, len(tgt_1), len(tgt_1) + len(tgt_2), len(targets_all)]

N_total = len(targets_all)
print(f"[DATA] Total Mode 0 samples: {N_total}")
print(f"[DATA] Per-episode: Ep1={len(tgt_1)}, Ep2={len(tgt_2)}, Ep3={len(tgt_3)}")
print(f"[DATA] Features shape: {features_all.shape}")
print(f"[DATA] Feature names: {FEATURE_NAMES}")
print(f"[DATA] theta={THETA_EST:.4f}, omega={OMEGA_EST:.4f}")

# Feature statistics
print(f"\n  Feature statistics (Mode 0):")
for i, name in enumerate(FEATURE_NAMES):
    col = features_all[:, i]
    print(f"    {name:20s}: mean={col.mean():+.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

print(f"\n  Target (x_ddot) statistics:")
print(f"    mean={targets_all.mean():.6f}, std={targets_all.std():.6f}")
print(f"    min={targets_all.min():.6f}, max={targets_all.max():.6f}")

# Normalize features and targets
feat_mean = features_all.mean(axis=0)
feat_std = features_all.std(axis=0)
feat_std[feat_std < 1e-10] = 1.0
tgt_mean = targets_all.mean()
tgt_std = targets_all.std()
if tgt_std < 1e-10:
    tgt_std = 1.0

features_n = (features_all - feat_mean) / feat_std
targets_n = (targets_all - tgt_mean) / tgt_std

print(f"\n[NORM] Feature stds: {feat_std.round(4)}")
print(f"[NORM] Target std: {tgt_std:.6f}, mean: {tgt_mean:.6f}")

# ---------------------------------------------------------------------------
# 7. OLS baseline (determinism check) — Mode 0 only
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 6: OLS baseline (Mode 0)")
print(f"{'='*70}")

from numpy.linalg import lstsq

# OLS with constant feature
features_ols = np.column_stack([np.ones(N_total), features_all])
ols_feat_names = ["1"] + FEATURE_NAMES

coeffs_ols, _, _, _ = lstsq(features_ols, targets_all, rcond=None)
pred_ols = features_ols @ coeffs_ols
ss_res = np.sum((targets_all - pred_ols) ** 2)
ss_tot = np.sum((targets_all - targets_all.mean()) ** 2)
ols_r2 = 1 - ss_res / max(ss_tot, 1e-12)

print(f"\n  OLS (Mode 0, all episodes): R^2 = {ols_r2:.4f}")
print(f"  Equation: x_ddot = ", end="")
terms = []
for j, name in enumerate(ols_feat_names):
    c = coeffs_ols[j]
    if abs(c) > 1e-4:
        if name == "1":
            terms.append(f"{c:+.4f}")
        else:
            terms.append(f"{c:+.4f}*{name}")
print(" ".join(terms) if terms else "0")

# Per-episode OLS
print(f"\n  Per-episode OLS (Mode 0):")
for ep_idx, (start, end) in enumerate(
    zip(ep_boundaries[:-1], ep_boundaries[1:]), 1
):
    f_ep = features_ols[start:end]
    t_ep = targets_all[start:end]
    c_ep, _, _, _ = lstsq(f_ep, t_ep, rcond=None)
    p_ep = f_ep @ c_ep
    ss_r = np.sum((t_ep - p_ep) ** 2)
    ss_t = np.sum((t_ep - t_ep.mean()) ** 2)
    r2_ep = 1 - ss_r / max(ss_t, 1e-12)
    print(f"    Episode {ep_idx}: R^2 = {r2_ep:.4f} ({end-start} samples)")
    # Show top coefficients
    idx_sorted = np.argsort(np.abs(c_ep))[::-1]
    for jj in idx_sorted[:5]:
        if abs(c_ep[jj]) > 1e-3:
            print(f"      {ols_feat_names[jj]:25s}: {c_ep[jj]:+.6f}")

# Quality assessment
if ols_r2 > 0.3:
    print(f"\n  R^2 = {ols_r2:.4f} -- decent deterministic signal for Mode 0")
elif ols_r2 > 0.1:
    print(f"\n  R^2 = {ols_r2:.4f} -- weak but present deterministic signal (Mode 0)")
else:
    print(f"\n  R^2 = {ols_r2:.4f} -- very weak signal")

# ---------------------------------------------------------------------------
# 7b. LASSO sparsification — Mode 0 only
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 6b: LASSO sparsification (Mode 0)")
print(f"{'='*70}")

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, max_iter=10000, random_state=SEED)
lasso.fit(features_all, targets_all)
pred_lasso = lasso.predict(features_all)
ss_res_l = np.sum((targets_all - pred_lasso) ** 2)
ss_tot_l = np.sum((targets_all - targets_all.mean()) ** 2)
lasso_r2 = 1 - ss_res_l / max(ss_tot_l, 1e-12)

n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-6) + (1 if abs(lasso.intercept_) > 1e-6 else 0)
print(f"  LASSO: R^2={lasso_r2:.4f}, alpha={lasso.alpha_:.6f}, {n_nonzero} nonzero terms")
print(f"  Equation: x_ddot = ", end="")
lasso_terms = []
if abs(lasso.intercept_) > 1e-4:
    lasso_terms.append(f"{lasso.intercept_:+.6f}")
for j, name in enumerate(FEATURE_NAMES):
    c = lasso.coef_[j]
    if abs(c) > 1e-6:
        lasso_terms.append(f"{c:+.6f}*{name}")
print(" ".join(lasso_terms) if lasso_terms else "0")

# ---------------------------------------------------------------------------
# 8. HYPERPARAMETER SWEEP: lamb x grid for Mode 0 KANDy
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 7: Hyperparameter sweep (Mode 0 KANDy)")
print(f"{'='*70}")


class IdentityLiftWithNames(CustomLift):
    def __init__(self, n_feat, names):
        super().__init__(fn=lambda X: X, output_dim=n_feat, name="identity")
        self._names = names

    @property
    def feature_names(self):
        return self._names


# Sweep configurations
LAMB_VALUES = [0.0, 0.0001, 0.001, 0.005, 0.01]
GRID_VALUES = [3, 5, 7]
STEPS_SWEEP = 500
PATIENCE_SWEEP = 50
K_SPLINE = 3

sweep_results = []

targets_2d = targets_n[:, None]  # (N, 1)

print(f"  Sweep: lamb={LAMB_VALUES}, grid={GRID_VALUES}")
print(f"  Steps={STEPS_SWEEP}, patience={PATIENCE_SWEEP}, k={K_SPLINE}")
print(f"  Data: {N_total} samples (Mode 0, 3 episodes)")
print()

for grid_val in GRID_VALUES:
    for lamb_val in LAMB_VALUES:
        print(f"  --- grid={grid_val}, lamb={lamb_val} ---")

        # Reset seeds for fair comparison
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        lift_m = IdentityLiftWithNames(N_FEAT, FEATURE_NAMES)
        model_m = KANDy(
            lift=lift_m, grid=grid_val, k=K_SPLINE, steps=STEPS_SWEEP,
            seed=SEED, device="cpu",
        )
        model_m.fit(
            X=features_n, X_dot=targets_2d,
            val_frac=0.15, test_frac=0.10,
            lamb=lamb_val, patience=PATIENCE_SWEEP, verbose=False,
        )

        # One-step R^2 (un-normalized)
        pred_m = model_m.predict(features_n)
        if pred_m.ndim == 1:
            pred_m = pred_m[:, None]
        pred_un = pred_m[:, 0] * tgt_std + tgt_mean
        ss_r = np.sum((targets_all - pred_un) ** 2)
        ss_t = np.sum((targets_all - targets_all.mean()) ** 2)
        r2_m = 1 - ss_r / max(ss_t, 1e-12)

        # Count active edges
        sym_in_m = torch.tensor(
            features_n[:min(512, N_total)], dtype=torch.float32
        )
        model_m.model_.save_act = True
        with torch.no_grad():
            model_m.model_(sym_in_m)

        edge_ranges = []
        for i in range(N_FEAT):
            x_a, y_a = get_edge_activation(model_m.model_, l=0, i=i, j=0)
            rng = np.max(y_a) - np.min(y_a)
            edge_ranges.append(rng)

        max_range = max(edge_ranges)
        thresh = 0.05 * max_range if max_range > 1e-8 else 1e-8
        n_active = sum(1 for r in edge_ranges if r > thresh)

        # Training loss
        train_loss = model_m.train_results_["train_loss"][-1]

        result = {
            "grid": grid_val,
            "lamb": lamb_val,
            "r2": r2_m,
            "n_active": n_active,
            "train_loss": train_loss,
            "edge_ranges": edge_ranges,
            "model": model_m,
        }
        sweep_results.append(result)

        print(f"    R^2={r2_m:.4f}, active={n_active}/{N_FEAT}, "
              f"loss={train_loss:.6f}")
        # Show edge ranges compactly
        for i, name in enumerate(FEATURE_NAMES):
            status = "ON" if edge_ranges[i] > thresh else "off"
            print(f"      {name:20s}: {edge_ranges[i]:.6f} [{status}]")
        print()

# ---------------------------------------------------------------------------
# 9. Select best configurations: (A) highest R^2, (B) most active edges
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 8: Select best configuration")
print(f"{'='*70}")

valid_results = [r for r in sweep_results if r["r2"] > 0.0 and r["n_active"] > 0]

# Sort by R^2 for "best fit" selection
valid_by_r2 = sorted(valid_results, key=lambda r: r["r2"], reverse=True)
# Sort by most active edges first, then R^2 for tie-breaking
valid_by_edges = sorted(valid_results, key=lambda r: (r["n_active"], r["r2"]), reverse=True)

best_r2 = valid_by_r2[0] if valid_by_r2 else sweep_results[0]
best_struct = valid_by_edges[0] if valid_by_edges else sweep_results[0]

# Use structural model (most active edges) as primary, R^2 model as secondary
# Rationale: with R^2 ~ 0.1, more active edges = more structural info
model = best_struct["model"]
best = best_struct

print(f"  Best R^2:    grid={best_r2['grid']}, lamb={best_r2['lamb']}, "
      f"R^2={best_r2['r2']:.4f}, active={best_r2['n_active']}/{N_FEAT}")
print(f"  Best struct: grid={best_struct['grid']}, lamb={best_struct['lamb']}, "
      f"R^2={best_struct['r2']:.4f}, active={best_struct['n_active']}/{N_FEAT}")
print(f"  --> Using structural model (most active edges)")

# Print sweep summary table
print(f"\n  Full sweep results:")
print(f"  {'grid':>4s} {'lamb':>8s} {'R^2':>8s} {'active':>6s} {'loss':>10s}")
print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
for r in sweep_results:
    markers = []
    if r is best_r2:
        markers.append("R^2")
    if r is best_struct:
        markers.append("STRUCT")
    marker = f" <-- {'+'.join(markers)}" if markers else ""
    print(f"  {r['grid']:4d} {r['lamb']:8.4f} {r['r2']:8.4f} "
          f"{r['n_active']:6d} {r['train_loss']:10.6f}{marker}")

# ---------------------------------------------------------------------------
# 10. Detailed edge analysis BEFORE symbolic extraction
#     (auto_symbolic destructively replaces splines; must analyze first)
# ---------------------------------------------------------------------------


def analyze_edges(model_obj, features_norm, n_feat, feat_names, feat_mean_arr,
                  feat_std_arr, tgt_mean_val, tgt_std_val, label=""):
    """Analyze KAN edges: compute ranges, slopes, R^2, and extract equation.

    Returns dict with edge info and discovered equation string.
    """
    n_sym = min(512, len(features_norm))
    sym_input = torch.tensor(features_norm[:n_sym], dtype=torch.float32)
    model_obj.model_.save_act = True
    with torch.no_grad():
        model_obj.model_(sym_input)

    edges = []
    for i in range(n_feat):
        x_a, y_a = get_edge_activation(model_obj.model_, l=0, i=i, j=0)
        max_act = np.max(np.abs(y_a))
        range_act = np.max(y_a) - np.min(y_a)
        if np.std(y_a) > 1e-10:
            p = np.polyfit(x_a, y_a, 1)
            m_slope, b_int = p[0], p[1]
            y_lin = m_slope * x_a + b_int
            ss_res_e = np.sum((y_a - y_lin) ** 2)
            ss_tot_e = np.sum((y_a - y_a.mean()) ** 2)
            r2_lin = 1 - ss_res_e / max(ss_tot_e, 1e-12)
            # Quadratic fit
            p2 = np.polyfit(x_a, y_a, 2)
            y_quad = np.polyval(p2, x_a)
            ss_res_q = np.sum((y_a - y_quad) ** 2)
            r2_quad = 1 - ss_res_q / max(ss_tot_e, 1e-12)
        else:
            m_slope, b_int, r2_lin, r2_quad = 0.0, 0.0, 0.0, 0.0
        edges.append({
            "idx": i,
            "name": feat_names[i],
            "max_act": max_act,
            "range": range_act,
            "slope": m_slope,
            "intercept": b_int,
            "r2_linear": r2_lin,
            "r2_quadratic": r2_quad,
        })

    # Sort by activation range
    edges.sort(key=lambda e: e["range"], reverse=True)

    max_range = max(e["range"] for e in edges)
    thresh = 0.05 * max_range if max_range > 1e-8 else 1e-8
    n_active_e = sum(1 for e in edges if e["range"] > thresh)
    n_zeroed_e = n_feat - n_active_e

    if label:
        print(f"\n  --- {label} ---")
    print(f"  Active: {n_active_e}/{n_feat} (threshold = {thresh:.6f})")
    print()
    print(f"  {'Edge':25s} {'Range':>10s} {'Slope':>10s} {'R2_lin':>8s} {'R2_quad':>8s} {'Status':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for e in edges:
        status = "ACTIVE" if e["range"] > thresh else "zeroed"
        print(f"  {e['name']:25s} {e['range']:10.6f} {e['slope']:+10.4f} "
              f"{e['r2_linear']:8.4f} {e['r2_quadratic']:8.4f} {status:>8s}")

    # Extract equation from linear edge approximation
    # normalized -> original space
    eq_terms = []
    constant = tgt_mean_val
    coeff_dict = {}
    for e in edges:
        if e["range"] > thresh:
            i = e["idx"]
            coeff_orig = tgt_std_val * e["slope"] / feat_std_arr[i]
            constant -= tgt_std_val * e["slope"] * feat_mean_arr[i] / feat_std_arr[i]
            eq_terms.append(f"{coeff_orig:+.4f}*{e['name']}")
            coeff_dict[e["name"]] = coeff_orig
    if abs(constant) > 1e-4:
        eq_terms.insert(0, f"{constant:+.4f}")
        coeff_dict["const"] = constant

    eq_str = " ".join(eq_terms) if eq_terms else "0"
    print(f"\n  Discovered equation (linear edge approx, original space):")
    print(f"  x_ddot = {eq_str}")

    return {
        "edges": edges, "threshold": thresh, "n_active": n_active_e,
        "eq_str": eq_str, "coeff_dict": coeff_dict, "sym_input": sym_input,
    }


print(f"\n{'='*70}")
print(f"  STEP 9: Edge analysis (structural model)")
print(f"{'='*70}")

# Analyze the structural model (most active edges)
struct_analysis = analyze_edges(
    model, features_n, N_FEAT, FEATURE_NAMES,
    feat_mean, feat_std, tgt_mean, tgt_std,
    label=f"Structural model: grid={best_struct['grid']}, lamb={best_struct['lamb']}, "
          f"R^2={best_struct['r2']:.4f}",
)

active_edges = struct_analysis["edges"]
threshold = struct_analysis["threshold"]
n_active = struct_analysis["n_active"]
n_zeroed = N_FEAT - n_active
sym_in = struct_analysis["sym_input"]

# If structural and R^2 models differ, analyze R^2 model too
if best_struct is not best_r2:
    r2_analysis = analyze_edges(
        best_r2["model"], features_n, N_FEAT, FEATURE_NAMES,
        feat_mean, feat_std, tgt_mean, tgt_std,
        label=f"R^2 model: grid={best_r2['grid']}, lamb={best_r2['lamb']}, "
              f"R^2={best_r2['r2']:.4f}",
    )

# ---------------------------------------------------------------------------
# 12. One-step prediction accuracy
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 11: One-step prediction accuracy")
print(f"{'='*70}")

X_dot_pred_n = model.predict(features_n)
if X_dot_pred_n.ndim == 1:
    X_dot_pred_n = X_dot_pred_n[:, None]

X_dot_pred = X_dot_pred_n[:, 0] * tgt_std + tgt_mean

ss_res_k = np.sum((targets_all - X_dot_pred) ** 2)
ss_tot_k = np.sum((targets_all - targets_all.mean()) ** 2)
kandy_r2 = 1 - ss_res_k / max(ss_tot_k, 1e-12)

mse_k = np.mean((targets_all - X_dot_pred) ** 2)
rmse_k = np.sqrt(mse_k)

print(f"  KANDy R^2 = {kandy_r2:.4f} (OLS ceiling: {ols_r2:.4f})")
print(f"  KANDy RMSE = {rmse_k:.6f}")

# Per-episode R^2
print(f"\n  Per-episode R^2:")
for ep_idx, (start, end) in enumerate(
    zip(ep_boundaries[:-1], ep_boundaries[1:]), 1
):
    ss_r = np.sum((targets_all[start:end] - X_dot_pred[start:end]) ** 2)
    ss_t = np.sum((targets_all[start:end] - targets_all[start:end].mean()) ** 2)
    r2_ep = 1 - ss_r / max(ss_t, 1e-12)
    print(f"    Episode {ep_idx}: R^2 = {r2_ep:.4f}")

# ---------------------------------------------------------------------------
# 13. Rollout (2D system: x_dot=y, y_dot=KAN_predict)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 12: Autoregressive rollout (Ep1 Mode 0)")
print(f"{'='*70}")

CLIP_VAL = 10.0


def kandy_dynamics(state, t_val, model_, feat_mean_, feat_std_, tgt_mean_, tgt_std_,
                   theta, omega, clip=CLIP_VAL):
    """Full 2D dynamics: [x_dot, y_dot] = [y, KAN(features)]."""
    x, y = state
    x_clip = np.clip(x, -clip, clip)
    y_clip = np.clip(y, -clip, clip)

    feats = np.array([
        x_clip,
        y_clip,
        x_clip**2,
        x_clip**3,
        x_clip * y_clip,
        max(x_clip - theta, 0.0),
        np.sin(omega * t_val),
        np.cos(omega * t_val),
    ]).reshape(1, -1)

    feats_n = (feats - feat_mean_) / feat_std_

    with torch.no_grad():
        pred_n = model_.model_(torch.tensor(feats_n, dtype=torch.float32))
    xddot = pred_n.cpu().numpy().flatten()[0] * tgt_std_ + tgt_mean_
    xddot = np.clip(xddot, -50.0, 50.0)

    return np.array([y_clip, xddot])


# Rollout on Episode 1, Mode 0
T_ep1 = modes_1_ds.shape[0]
t_ep1 = np.arange(T_ep1) * dt_ds
x_true_ep1 = modes_1_ds[TRIM:T_ep1-TRIM, 0]
y_true_ep1 = xdot_1[TRIM:T_ep1-TRIM, 0]
xdd_true_ep1 = xddot_1[TRIM:T_ep1-TRIM, 0]
t_roll_ep1 = t_ep1[TRIM:T_ep1-TRIM]

N_ROLL = len(x_true_ep1)
x0_roll = np.array([x_true_ep1[0], y_true_ep1[0]])

# RK4 integration
traj_pred = [x0_roll.copy()]
state_curr = x0_roll.copy()
for step in range(N_ROLL - 1):
    t_curr = t_roll_ep1[step]
    t_next = t_roll_ep1[step + 1]
    h = t_next - t_curr

    k1 = kandy_dynamics(state_curr, t_curr, model, feat_mean, feat_std,
                        tgt_mean, tgt_std, THETA_EST, OMEGA_EST)
    k2 = kandy_dynamics(state_curr + 0.5*h*k1, t_curr + 0.5*h, model,
                        feat_mean, feat_std, tgt_mean, tgt_std, THETA_EST, OMEGA_EST)
    k3 = kandy_dynamics(state_curr + 0.5*h*k2, t_curr + 0.5*h, model,
                        feat_mean, feat_std, tgt_mean, tgt_std, THETA_EST, OMEGA_EST)
    k4 = kandy_dynamics(state_curr + h*k3, t_next, model,
                        feat_mean, feat_std, tgt_mean, tgt_std, THETA_EST, OMEGA_EST)
    state_curr = state_curr + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    state_curr = np.clip(state_curr, -CLIP_VAL, CLIP_VAL)
    traj_pred.append(state_curr.copy())

traj_pred = np.array(traj_pred)  # (N_ROLL, 2)
traj_true = np.column_stack([x_true_ep1, y_true_ep1])[:N_ROLL]

rollout_rmse = np.sqrt(np.mean((traj_pred - traj_true) ** 2))
rollout_rmse_x = np.sqrt(np.mean((traj_pred[:, 0] - traj_true[:, 0]) ** 2))
rollout_rmse_y = np.sqrt(np.mean((traj_pred[:, 1] - traj_true[:, 1]) ** 2))
nrmse = rollout_rmse / max(np.std(traj_true), 1e-12)

print(f"  Steps: {N_ROLL}, Duration: {N_ROLL * dt_ds:.1f}s")
print(f"  RMSE (total) = {rollout_rmse:.4f}")
print(f"  RMSE (x)     = {rollout_rmse_x:.4f}")
print(f"  RMSE (y)     = {rollout_rmse_y:.4f}")
print(f"  NRMSE        = {nrmse:.4f}")
print(f"  Max |pred|:  {np.max(np.abs(traj_pred)):.2f}")
bounded = np.all(np.isfinite(traj_pred)) and np.max(np.abs(traj_pred)) < 100
print(f"  Bounded: {'YES' if bounded else 'NO'}")

# OLS rollout for comparison
print(f"\n  OLS Rollout (Episode 1, Mode 0):")


def ols_dynamics(state, t_val, A_ols, theta, omega, clip=CLIP_VAL):
    x, y = state
    x_clip = np.clip(x, -clip, clip)
    y_clip = np.clip(y, -clip, clip)
    feats = np.array([
        1.0,
        x_clip, y_clip, x_clip**2, x_clip**3, x_clip*y_clip,
        max(x_clip - theta, 0.0),
        np.sin(omega * t_val), np.cos(omega * t_val),
    ])
    xddot = feats @ A_ols
    xddot = np.clip(xddot, -50.0, 50.0)
    return np.array([y_clip, xddot])


traj_ols = [x0_roll.copy()]
state_ols = x0_roll.copy()
for step in range(N_ROLL - 1):
    t_curr = t_roll_ep1[step]
    t_next = t_roll_ep1[step + 1]
    h = t_next - t_curr
    k1 = ols_dynamics(state_ols, t_curr, coeffs_ols, THETA_EST, OMEGA_EST)
    k2 = ols_dynamics(state_ols + 0.5*h*k1, t_curr + 0.5*h, coeffs_ols, THETA_EST, OMEGA_EST)
    k3 = ols_dynamics(state_ols + 0.5*h*k2, t_curr + 0.5*h, coeffs_ols, THETA_EST, OMEGA_EST)
    k4 = ols_dynamics(state_ols + h*k3, t_next, coeffs_ols, THETA_EST, OMEGA_EST)
    state_ols = state_ols + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    state_ols = np.clip(state_ols, -CLIP_VAL, CLIP_VAL)
    traj_ols.append(state_ols.copy())

traj_ols = np.array(traj_ols)
ols_rollout_rmse = np.sqrt(np.mean((traj_ols - traj_true) ** 2))
ols_rollout_rmse_x = np.sqrt(np.mean((traj_ols[:, 0] - traj_true[:, 0]) ** 2))
ols_nrmse = ols_rollout_rmse / max(np.std(traj_true), 1e-12)

print(f"  RMSE (total) = {ols_rollout_rmse:.4f}")
print(f"  RMSE (x)     = {ols_rollout_rmse_x:.4f}")
print(f"  NRMSE        = {ols_nrmse:.4f}")
print(f"  Bounded: {'YES' if np.all(np.isfinite(traj_ols)) and np.max(np.abs(traj_ols)) < 100 else 'NO'}")

# ---------------------------------------------------------------------------
# 14. Plots
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  STEP 13: Publication-quality plots")
print(f"{'='*70}")

use_pub_style()

# 14a. Data overview: Mode 0 across all episodes
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for ep_idx, (modes_ds, xdot_ep, onset_s) in enumerate([
    (modes_1_ds, xdot_1, ONSET_TIMES[1]),
    (modes_2_ds, xdot_2, ONSET_TIMES[2]),
    (modes_3_ds, xdot_3, ONSET_TIMES[3]),
]):
    t_ep = np.arange(modes_ds.shape[0]) * dt_ds
    ax = axes[ep_idx]
    ax.plot(t_ep, modes_ds[:, 0], "steelblue", lw=0.8, alpha=0.9, label="x (mode 0)")
    ax.plot(t_ep, xdot_ep[:, 0], "coral", lw=0.6, alpha=0.7, label="y = x_dot")
    ax.axvline(onset_s, color="red", ls="--", lw=1.0, alpha=0.7, label=f"Onset {onset_s:.1f}s")
    ax.axhline(THETA_EST, color="orange", ls=":", lw=0.7, alpha=0.5, label=f"theta={THETA_EST:.2f}")
    ax.set_ylabel(f"Episode {ep_idx+1}")
    ax.legend(fontsize=7, loc="upper left")
    if ep_idx == 2:
        ax.set_xlabel("Time (s)")
fig.suptitle(
    f"Mode 0: Alpha-band ({ALPHA_LOW}-{ALPHA_HIGH} Hz, {SMOOTH_WIN_S}s avg, {fs_ds} Hz)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/data_overview.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/data_overview.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/data_overview.png/pdf")

# 14b. Phase portrait and target scatter (Mode 0 only)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# Phase portrait
axes[0].scatter(features_all[:, 0], features_all[:, 1], s=3, alpha=0.4, c="steelblue")
axes[0].set_xlabel("x (mode 0)")
axes[0].set_ylabel("y = x_dot")
axes[0].set_title("Phase portrait")
# x vs x_ddot
axes[1].scatter(features_all[:, 0], targets_all, s=3, alpha=0.4, c="coral")
axes[1].set_xlabel("x (mode 0)")
axes[1].set_ylabel("x_ddot")
axes[1].set_title("x vs target")
# y vs x_ddot
axes[2].scatter(features_all[:, 1], targets_all, s=3, alpha=0.4, c="forestgreen")
axes[2].set_xlabel("y = x_dot")
axes[2].set_ylabel("x_ddot")
axes[2].set_title("y vs target")
fig.suptitle("Mode 0: 2-jet embedding diagnostics", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/derivative_quality.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/derivative_quality.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/derivative_quality.png/pdf")

# 14c. Edge activations for best model
fig_edges, axes_edges = plot_all_edges(
    model.model_,
    X=sym_in,
    fits=["linear"],
    in_var_names=FEATURE_NAMES,
    out_var_names=["x_ddot"],
    figsize_per_panel=(3.5, 2.5),
    save=f"{OUT_DIR}/edge_activations",
)
plt.close(fig_edges)
print(f"[PLOT] Saved {OUT_DIR}/edge_activations.png/pdf")

# 14d. Detailed per-edge plots (individual subplots with more detail)
fig, axes_detail = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes_detail.flat):
    if i >= N_FEAT:
        ax.set_visible(False)
        continue
    x_a, y_a = get_edge_activation(model.model_, l=0, i=i, j=0)
    rng = np.max(y_a) - np.min(y_a)
    is_active = rng > threshold

    # Sort for clean plotting
    order = np.argsort(x_a)
    x_sorted = x_a[order]
    y_sorted = y_a[order]

    ax.plot(x_sorted, y_sorted, "steelblue", lw=1.5, alpha=0.9)

    # Overlay linear fit
    if np.std(y_a) > 1e-10:
        m_slope, b_int = np.polyfit(x_a, y_a, 1)
        ax.plot(x_sorted, m_slope * x_sorted + b_int, "r--", lw=0.8, alpha=0.6)

    color = "black" if is_active else "gray"
    status_str = f"ACTIVE (range={rng:.4f})" if is_active else f"zeroed (range={rng:.4f})"
    ax.set_title(f"{FEATURE_NAMES[i]}\n{status_str}", fontsize=9, color=color)
    ax.set_xlabel(f"input ({FEATURE_NAMES[i]})", fontsize=8)
    ax.set_ylabel("spline output", fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle(f"Mode 0 KAN edges (grid={best['grid']}, lamb={best['lamb']}, R^2={kandy_r2:.4f})", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/edge_activations_detail.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/edge_activations_detail.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/edge_activations_detail.png/pdf")

# 14e. Loss curves
fig_loss, ax_loss = plot_loss_curves(
    model.train_results_,
    save=f"{OUT_DIR}/loss_curves",
)
plt.close(fig_loss)
print(f"[PLOT] Saved {OUT_DIR}/loss_curves.png/pdf")

# 14f. One-step prediction scatter
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
# KANDy
axes[0].scatter(targets_all, X_dot_pred, s=3, alpha=0.3, c="steelblue")
lims = [min(targets_all.min(), X_dot_pred.min()), max(targets_all.max(), X_dot_pred.max())]
axes[0].plot(lims, lims, "k--", lw=0.7, alpha=0.5)
axes[0].set_xlabel("True x_ddot")
axes[0].set_ylabel("Predicted x_ddot")
axes[0].set_title(f"KANDy one-step (R^2={kandy_r2:.4f})")
# OLS
pred_ols_all = features_ols @ coeffs_ols
axes[1].scatter(targets_all, pred_ols_all, s=3, alpha=0.3, c="coral")
axes[1].plot(lims, lims, "k--", lw=0.7, alpha=0.5)
axes[1].set_xlabel("True x_ddot")
axes[1].set_ylabel("Predicted x_ddot")
axes[1].set_title(f"OLS one-step (R^2={ols_r2:.4f})")
fig.suptitle("One-step prediction: KANDy vs OLS (Mode 0)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/onestep.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/onestep.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/onestep.png/pdf")

# 14g. Rollout comparison (Episode 1, Mode 0)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# x trajectory
axes[0].plot(t_roll_ep1[:N_ROLL], traj_true[:, 0], "k-", lw=1.5, label="True", alpha=0.8)
axes[0].plot(t_roll_ep1[:N_ROLL], traj_pred[:, 0], "steelblue", lw=1.2,
             ls="--", label=f"KANDy (RMSE={rollout_rmse_x:.3f})")
axes[0].plot(t_roll_ep1[:N_ROLL], traj_ols[:, 0], "coral", lw=1.0,
             ls=":", label=f"OLS (RMSE={ols_rollout_rmse_x:.3f})")
axes[0].axvline(ONSET_TIMES[1], color="red", ls="--", lw=0.7, alpha=0.5, label="Onset")
axes[0].set_ylabel("x (mode 0)")
axes[0].legend(fontsize=8)
axes[0].set_title("Rollout: Episode 1, Mode 0")

# y trajectory
axes[1].plot(t_roll_ep1[:N_ROLL], traj_true[:, 1], "k-", lw=1.5, label="True", alpha=0.8)
axes[1].plot(t_roll_ep1[:N_ROLL], traj_pred[:, 1], "steelblue", lw=1.2,
             ls="--", label="KANDy")
axes[1].plot(t_roll_ep1[:N_ROLL], traj_ols[:, 1], "coral", lw=1.0,
             ls=":", label="OLS")
axes[1].axvline(ONSET_TIMES[1], color="red", ls="--", lw=0.7, alpha=0.5)
axes[1].set_ylabel("y (x_dot)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/rollout.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/rollout.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/rollout.png/pdf")

# 14h. Phase portrait overlay (x vs y)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot(traj_true[:, 0], traj_true[:, 1], "k-", lw=1.0, alpha=0.4, label="True")
ax.plot(traj_pred[:, 0], traj_pred[:, 1], "steelblue", lw=1.0, alpha=0.8, label="KANDy")
ax.plot(traj_ols[:, 0], traj_ols[:, 1], "coral", lw=0.8, alpha=0.7, label="OLS")
ax.scatter([traj_true[0, 0]], [traj_true[0, 1]], s=50, c="green", zorder=5, label="IC")
ax.axvline(THETA_EST, color="orange", ls=":", lw=0.7, alpha=0.5, label=f"theta={THETA_EST:.2f}")
ax.set_xlabel("x (mode 0)")
ax.set_ylabel("y (x_dot)")
ax.set_title("Phase portrait: Episode 1, Mode 0")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/phase_portrait.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/phase_portrait.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/phase_portrait.png/pdf")

# 14i. ReLU activation check
fig, ax = plt.subplots(figsize=(6, 5))
x_m = features_all[:, 0]
relu_m = features_all[:, 5]
xdd_m = targets_all
sc = ax.scatter(x_m, xdd_m, c=relu_m, s=5, alpha=0.5, cmap="YlOrRd")
ax.axvline(THETA_EST, color="orange", ls="--", lw=1.0, label=f"theta={THETA_EST:.2f}")
ax.set_xlabel("x (mode 0)")
ax.set_ylabel("x_ddot")
ax.set_title("Mode 0: color=ReLU(x-theta)")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="ReLU(x-theta)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/relu_activation.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/relu_activation.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/relu_activation.png/pdf")

# 14j. Hyperparameter sweep heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R^2 heatmap
r2_grid = np.zeros((len(GRID_VALUES), len(LAMB_VALUES)))
active_grid = np.zeros((len(GRID_VALUES), len(LAMB_VALUES)))
for r in sweep_results:
    gi = GRID_VALUES.index(r["grid"])
    li = LAMB_VALUES.index(r["lamb"])
    r2_grid[gi, li] = r["r2"]
    active_grid[gi, li] = r["n_active"]

im1 = axes[0].imshow(r2_grid, aspect="auto", cmap="RdYlGn", origin="lower")
axes[0].set_xticks(range(len(LAMB_VALUES)))
axes[0].set_xticklabels([f"{l}" for l in LAMB_VALUES], fontsize=8, rotation=45)
axes[0].set_yticks(range(len(GRID_VALUES)))
axes[0].set_yticklabels([f"grid={g}" for g in GRID_VALUES])
axes[0].set_xlabel("lamb")
axes[0].set_title("R^2")
plt.colorbar(im1, ax=axes[0])
# Annotate
for gi in range(len(GRID_VALUES)):
    for li in range(len(LAMB_VALUES)):
        axes[0].text(li, gi, f"{r2_grid[gi, li]:.3f}", ha="center", va="center", fontsize=7)

im2 = axes[1].imshow(active_grid, aspect="auto", cmap="Blues", origin="lower")
axes[1].set_xticks(range(len(LAMB_VALUES)))
axes[1].set_xticklabels([f"{l}" for l in LAMB_VALUES], fontsize=8, rotation=45)
axes[1].set_yticks(range(len(GRID_VALUES)))
axes[1].set_yticklabels([f"grid={g}" for g in GRID_VALUES])
axes[1].set_xlabel("lamb")
axes[1].set_title("Active edges (out of 8)")
plt.colorbar(im2, ax=axes[1])
for gi in range(len(GRID_VALUES)):
    for li in range(len(LAMB_VALUES)):
        axes[1].text(li, gi, f"{int(active_grid[gi, li])}", ha="center", va="center", fontsize=8)

fig.suptitle("Hyperparameter sweep: Mode 0 KANDy", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/hyperparameter_sweep.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/hyperparameter_sweep.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/hyperparameter_sweep.png/pdf")

# 14k. SVD spectrum
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].semilogy(S[:20], "ko-", markersize=4)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Singular value")
axes[0].set_title("SVD spectrum")
axes[0].axhline(S[0], color="r", ls="--", alpha=0.5, label=f"Mode 0: {S[0]:.1f}")
axes[0].legend(fontsize=8)

axes[1].plot(cumvar[:20], "ko-", markersize=4)
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance (%)")
axes[1].set_title("Cumulative explained variance")
axes[1].axhline(cumvar[0], color="r", ls="--", alpha=0.5,
                label=f"Mode 0: {cumvar[0]:.1f}%")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/svd_spectrum.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/svd_spectrum.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] Saved {OUT_DIR}/svd_spectrum.png/pdf")

# ---------------------------------------------------------------------------
# 15. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  SUMMARY: Mode 0 Duffing-ReLU Oscillator for iEEG")
print(f"{'='*70}")
print(f"""
  Target model:
    x_ddot = mu*x - x^3 - alpha*y + beta*ReLU(x-theta)
             + gamma*sin(omega*t) + delta*cos(omega*t) + eta*x^2 - kappa*x*y

  Configuration:
    Data:         {N_total} Mode 0 samples (3 episodes, {fs_ds} Hz)
    Preprocessing: Alpha {ALPHA_LOW}-{ALPHA_HIGH} Hz, Hilbert, {SMOOTH_WIN_S}s avg, log(A+1)
    SVD mode 0:   {cumvar[0]:.1f}% variance
    Embedding:    2-jet (x, y=x_dot) -> target x_ddot
    Lift:         {FEATURE_NAMES}
    Best KAN:     [{N_FEAT}, 1], grid={best['grid']}, k={K_SPLINE}
    Best lamb:    {best['lamb']}
    theta (ReLU): {THETA_EST:.4f}
    omega:        {OMEGA_EST:.4f} rad/s

  Results (Mode 0 only):
    OLS R^2:      {ols_r2:.4f}
    LASSO R^2:    {lasso_r2:.4f} ({n_nonzero} terms)
    KANDy R^2:    {kandy_r2:.4f}
    Active edges: {n_active}/{N_FEAT}

  Rollout (Ep1, Mode 0):
    KANDy RMSE:   {rollout_rmse:.4f} (NRMSE={nrmse:.4f})
    OLS RMSE:     {ols_rollout_rmse:.4f} (NRMSE={ols_nrmse:.4f})

  Output:         {OUT_DIR}/
""")

print(f"\n[DONE] All results saved to {OUT_DIR}/")
