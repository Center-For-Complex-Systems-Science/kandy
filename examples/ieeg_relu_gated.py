#!/usr/bin/env python3
"""KANDy example: Intracranial EEG -- ReLU-gated Duffing oscillator with time forcing.

Extends the Mode 0 Duffing-ReLU model by allowing the ReLU to modulate MORE
of the dynamics (damping, position coupling) and introducing gated periodic
forcing that only activates at seizure onset:

    x_dot  = y
    x_ddot = mu*x - x^3 - alpha*y + eta*x^2 - kappa*x*y     (base Duffing)
             + beta1*ReLU(x-theta)                              (ReLU acceleration)
             + beta2*ReLU(x-theta)*y                            (ReLU-gated damping)
             + beta3*ReLU(x-theta)*x                            (ReLU-gated position)
             + gamma*sin(omega*t)*H(t-t_onset)                  (gated sin forcing)
             + delta*cos(omega*t)*H(t-t_onset)                  (gated cos forcing)
             + epsilon*H(t-t_onset)                             (DC shift at seizure)
             + zeta*ReLU(x-theta)^2                             (quadratic ReLU)

Lift design (12 features)
-------------------------
 0.  x              linear restoring (mu*x)
 1.  y              damping (-alpha*y)
 2.  x^2            asymmetric nonlinearity (eta*x^2)
 3.  x^3            cubic saturation (-x^3)
 4.  x*y            nonlinear damping (-kappa*x*y)
 5.  ReLU(x-theta)  threshold acceleration (beta1)
 6.  ReLU*y         ReLU-gated damping (beta2)   -- NEW
 7.  ReLU*x         ReLU-gated position (beta3)  -- NEW
 8.  sin(wt)*gate   gated sin forcing (gamma)    -- NEW
 9.  cos(wt)*gate   gated cos forcing (delta)    -- NEW
10.  gate           DC shift at seizure (epsilon) -- NEW
11.  ReLU^2         quadratic ReLU (zeta)

KAN: [12, 1], grid=3, k=3 (validated sweet spot for ~1400 samples)

Data
----
E3Data.mat: channels 21-30, alpha band 8-13 Hz, Hilbert envelope,
4s running average, log(A+1), joint SVD -> Mode 0 only,
downsample to 5 Hz (dt=0.2s), Savitzky-Golay derivatives.
"""

import os
import numpy as np
import torch
import scipy.io
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_edge_activation,
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ===========================================================================
# 0. Configuration
# ===========================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "results/iEEG/relu_gated"
os.makedirs(OUT_DIR, exist_ok=True)

# Seizure onset times (seconds)
ONSET_TIMES = {1: 80.25, 2: 88.25, 3: 87.00}

# Time gate ramp duration (seconds)
GATE_RAMP_S = 5.0

# ===========================================================================
# 1. Load data
# ===========================================================================
DATA_PATH = "data/E3Data.mat"
print(f"[DATA] Loading {DATA_PATH} ...")
mat = scipy.io.loadmat(DATA_PATH)
X1 = mat["X1"].astype(np.float64)
X2 = mat["X2"].astype(np.float64)
X3 = mat["X3"].astype(np.float64)
fs = 500.0
dt_raw = 1.0 / fs
print(f"[DATA] X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}")

SEIZURE_CHS = list(range(21, 31))  # channels 21-30
N_CH = len(SEIZURE_CHS)
print(f"[DATA] Seizure-zone channels: {SEIZURE_CHS} ({N_CH} channels)")

# ===========================================================================
# 2. Alpha-band filtering + Hilbert envelope + 4s smoothing + log
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 1: Alpha-band envelope with moderate smoothing")
print(f"{'='*70}")

ALPHA_LOW = 8.0
ALPHA_HIGH = 13.0
b_alpha, a_alpha = butter(4, [ALPHA_LOW / (fs / 2), ALPHA_HIGH / (fs / 2)], btype="band")

SMOOTH_WIN_S = 4.0
SMOOTH_SAMPLES = int(SMOOTH_WIN_S * fs)
smooth_kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES

print(f"[PREPROC] Alpha band: {ALPHA_LOW}-{ALPHA_HIGH} Hz")
print(f"[PREPROC] Smoothing: {SMOOTH_WIN_S}s = {SMOOTH_SAMPLES} samples")


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
log_amps_3, chs_3 = preprocess_episode(X3, SEIZURE_CHS, n_ch_available=X3.shape[1])

print(f"[PREPROC] Episode 1: {log_amps_1.shape}")
print(f"[PREPROC] Episode 2: {log_amps_2.shape}")
print(f"[PREPROC] Episode 3: {log_amps_3.shape}")

# ===========================================================================
# 3. SVD -> mode 0
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 2: SVD dimensionality reduction")
print(f"{'='*70}")

N_MODES = 3

n_ch_common = min(log_amps_1.shape[1], log_amps_2.shape[1], log_amps_3.shape[1])
la1 = log_amps_1[:, :n_ch_common]
la2 = log_amps_2[:, :n_ch_common]
la3 = log_amps_3[:, :n_ch_common]

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

# ===========================================================================
# 4. Downsample to 5 Hz and compute derivatives
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 3: Downsample to 5 Hz + compute derivatives")
print(f"{'='*70}")

DS = 100
dt_ds = dt_raw * DS  # 0.2 s
fs_ds = 1.0 / dt_ds  # 5 Hz

modes_1_ds = modes_1[::DS]
modes_2_ds = modes_2[::DS]
modes_3_ds = modes_3[::DS]

print(f"[DS] {fs:.0f} Hz -> {fs_ds:.0f} Hz, dt={dt_ds:.3f}s")
print(f"[DS] ep1={modes_1_ds.shape}, ep2={modes_2_ds.shape}, ep3={modes_3_ds.shape}")

SG_WINDOW = 13
SG_ORDER = 4


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

for ep_idx, (modes_ds, xdot, xddot) in enumerate([
    (modes_1_ds, xdot_1, xddot_1),
    (modes_2_ds, xdot_2, xddot_2),
    (modes_3_ds, xdot_3, xddot_3),
], 1):
    sig_std = np.std(modes_ds[:, 0])
    dot_std = np.std(xdot[:, 0])
    ddot_std = np.std(xddot[:, 0])
    print(f"  Ep{ep_idx} mode 0: x_std={sig_std:.4f}, xdot_std={dot_std:.4f}, "
          f"xddot_std={ddot_std:.4f}")

# ===========================================================================
# 5. Estimate theta (ReLU threshold) and omega (forcing frequency)
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 4: Estimate theta and omega")
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
    print(f"  Episode {ep_idx}: onset={onset_s}s, mode0 at onset={onset_val:.4f}")

THETA_EST = np.mean(onset_vals)
print(f"  Mean onset threshold (theta): {THETA_EST:.4f}")

# Omega from FFT of pre-seizure signal
pre_sig = modes_1_ds[:int(70.0 / dt_ds), 0]
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
    print(f"  No clear peak, fallback omega: {OMEGA_EST:.4f} rad/s")

# ===========================================================================
# 6. Build ReLU-gated features: 12-feature lift
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 5: Build ReLU-gated features (12-feature lift)")
print(f"{'='*70}")

TRIM = max(SG_WINDOW, 10)

FEATURE_NAMES = [
    "x", "y", "x^2", "x^3", "x*y",
    "ReLU", "ReLU*y", "ReLU*x",
    "sin(wt)*gate", "cos(wt)*gate", "gate", "ReLU^2",
]
N_FEAT = len(FEATURE_NAMES)


def build_features_gated(modes_ds, xdot, xddot, dt, trim, theta, omega,
                         onset_s, gate_ramp):
    """Build 12-feature ReLU-gated matrix for Mode 0.

    Parameters
    ----------
    modes_ds : array (T, n_modes)
    xdot, xddot : arrays (T, n_modes)
    dt : float
    trim : int -- edge samples to discard (SG boundary effects)
    theta : float -- ReLU threshold
    omega : float -- forcing angular frequency
    onset_s : float -- seizure onset time in seconds
    gate_ramp : float -- time (seconds) for gate to ramp 0->1

    Returns
    -------
    features : (N, 12) feature matrix
    targets : (N,) x_ddot targets
    t_arr : (N,) time values
    states : (N, 2) [x, y] state
    """
    T = modes_ds.shape[0]
    t_full = np.arange(T) * dt
    valid = slice(trim, T - trim)

    x = modes_ds[valid, 0]
    y = xdot[valid, 0]
    xdd = xddot[valid, 0]
    t = t_full[valid]

    relu_val = np.maximum(x - theta, 0.0)
    time_gate = np.clip((t - onset_s) / gate_ramp, 0.0, 1.0)

    features = np.column_stack([
        x,                                   # 0: x
        y,                                   # 1: y
        x**2,                                # 2: x^2
        x**3,                                # 3: x^3
        x * y,                               # 4: x*y
        relu_val,                            # 5: ReLU(x-theta)
        relu_val * y,                        # 6: ReLU*y
        relu_val * x,                        # 7: ReLU*x
        np.sin(omega * t) * time_gate,       # 8: sin(wt)*gate
        np.cos(omega * t) * time_gate,       # 9: cos(wt)*gate
        time_gate,                           # 10: gate
        relu_val**2,                         # 11: ReLU^2
    ])

    return features, xdd, t, np.column_stack([x, y])


# Build per episode
feats_1, tgt_1, t_1, st_1 = build_features_gated(
    modes_1_ds, xdot_1, xddot_1, dt_ds, TRIM, THETA_EST, OMEGA_EST,
    ONSET_TIMES[1], GATE_RAMP_S)
feats_2, tgt_2, t_2, st_2 = build_features_gated(
    modes_2_ds, xdot_2, xddot_2, dt_ds, TRIM, THETA_EST, OMEGA_EST,
    ONSET_TIMES[2], GATE_RAMP_S)
feats_3, tgt_3, t_3, st_3 = build_features_gated(
    modes_3_ds, xdot_3, xddot_3, dt_ds, TRIM, THETA_EST, OMEGA_EST,
    ONSET_TIMES[3], GATE_RAMP_S)

# Concatenate
features_all = np.vstack([feats_1, feats_2, feats_3])
targets_all = np.concatenate([tgt_1, tgt_2, tgt_3])
t_all = np.concatenate([t_1, t_2, t_3])
states_all = np.vstack([st_1, st_2, st_3])

ep_boundaries = [0, len(tgt_1), len(tgt_1) + len(tgt_2), len(targets_all)]

N_total = len(targets_all)
print(f"[DATA] Total samples: {N_total}")
print(f"[DATA] Per-episode: Ep1={len(tgt_1)}, Ep2={len(tgt_2)}, Ep3={len(tgt_3)}")
print(f"[DATA] Features: {features_all.shape}")
print(f"[DATA] Feature names: {FEATURE_NAMES}")
print(f"[DATA] theta={THETA_EST:.4f}, omega={OMEGA_EST:.4f}")

# Feature statistics
print(f"\n  Feature statistics:")
for i, name in enumerate(FEATURE_NAMES):
    col = features_all[:, i]
    nz = np.sum(np.abs(col) > 1e-8)
    print(f"    {name:20s}: mean={col.mean():+.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}, nonzero={nz}/{N_total}")

print(f"\n  Target (x_ddot) stats: mean={targets_all.mean():.6f}, "
      f"std={targets_all.std():.6f}")

# Normalize
feat_mean = features_all.mean(axis=0)
feat_std = features_all.std(axis=0)
feat_std[feat_std < 1e-10] = 1.0
tgt_mean = targets_all.mean()
tgt_std = targets_all.std()
if tgt_std < 1e-10:
    tgt_std = 1.0

features_n = (features_all - feat_mean) / feat_std
targets_n = (targets_all - tgt_mean) / tgt_std

print(f"[NORM] Feature stds: {feat_std.round(4)}")
print(f"[NORM] Target std: {tgt_std:.6f}, mean: {tgt_mean:.6f}")

# ===========================================================================
# 7. OLS baseline
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 6: OLS baseline")
print(f"{'='*70}")

from numpy.linalg import lstsq

features_ols = np.column_stack([np.ones(N_total), features_all])
ols_feat_names = ["1"] + FEATURE_NAMES

coeffs_ols, _, _, _ = lstsq(features_ols, targets_all, rcond=None)
pred_ols = features_ols @ coeffs_ols
ss_res = np.sum((targets_all - pred_ols) ** 2)
ss_tot = np.sum((targets_all - targets_all.mean()) ** 2)
ols_r2 = 1 - ss_res / max(ss_tot, 1e-12)

print(f"\n  OLS R^2 = {ols_r2:.4f}")
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
print(f"\n  Per-episode OLS:")
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

# ===========================================================================
# 8. KANDy training
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 7: KANDy training ([{N_FEAT}, 1], grid=3, lamb=0.001)")
print(f"{'='*70}")


class IdentityLiftWithNames(CustomLift):
    """Identity lift that preserves feature names for pre-computed features."""
    def __init__(self, n_feat, names):
        super().__init__(fn=lambda X: X, output_dim=n_feat, name="identity")
        self._names = names

    @property
    def feature_names(self):
        return self._names


GRID = 3
K_SPLINE = 3
LAMB_VALUES = [0.0005, 0.001, 0.002]
SCREEN_STEPS = 100  # quick screen
FULL_STEPS = 400    # full training for top candidates
STEPS = FULL_STEPS
PATIENCE = 10       # sweet spot for noisy iEEG data (from sweep)
N_SEEDS_PER_LAMB = 20  # seeds per lambda value (wider search)

targets_2d = targets_n[:, None]
N_SYM_PTS = min(1024, N_total)
ss_t_k = np.sum((targets_all - targets_all.mean()) ** 2)

# Primary training (or load from checkpoint)
LOAD_CHECKPOINT = os.path.exists(os.path.join(OUT_DIR, "model_checkpoint.pt"))

import io
import contextlib


def evaluate_model(m, features_n_, targets_all_, feat_std_, tgt_std_, tgt_mean_,
                   n_feat_, n_sym_pts_, ep_boundaries_=None):
    """Evaluate a trained model: R^2, active edges, edge ranges, per-ep R^2."""
    pred = m.predict(features_n_)
    if pred.ndim == 1:
        pred = pred[:, None]
    pred_un_ = pred[:, 0] * tgt_std_ + tgt_mean_
    ss_r = np.sum((targets_all_ - pred_un_) ** 2)
    ss_t = np.sum((targets_all_ - targets_all_.mean()) ** 2)
    r2 = 1 - ss_r / max(ss_t, 1e-12)

    # Per-episode R^2 (min across episodes = robustness)
    min_ep_r2 = r2
    if ep_boundaries_ is not None:
        ep_r2s = []
        for s, e in zip(ep_boundaries_[:-1], ep_boundaries_[1:]):
            ss_r_ep = np.sum((targets_all_[s:e] - pred_un_[s:e]) ** 2)
            ss_t_ep = np.sum((targets_all_[s:e] - targets_all_[s:e].mean()) ** 2)
            ep_r2s.append(1 - ss_r_ep / max(ss_t_ep, 1e-12))
        min_ep_r2 = min(ep_r2s)

    sym_in = torch.tensor(features_n_[:n_sym_pts_], dtype=torch.float32)
    m.model_.save_act = True
    with torch.no_grad():
        m.model_(sym_in)

    ranges = []
    for idx in range(n_feat_):
        x_a_, y_a_ = get_edge_activation(m.model_, l=0, i=idx, j=0)
        ranges.append(np.max(y_a_) - np.min(y_a_))

    max_rng = max(ranges)
    thr = 0.05 * max_rng if max_rng > 1e-8 else 1e-8
    n_act = sum(1 for r in ranges if r > thr)
    relu_active = ranges[5] > thr  # index 5 = ReLU(x-theta)

    return r2, n_act, relu_active, ranges, max_rng, thr, pred_un_, sym_in, min_ep_r2


def train_silently(features_n_, targets_2d_, grid_, k_, steps_, seed_, lamb_,
                   patience_, n_feat_, feature_names_):
    """Train a KANDy model with suppressed stdout."""
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    lift_t = IdentityLiftWithNames(n_feat_, feature_names_)
    m = KANDy(lift=lift_t, grid=grid_, k=k_, steps=steps_, seed=seed_, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(X=features_n_, X_dot=targets_2d_,
              val_frac=0.15, test_frac=0.10,
              lamb=lamb_, patience=patience_, verbose=False)
    return m


if LOAD_CHECKPOINT:
    print(f"\n  [CHECKPOINT] Loading saved model from {OUT_DIR}/model_checkpoint.pt")
    LAMB = 0.001  # default for checkpoint loading
    lift_m = IdentityLiftWithNames(N_FEAT, FEATURE_NAMES)
    model = KANDy(
        lift=lift_m, grid=GRID, k=K_SPLINE,
        steps=STEPS, seed=SEED, device="cpu",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(
            X=features_n, X_dot=targets_2d,
            val_frac=0.15, test_frac=0.10,
            lamb=LAMB, patience=0, steps=1, verbose=False,
        )
    ckpt = torch.load(os.path.join(OUT_DIR, "model_checkpoint.pt"), weights_only=True)
    model.model_.load_state_dict(ckpt)
    print(f"  [CHECKPOINT] Model restored successfully -- skipping training")

    kandy_r2, n_active, relu_on, edge_ranges, max_range, thresh, pred_un, sym_input, _ = \
        evaluate_model(model, features_n, targets_all, feat_std, tgt_std, tgt_mean,
                       N_FEAT, N_SYM_PTS)
    train_loss = model.train_results_["train_loss"][-1]
    print(f"  KANDy R^2={kandy_r2:.4f}, active={n_active}/{N_FEAT}, ReLU={'ON' if relu_on else 'off'}")

else:
    # Phase 1: Quick screen at SCREEN_STEPS with patience=0
    n_total = N_SEEDS_PER_LAMB * len(LAMB_VALUES)
    print(f"\n  Phase 1: Quick screen ({n_total} trials, {SCREEN_STEPS} steps) ...")
    screen_results = []

    for lamb_val in LAMB_VALUES:
        print(f"\n  --- lambda = {lamb_val} ---")
        for trial_seed in range(SEED, SEED + N_SEEDS_PER_LAMB):
            m_trial = train_silently(features_n, targets_2d, GRID, K_SPLINE,
                                     SCREEN_STEPS, trial_seed, lamb_val, 0,
                                     N_FEAT, FEATURE_NAMES)

            r2_t, n_act_t, relu_on_t, ranges_t, max_rng_t, thr_t, pred_un_t, sym_t, min_ep_r2 = \
                evaluate_model(m_trial, features_n, targets_all, feat_std, tgt_std,
                               tgt_mean, N_FEAT, N_SYM_PTS, ep_boundaries)

            score = r2_t + 0.03 * n_act_t + (0.15 if relu_on_t else 0.0) + 0.3 * max(min_ep_r2, -0.5)

            print(f"    seed={trial_seed}: R^2={r2_t:.4f}, active={n_act_t}/{N_FEAT}, "
                  f"ReLU={'ON' if relu_on_t else 'off'}, min_ep={min_ep_r2:.3f}, score={score:.4f}")

            screen_results.append((score, trial_seed, lamb_val, r2_t, n_act_t, relu_on_t, m_trial))

    # Phase 2: Full train on top 5 candidates (with patience for early stopping)
    screen_results.sort(key=lambda x: x[0], reverse=True)
    top_n = 5
    print(f"\n  Phase 2: Full training on top {top_n} configs ({FULL_STEPS} steps, patience={PATIENCE}) ...")

    # Include screen models as candidates too (they might already be optimal)
    candidates = []
    for scr, s, lv, r2_scr, n_act_scr, relu_scr, m_scr in screen_results[:top_n]:
        # Evaluate screen model
        r2_s, n_act_s, relu_s, rng_s, mx_s, thr_s, pu_s, si_s, mep_s = \
            evaluate_model(m_scr, features_n, targets_all, feat_std, tgt_std,
                           tgt_mean, N_FEAT, N_SYM_PTS, ep_boundaries)
        score_s = r2_s + 0.03 * n_act_s + (0.15 if relu_s else 0.0) + 0.3 * max(mep_s, -0.5)
        loss_s = m_scr.train_results_["train_loss"][-1]
        candidates.append((score_s, m_scr, s, lv, r2_s, n_act_s, relu_s,
                           rng_s, mx_s, thr_s, pu_s, si_s, loss_s, "screen"))

        # Full retrain
        print(f"\n    [{len(candidates)//2}/{top_n}] seed={s}, lamb={lv} (screen R^2={r2_scr:.4f})")
        m_full = train_silently(features_n, targets_2d, GRID, K_SPLINE,
                                FULL_STEPS, s, lv, PATIENCE,
                                N_FEAT, FEATURE_NAMES)
        r2_f, n_act_f, relu_f, rng_f, mx_f, thr_f, pu_f, si_f, mep_f = \
            evaluate_model(m_full, features_n, targets_all, feat_std, tgt_std,
                           tgt_mean, N_FEAT, N_SYM_PTS, ep_boundaries)
        score_f = r2_f + 0.03 * n_act_f + (0.15 if relu_f else 0.0) + 0.3 * max(mep_f, -0.5)
        loss_f = m_full.train_results_["train_loss"][-1]
        candidates.append((score_f, m_full, s, lv, r2_f, n_act_f, relu_f,
                           rng_f, mx_f, thr_f, pu_f, si_f, loss_f, "full"))
        print(f"      screen: R^2={r2_s:.4f}, active={n_act_s}/{N_FEAT}")
        print(f"      full:   R^2={r2_f:.4f}, active={n_act_f}/{N_FEAT}")

    # Pick the best across all candidates (screen + full)
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    _, model, best_seed, best_lamb, kandy_r2, n_active, relu_on, \
        edge_ranges, max_range, thresh, pred_un, sym_input, train_loss, phase = best
    LAMB = best_lamb

    print(f"\n  BEST ({phase}): seed={best_seed}, lamb={best_lamb}, R^2={kandy_r2:.4f}, "
          f"active={n_active}/{N_FEAT}, ReLU={'ON' if relu_on else 'off'}")

# --- Save model checkpoint ---
CKPT_PATH = os.path.join(OUT_DIR, "model_checkpoint.pt")
torch.save(model.model_.state_dict(), CKPT_PATH)
print(f"\n  [CHECKPOINT] Saved model to {CKPT_PATH}")
print(f"  To reload without retraining, set LOAD_CHECKPOINT = True")

print(f"\n  Edge activity summary:")
for i, name in enumerate(FEATURE_NAMES):
    status = "ON" if edge_ranges[i] > thresh else "off"
    print(f"    {name:20s}: range={edge_ranges[i]:.6f} [{status}]")

# ===========================================================================
# 9. Detailed edge analysis
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 8: Edge analysis")
print(f"{'='*70}")

# Refresh activations
model.model_.save_act = True
with torch.no_grad():
    model.model_(sym_input)

spline_data = {}
for i in range(N_FEAT):
    x_a, y_a = get_edge_activation(model.model_, l=0, i=i, j=0)
    spline_data[i] = (x_a, y_a)

edges_info = []
for i in range(N_FEAT):
    x_a, y_a = spline_data[i]
    rng = np.max(y_a) - np.min(y_a)
    is_active = rng > thresh

    if np.std(y_a) > 1e-10:
        p = np.polyfit(x_a, y_a, 1)
        slope, intercept = p[0], p[1]
        y_lin = slope * x_a + intercept
        ss_res_e = np.sum((y_a - y_lin) ** 2)
        ss_tot_e = np.sum((y_a - y_a.mean()) ** 2)
        r2_lin = 1 - ss_res_e / max(ss_tot_e, 1e-12)
        p2 = np.polyfit(x_a, y_a, 2)
        y_quad = np.polyval(p2, x_a)
        ss_res_q = np.sum((y_a - y_quad) ** 2)
        r2_quad = 1 - ss_res_q / max(ss_tot_e, 1e-12)
    else:
        slope, intercept, r2_lin, r2_quad = 0.0, 0.0, 0.0, 0.0

    edges_info.append({
        "idx": i, "name": FEATURE_NAMES[i], "range": rng,
        "slope": slope, "intercept": intercept,
        "r2_linear": r2_lin, "r2_quadratic": r2_quad,
        "active": is_active,
    })

print(f"  Active: {n_active}/{N_FEAT} (threshold = {thresh:.6f})")
print()
print(f"  {'Edge':20s} {'Range':>10s} {'Slope':>10s} {'R2_lin':>8s} {'R2_quad':>8s} {'Status':>8s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for e in sorted(edges_info, key=lambda e: e["range"], reverse=True):
    status = "ACTIVE" if e["active"] else "zeroed"
    print(f"  {e['name']:20s} {e['range']:10.6f} {e['slope']:+10.4f} "
          f"{e['r2_linear']:8.4f} {e['r2_quadratic']:8.4f} {status:>8s}")

# Extract linear-approximation equation in original space
constant_linear = tgt_mean
coeff_linear = {}
for e in edges_info:
    if e["active"]:
        i = e["idx"]
        coeff_orig = tgt_std * e["slope"] / feat_std[i]
        constant_linear -= tgt_std * e["slope"] * feat_mean[i] / feat_std[i]
        coeff_linear[e["name"]] = coeff_orig

eq_linear_terms = []
if abs(constant_linear) > 1e-6:
    eq_linear_terms.append(f"{constant_linear:+.6f}")
for fname, coeff in coeff_linear.items():
    if abs(coeff) > 1e-6:
        eq_linear_terms.append(f"{coeff:+.6f}*{fname}")
eq_linear_str = " ".join(eq_linear_terms) if eq_linear_terms else "0"
print(f"\n  Linear-approx equation: x_ddot = {eq_linear_str}")

# ===========================================================================
# 10. Approximate vanishing ideal: symbolic spline fitting
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 9: Symbolic spline fitting (approximate vanishing ideal)")
print(f"{'='*70}")


def estimate_frequency_fft(x_data, y_data):
    """Estimate dominant frequency from spline data via FFT."""
    n_pts = len(x_data)
    dx = (x_data[-1] - x_data[0]) / max(n_pts - 1, 1)
    if dx < 1e-12:
        return 1.0
    y_detrend = y_data - np.mean(y_data)
    fft_vals = np.abs(np.fft.rfft(y_detrend))
    freqs = np.fft.rfftfreq(n_pts, d=dx)
    mask = freqs > 0.5 / max(x_data[-1] - x_data[0], 1e-12)
    if not np.any(mask):
        return 1.0
    peak_idx = np.argmax(fft_vals[mask])
    return 2.0 * np.pi * freqs[mask][peak_idx]


def fit_spline_symbolic(x_data, y_data, basis_funcs, basis_names):
    """Backward-elimination symbolic fit on spline data.

    Returns kept_names, kept_coeffs, r2_full, r2_sparse, y_pred_sparse.
    """
    Phi = np.column_stack([f(x_data) for f in basis_funcs])
    n_basis = Phi.shape[1]

    coeffs, _, _, _ = np.linalg.lstsq(Phi, y_data, rcond=None)
    y_pred = Phi @ coeffs
    ss_tot = np.sum((y_data - y_data.mean())**2)
    r2_full = 1.0 - np.sum((y_data - y_pred)**2) / max(ss_tot, 1e-12)

    active = list(range(n_basis))
    r2_current = r2_full

    while len(active) > 1:
        worst_idx = None
        best_r2_after_drop = -np.inf
        for trial_drop in range(len(active)):
            trial_active = active[:trial_drop] + active[trial_drop + 1:]
            Phi_trial = Phi[:, trial_active]
            c_trial, _, _, _ = np.linalg.lstsq(Phi_trial, y_data, rcond=None)
            y_trial = Phi_trial @ c_trial
            r2_trial = 1.0 - np.sum((y_data - y_trial)**2) / max(ss_tot, 1e-12)
            if r2_trial > best_r2_after_drop:
                best_r2_after_drop = r2_trial
                worst_idx = trial_drop

        r2_loss_frac = (r2_current - best_r2_after_drop) / max(abs(r2_current), 1e-12)
        if r2_loss_frac < 0.05 and best_r2_after_drop > r2_full * 0.8:
            active = active[:worst_idx] + active[worst_idx + 1:]
            r2_current = best_r2_after_drop
        else:
            break

    Phi_sparse = Phi[:, active]
    coeffs_sparse, _, _, _ = np.linalg.lstsq(Phi_sparse, y_data, rcond=None)
    y_pred_sparse = Phi_sparse @ coeffs_sparse
    r2_sparse = 1.0 - np.sum((y_data - y_pred_sparse)**2) / max(ss_tot, 1e-12)

    kept_names = [basis_names[i] for i in active]
    return kept_names, coeffs_sparse, r2_full, r2_sparse, y_pred_sparse


def fit_poly_monomial(x_data, y_data, max_degree=4):
    """Polynomial basis up to degree max_degree."""
    basis_funcs = [lambda t, d=d: t**d for d in range(max_degree + 1)]
    basis_names = ["1"] + [f"t^{d}" if d > 1 else "t" for d in range(1, max_degree + 1)]
    return fit_spline_symbolic(x_data, y_data, basis_funcs, basis_names)


def fit_poly_trig_mixed(x_data, y_data):
    """Fit y = a0 + a1*t + a2*t^2 + a3*sin(t) + a4*cos(t)."""
    basis_funcs = [
        lambda t: np.ones_like(t),
        lambda t: t,
        lambda t: t**2,
        lambda t: np.sin(t),
        lambda t: np.cos(t),
    ]
    basis_names = ["1", "t", "t^2", "sin(t)", "cos(t)"]
    return fit_spline_symbolic(x_data, y_data, basis_funcs, basis_names)


def fit_fourier_sincos(x_data, y_data, alpha):
    """Fourier basis at frequency alpha."""
    basis_funcs = [
        lambda t: np.ones_like(t),
        lambda t, a=alpha: np.sin(a * t),
        lambda t, a=alpha: np.cos(a * t),
        lambda t, a=alpha: np.sin(2 * a * t),
        lambda t, a=alpha: np.cos(2 * a * t),
    ]
    basis_names = ["1", f"sin({alpha:.2f}t)", f"cos({alpha:.2f}t)",
                   f"sin({2*alpha:.2f}t)", f"cos({2*alpha:.2f}t)"]
    return fit_spline_symbolic(x_data, y_data, basis_funcs, basis_names)


def fit_linear_trend_sincos(x_data, y_data, alpha):
    """Linear trend + oscillatory."""
    basis_funcs = [
        lambda t: np.ones_like(t),
        lambda t: t,
        lambda t, a=alpha: np.sin(a * t),
        lambda t, a=alpha: np.cos(a * t),
    ]
    basis_names = ["1", "t", f"sin({alpha:.2f}t)", f"cos({alpha:.2f}t)"]
    return fit_spline_symbolic(x_data, y_data, basis_funcs, basis_names)


def fit_relu_response(x_data, y_data):
    """Fit ReLU edge: polynomial vs oscillatory."""
    poly_names, poly_coeffs, poly_r2f, poly_r2s, poly_pred = fit_poly_monomial(x_data, y_data, 4)

    alpha_est = estimate_frequency_fft(x_data, y_data)

    def _osc_model(t, a0, a1, a2, b, c):
        return a0 + a1 * t + a2 * np.sin(b * t + c)

    p0 = [np.mean(y_data), 0.0, np.ptp(y_data) / 2, alpha_est, 0.0]
    try:
        params, _ = curve_fit(_osc_model, x_data, y_data, p0=p0, maxfev=20000)
        y_osc = _osc_model(x_data, *params)
        ss_tot = np.sum((y_data - y_data.mean())**2)
        osc_r2 = 1.0 - np.sum((y_data - y_osc)**2) / max(ss_tot, 1e-12)
    except RuntimeError:
        osc_r2 = -1.0
        params = p0
        y_osc = _osc_model(x_data, *params)

    if osc_r2 > poly_r2s:
        a0, a1, a2, b, c = params
        t_parts = [f"{a0:.4f}"]
        if abs(a1) > 1e-4:
            t_parts.append(f"{a1:+.4f}*r")
        t_parts.append(f"{a2:+.4f}*sin({b:.2f}*r{c:+.2f})")
        name_str = " ".join(t_parts)
        return (name_str, params, osc_r2, y_osc, "oscillatory")
    else:
        return (poly_names, poly_coeffs, poly_r2s, poly_pred, "polynomial")


# Truncation fraction for x^2, x^3 edges (boundary effects)
TRUNCATE_FRAC = 0.12

spline_fits = {}

for i in range(N_FEAT):
    x_a, y_a = spline_data[i]
    fname = FEATURE_NAMES[i]
    rng = np.max(y_a) - np.min(y_a)
    is_active = rng > thresh

    print(f"\n  --- Edge {i}: {fname} (range={rng:.6f}, "
          f"{'ACTIVE' if is_active else 'zeroed'}) ---")

    if not is_active:
        spline_fits[i] = {
            "name": fname, "fit_type": "zeroed", "r2": 0.0,
            "equation": "0", "y_pred": np.zeros_like(y_a),
            "x_data": x_a, "y_data": y_a, "truncated": False,
            "trunc_idx": 0, "terms": [], "coeffs": np.array([]),
            "x_fit": x_a, "y_fit": y_a,
        }
        print(f"    -> zeroed")
        continue

    # Decide truncation
    truncated = False
    trunc_idx = 0
    x_fit, y_fit = x_a, y_a

    if i in [2, 3]:  # x^2, x^3 edges -- boundary effects
        n_trunc = int(TRUNCATE_FRAC * len(x_a))
        if n_trunc > 5:
            x_fit = x_a[n_trunc:]
            y_fit = y_a[n_trunc:]
            truncated = True
            trunc_idx = n_trunc
            print(f"    Truncated first {n_trunc} points ({TRUNCATE_FRAC*100:.0f}%)")

    best_fit = None
    best_r2 = -999.0
    best_type = "none"

    if i <= 4:
        # Duffing terms: polynomial + mixed poly-trig
        pn, pc, prf, prs, ppred = fit_poly_monomial(x_fit, y_fit, 4)
        if prs > best_r2:
            best_r2 = prs
            best_fit = (pn, pc, prs, ppred)
            best_type = "poly4"
        print(f"    poly(4): R^2={prs:.4f}, terms={pn}")

        pn2, pc2, prf2, prs2, ppred2 = fit_poly_monomial(x_fit, y_fit, 2)
        if prs2 > 0.9 * prs and prs2 > best_r2 * 0.95:
            best_r2 = prs2
            best_fit = (pn2, pc2, prs2, ppred2)
            best_type = "poly2"
        print(f"    poly(2): R^2={prs2:.4f}")

        pnm, pcm, prfm, prsm, ppredm = fit_poly_trig_mixed(x_fit, y_fit)
        if prsm > best_r2:
            best_r2 = prsm
            best_fit = (pnm, pcm, prsm, ppredm)
            best_type = "poly_trig"
        print(f"    poly+trig: R^2={prsm:.4f}")

    elif i in [5, 6, 7, 11]:
        # ReLU-type edges: polynomial and oscillatory
        result = fit_relu_response(x_fit, y_fit)
        if result[4] == "oscillatory":
            best_fit = (result[0], result[1], result[2], result[3])
            best_r2 = result[2]
            best_type = "relu_osc"
            print(f"    relu(osc): R^2={result[2]:.4f}")
        else:
            best_fit = (result[0], result[1], result[2], result[3])
            best_r2 = result[2]
            best_type = "relu_poly"
            print(f"    relu(poly): R^2={result[2]:.4f}")

        pn, pc, prf, prs, ppred = fit_poly_monomial(x_fit, y_fit, 4)
        print(f"    poly(4): R^2={prs:.4f}")
        if prs > best_r2:
            best_r2 = prs
            best_fit = (pn, pc, prs, ppred)
            best_type = "relu_poly_alt"

    else:
        # Gated forcing / gate edges: Fourier + linear trend
        alpha_est = estimate_frequency_fft(x_fit, y_fit)
        print(f"    Estimated freq: alpha={alpha_est:.4f}")

        fn, fc, frf, frs, fpred = fit_fourier_sincos(x_fit, y_fit, alpha_est)
        if frs > best_r2:
            best_r2 = frs
            best_fit = (fn, fc, frs, fpred)
            best_type = "fourier"
        print(f"    fourier: R^2={frs:.4f}")

        ln, lc, lrf, lrs, lpred = fit_linear_trend_sincos(x_fit, y_fit, alpha_est)
        if lrs > best_r2:
            best_r2 = lrs
            best_fit = (ln, lc, lrs, lpred)
            best_type = "lin_osc"
        print(f"    lin+osc: R^2={lrs:.4f}")

        pn, pc, prf, prs, ppred = fit_poly_monomial(x_fit, y_fit, 3)
        if prs > best_r2:
            best_r2 = prs
            best_fit = (pn, pc, prs, ppred)
            best_type = "poly3"
        print(f"    poly(3): R^2={prs:.4f}")

    terms_fit, coeffs_fit, r2_chosen, y_pred_fit = best_fit

    # Build equation string
    if isinstance(terms_fit, str):
        eq_str = terms_fit
    else:
        eq_parts = []
        for tn, tc in zip(terms_fit, coeffs_fit):
            if tn == "1":
                eq_parts.append(f"{tc:.4f}")
            else:
                eq_parts.append(f"{tc:+.4f}*{tn}")
        eq_str = " ".join(eq_parts)

    print(f"    BEST ({best_type}): R^2={r2_chosen:.4f}")
    print(f"    psi_{i}(t) = {eq_str}")

    spline_fits[i] = {
        "name": fname, "fit_type": best_type, "r2": r2_chosen,
        "equation": eq_str, "y_pred": y_pred_fit,
        "x_data": x_a, "y_data": y_a,
        "x_fit": x_fit, "y_fit": y_fit,
        "truncated": truncated, "trunc_idx": trunc_idx,
        "terms": terms_fit if isinstance(terms_fit, list) else [terms_fit],
        "coeffs": coeffs_fit if isinstance(coeffs_fit, np.ndarray) else np.array([]),
    }

# ===========================================================================
# 11. Compose equation in original space
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 10: Composed equation (original space)")
print(f"{'='*70}")

print(f"  (tgt_mean={tgt_mean:.6f}, tgt_std={tgt_std:.6f})")

constant_total = tgt_mean
composed_coeffs = {}

for i in range(N_FEAT):
    fname = FEATURE_NAMES[i]
    fit = spline_fits[i]
    rng = np.max(fit["y_data"]) - np.min(fit["y_data"])

    if rng <= thresh:
        continue

    x_a, y_a = fit["x_data"], fit["y_data"]
    if np.std(x_a) > 1e-10:
        p = np.polyfit(x_a, y_a, 1)
        slope_norm = p[0]
        intercept_norm = p[1]
    else:
        slope_norm = 0.0
        intercept_norm = np.mean(y_a)

    coeff_orig = tgt_std * slope_norm / feat_std[i]
    constant_total += tgt_std * (intercept_norm - slope_norm * feat_mean[i] / feat_std[i])

    composed_coeffs[fname] = {
        "linear_coeff": coeff_orig,
        "slope_norm": slope_norm,
        "intercept_norm": intercept_norm,
        "fit_type": fit["fit_type"],
        "fit_r2": fit["r2"],
        "fit_eq": fit["equation"],
    }

    print(f"    {fname:20s}: coeff={coeff_orig:+.6f}, fit_type={fit['fit_type']}, "
          f"R^2={fit['r2']:.4f}")

eq_terms = []
if abs(constant_total) > 1e-6:
    eq_terms.append(f"{constant_total:+.6f}")
for fname, info in composed_coeffs.items():
    if abs(info["linear_coeff"]) > 1e-6:
        eq_terms.append(f"{info['linear_coeff']:+.6f}*{fname}")

full_eq = " ".join(eq_terms) if eq_terms else "0"
print(f"\n  Full discovered equation:")
print(f"  x_ddot = {full_eq}")

# ===========================================================================
# 12. One-step prediction accuracy
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 11: One-step prediction accuracy")
print(f"{'='*70}")

mse_k = np.mean((targets_all - pred_un) ** 2)
rmse_k = np.sqrt(mse_k)
print(f"  KANDy R^2 = {kandy_r2:.4f} (OLS ceiling: {ols_r2:.4f})")
print(f"  KANDy RMSE = {rmse_k:.6f}")

for ep_idx, (start, end) in enumerate(
    zip(ep_boundaries[:-1], ep_boundaries[1:]), 1
):
    ss_r = np.sum((targets_all[start:end] - pred_un[start:end]) ** 2)
    ss_t = np.sum((targets_all[start:end] - targets_all[start:end].mean()) ** 2)
    r2_ep = 1 - ss_r / max(ss_t, 1e-12)
    print(f"    Episode {ep_idx}: R^2 = {r2_ep:.4f}")

# ===========================================================================
# 13. Rollout (2D system: x_dot=y, y_dot=KAN_predict)
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 12: Autoregressive rollout (Ep1)")
print(f"{'='*70}")

CLIP_VAL = 10.0


def kandy_dynamics_gated(state, t_val, model_, feat_mean_, feat_std_,
                         tgt_mean_, tgt_std_, theta, omega, onset_s,
                         gate_ramp, clip=CLIP_VAL):
    """Full 2D dynamics with gated features: [x_dot, y_dot] = [y, KAN(features)]."""
    x, y = state
    x_c = np.clip(x, -clip, clip)
    y_c = np.clip(y, -clip, clip)

    relu_val = max(x_c - theta, 0.0)
    time_gate = np.clip((t_val - onset_s) / gate_ramp, 0.0, 1.0)

    feats = np.array([
        x_c,                             # 0: x
        y_c,                             # 1: y
        x_c**2,                          # 2: x^2
        x_c**3,                          # 3: x^3
        x_c * y_c,                       # 4: x*y
        relu_val,                        # 5: ReLU
        relu_val * y_c,                  # 6: ReLU*y
        relu_val * x_c,                  # 7: ReLU*x
        np.sin(omega * t_val) * time_gate,  # 8: sin(wt)*gate
        np.cos(omega * t_val) * time_gate,  # 9: cos(wt)*gate
        time_gate,                       # 10: gate
        relu_val**2,                     # 11: ReLU^2
    ]).reshape(1, -1)

    feats_n = (feats - feat_mean_) / feat_std_

    with torch.no_grad():
        pred_n = model_.model_(torch.tensor(feats_n, dtype=torch.float32))
    xddot = pred_n.cpu().numpy().flatten()[0] * tgt_std_ + tgt_mean_
    xddot = np.clip(xddot, -50.0, 50.0)

    return np.array([y_c, xddot])


# Rollout on Episode 1
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

    def _rhs(s, t):
        return kandy_dynamics_gated(s, t, model, feat_mean, feat_std,
                                    tgt_mean, tgt_std, THETA_EST, OMEGA_EST,
                                    ONSET_TIMES[1], GATE_RAMP_S)

    k1 = _rhs(state_curr, t_curr)
    k2 = _rhs(state_curr + 0.5*h*k1, t_curr + 0.5*h)
    k3 = _rhs(state_curr + 0.5*h*k2, t_curr + 0.5*h)
    k4 = _rhs(state_curr + h*k3, t_next)
    state_curr = state_curr + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    state_curr = np.clip(state_curr, -CLIP_VAL, CLIP_VAL)
    traj_pred.append(state_curr.copy())

traj_pred = np.array(traj_pred)
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
bounded = np.all(np.isfinite(traj_pred)) and np.max(np.abs(traj_pred)) < 100
print(f"  Bounded: {'YES' if bounded else 'NO'}")

# OLS rollout for comparison
print(f"\n  OLS Rollout (Episode 1):")


def ols_dynamics_gated(state, t_val, A_ols, theta, omega, onset_s,
                       gate_ramp, clip=CLIP_VAL):
    x, y = state
    x_c = np.clip(x, -clip, clip)
    y_c = np.clip(y, -clip, clip)
    relu_val = max(x_c - theta, 0.0)
    time_gate = np.clip((t_val - onset_s) / gate_ramp, 0.0, 1.0)
    feats = np.array([
        1.0, x_c, y_c, x_c**2, x_c**3, x_c*y_c,
        relu_val, relu_val*y_c, relu_val*x_c,
        np.sin(omega * t_val) * time_gate,
        np.cos(omega * t_val) * time_gate,
        time_gate, relu_val**2,
    ])
    xddot = feats @ A_ols
    xddot = np.clip(xddot, -50.0, 50.0)
    return np.array([y_c, xddot])


traj_ols = [x0_roll.copy()]
state_ols = x0_roll.copy()
for step in range(N_ROLL - 1):
    t_curr = t_roll_ep1[step]
    t_next = t_roll_ep1[step + 1]
    h = t_next - t_curr

    def _rhs_ols(s, t):
        return ols_dynamics_gated(s, t, coeffs_ols, THETA_EST, OMEGA_EST,
                                  ONSET_TIMES[1], GATE_RAMP_S)

    k1 = _rhs_ols(state_ols, t_curr)
    k2 = _rhs_ols(state_ols + 0.5*h*k1, t_curr + 0.5*h)
    k3 = _rhs_ols(state_ols + 0.5*h*k2, t_curr + 0.5*h)
    k4 = _rhs_ols(state_ols + h*k3, t_next)
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

# ===========================================================================
# 14. Publication-quality plots
# ===========================================================================
print(f"\n{'='*70}")
print(f"  STEP 13: Publication-quality plots")
print(f"{'='*70}")

use_pub_style()

# ---- Plot 1: data_overview.png ---- Mode 0 with onset markers + time gate
fig_do, axes_do = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for ep_idx, (modes_ds, xdot_ep, onset_s) in enumerate([
    (modes_1_ds, xdot_1, ONSET_TIMES[1]),
    (modes_2_ds, xdot_2, ONSET_TIMES[2]),
    (modes_3_ds, xdot_3, ONSET_TIMES[3]),
]):
    t_ep = np.arange(modes_ds.shape[0]) * dt_ds
    ax = axes_do[ep_idx]
    ax.plot(t_ep, modes_ds[:, 0], "steelblue", lw=0.8, alpha=0.9, label="x (mode 0)")
    ax.plot(t_ep, xdot_ep[:, 0], "coral", lw=0.6, alpha=0.7, label=r"y = $\dot{x}$")
    ax.axvline(onset_s, color="red", ls="--", lw=1.0, alpha=0.7,
               label=f"Onset {onset_s:.1f}s")
    ax.axhline(THETA_EST, color="orange", ls=":", lw=0.7, alpha=0.5,
               label=rf"$\theta$={THETA_EST:.2f}")
    # Shade the gate ramp region
    gate_start = onset_s
    gate_end = onset_s + GATE_RAMP_S
    ax.axvspan(gate_start, gate_end, color="red", alpha=0.08, label="Gate ramp")
    # Shade full-gate region
    if gate_end < t_ep[-1]:
        ax.axvspan(gate_end, t_ep[-1], color="red", alpha=0.03)
    ax.set_ylabel(f"Episode {ep_idx+1}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.15, linewidth=0.4)
    if ep_idx == 2:
        ax.set_xlabel("Time (s)")

fig_do.suptitle(
    f"Mode 0: Alpha {ALPHA_LOW}-{ALPHA_HIGH} Hz, "
    f"{SMOOTH_WIN_S}s avg, {fs_ds} Hz (gate shaded)",
    fontsize=12,
)
fig_do.tight_layout()
fig_do.savefig(f"{OUT_DIR}/data_overview.png", dpi=300, bbox_inches="tight")
fig_do.savefig(f"{OUT_DIR}/data_overview.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_do)
print(f"[PLOT] Saved {OUT_DIR}/data_overview.png/pdf")

# ---- Plot 2: edge_activations.png ---- All 12 edges
fig_ed, axes_ed = plt.subplots(3, 4, figsize=(16, 10))
for i, ax in enumerate(axes_ed.flat):
    if i >= N_FEAT:
        ax.set_visible(False)
        continue
    x_a, y_a = spline_data[i]
    rng = np.max(y_a) - np.min(y_a)
    is_active = rng > thresh

    order = np.argsort(x_a)
    x_sorted = x_a[order]
    y_sorted = y_a[order]

    ax.plot(x_sorted, y_sorted, "steelblue", lw=1.5, alpha=0.9)
    if np.std(y_a) > 1e-10:
        m_s, b_i = np.polyfit(x_a, y_a, 1)
        ax.plot(x_sorted, m_s * x_sorted + b_i, "r--", lw=0.8, alpha=0.6)

    color = "black" if is_active else "gray"
    status_str = f"ACTIVE (rng={rng:.4f})" if is_active else f"zeroed (rng={rng:.4f})"
    ax.set_title(f"{FEATURE_NAMES[i]}\n{status_str}", fontsize=9, color=color)
    ax.set_xlabel(f"{FEATURE_NAMES[i]}", fontsize=8)
    ax.set_ylabel("spline out", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)

fig_ed.suptitle(
    f"KAN edges (grid={GRID}, lamb={LAMB}, R^2={kandy_r2:.4f}, "
    f"{n_active}/{N_FEAT} active)",
    fontsize=12,
)
fig_ed.tight_layout()
fig_ed.savefig(f"{OUT_DIR}/edge_activations.png", dpi=300, bbox_inches="tight")
fig_ed.savefig(f"{OUT_DIR}/edge_activations.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_ed)
print(f"[PLOT] Saved {OUT_DIR}/edge_activations.png/pdf")

# ---- Plot 3: spline_fits.png ---- Symbolic fits overlaid
fig_sf, axes_sf = plt.subplots(3, 4, figsize=(16, 10))
for i, ax in enumerate(axes_sf.flat):
    if i >= N_FEAT:
        ax.set_visible(False)
        continue

    fit = spline_fits[i]
    x_a, y_a = fit["x_data"], fit["y_data"]
    order = np.argsort(x_a)
    x_sorted = x_a[order]
    y_sorted = y_a[order]

    ax.plot(x_sorted, y_sorted, color="#1f77b4", lw=1.8, alpha=0.85, label="Learned spline")

    if fit["fit_type"] != "zeroed" and len(fit["y_pred"]) > 0:
        x_f = fit.get("x_fit", x_a)
        if fit["truncated"]:
            n_trunc = fit["trunc_idx"]
            order_fit = np.argsort(x_f)
            ax.plot(x_f[order_fit], fit["y_pred"][order_fit], color="#d62728",
                    lw=1.5, ls="--", alpha=0.9, label="Symbolic fit")
            ax.axvline(x_sorted[n_trunc], color="#ff7f0e", ls=":", lw=1.0,
                       alpha=0.7, label="Truncation")
        else:
            order_fit = np.argsort(x_f)
            ax.plot(x_f[order_fit], fit["y_pred"][order_fit], color="#d62728",
                    lw=1.5, ls="--", alpha=0.9, label="Symbolic fit")

    rng = np.max(y_a) - np.min(y_a)
    status = "ACTIVE" if rng > thresh else "zeroed"
    r2_str = f"R^2={fit['r2']:.3f}" if fit["fit_type"] != "zeroed" else "zeroed"
    ax.set_title(f"{FEATURE_NAMES[i]}  ({status})\n{r2_str}", fontsize=9)
    ax.set_xlabel(f"Normalized {FEATURE_NAMES[i]}", fontsize=8)
    ax.set_ylabel(r"$\psi_i$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    if fit["fit_type"] != "zeroed":
        ax.legend(fontsize=6.5, loc="best", framealpha=0.8)

fig_sf.suptitle(
    f"Spline activations + symbolic fits (grid={GRID}, lamb={LAMB})",
    fontsize=12,
)
fig_sf.tight_layout()
fig_sf.savefig(f"{OUT_DIR}/spline_fits.png", dpi=300, bbox_inches="tight")
fig_sf.savefig(f"{OUT_DIR}/spline_fits.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_sf)
print(f"[PLOT] Saved {OUT_DIR}/spline_fits.png/pdf")

# ---- Plot 4: limit_cycle.png ---- Phase portrait
fig_lc, ax_lc = plt.subplots(figsize=(5.5, 5.0))
ax_lc.plot(traj_true[:, 0], traj_true[:, 1],
           color="#B8B8B8", lw=0.8, alpha=0.5, zorder=1,
           label="True trajectory", rasterized=True)
ax_lc.plot(traj_pred[:, 0], traj_pred[:, 1],
           color="#1f77b4", lw=2.0, alpha=0.92, zorder=3,
           solid_capstyle="round", label="KANDy rollout")
ax_lc.axvline(THETA_EST, color="#ff7f0e", ls="--", lw=1.2, alpha=0.7,
              zorder=2, label=rf"$\theta$={THETA_EST:.2f}")
ax_lc.scatter([traj_true[0, 0]], [traj_true[0, 1]],
              s=80, c="#2ca02c", marker="o", zorder=5,
              edgecolors="white", linewidths=0.8, label="IC")
ax_lc.set_xlabel(r"$x$ (SVD mode 0)", fontsize=11)
ax_lc.set_ylabel(r"$\dot{x}$", fontsize=11)
ax_lc.spines["top"].set_visible(False)
ax_lc.spines["right"].set_visible(False)
ax_lc.legend(fontsize=9, frameon=True, framealpha=0.92, edgecolor="#888888",
             fancybox=False, borderpad=0.4, loc="upper left")
ax_lc.grid(True, alpha=0.15, linewidth=0.4)
fig_lc.tight_layout(pad=0.5)
fig_lc.savefig(f"{OUT_DIR}/limit_cycle.png", dpi=300, bbox_inches="tight")
fig_lc.savefig(f"{OUT_DIR}/limit_cycle.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_lc)
print(f"[PLOT] Saved {OUT_DIR}/limit_cycle.png/pdf")

# ---- Plot 5: rollout.png ---- Time series: true vs KANDy vs OLS
fig_ro, axes_ro = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes_ro[0].plot(t_roll_ep1[:N_ROLL], traj_true[:, 0], "k-", lw=1.5, label="True", alpha=0.8)
axes_ro[0].plot(t_roll_ep1[:N_ROLL], traj_pred[:, 0], "#1f77b4", lw=1.2,
                ls="--", label=f"KANDy (RMSE={rollout_rmse_x:.3f})")
axes_ro[0].plot(t_roll_ep1[:N_ROLL], traj_ols[:, 0], "#d62728", lw=1.0,
                ls=":", label=f"OLS (RMSE={ols_rollout_rmse_x:.3f})")
axes_ro[0].axvline(ONSET_TIMES[1], color="red", ls="--", lw=0.7, alpha=0.5, label="Onset")
axes_ro[0].axvspan(ONSET_TIMES[1], ONSET_TIMES[1]+GATE_RAMP_S, color="red",
                   alpha=0.05, label="Gate ramp")
axes_ro[0].set_ylabel("x (mode 0)")
axes_ro[0].legend(fontsize=8)
axes_ro[0].set_title("Rollout: Episode 1, Mode 0 (ReLU-gated)")
axes_ro[0].grid(True, alpha=0.15, linewidth=0.4)

axes_ro[1].plot(t_roll_ep1[:N_ROLL], traj_true[:, 1], "k-", lw=1.5, label="True", alpha=0.8)
axes_ro[1].plot(t_roll_ep1[:N_ROLL], traj_pred[:, 1], "#1f77b4", lw=1.2, ls="--", label="KANDy")
axes_ro[1].plot(t_roll_ep1[:N_ROLL], traj_ols[:, 1], "#d62728", lw=1.0, ls=":", label="OLS")
axes_ro[1].axvline(ONSET_TIMES[1], color="red", ls="--", lw=0.7, alpha=0.5)
axes_ro[1].set_ylabel(r"$\dot{x}$")
axes_ro[1].set_xlabel("Time (s)")
axes_ro[1].legend(fontsize=8)
axes_ro[1].grid(True, alpha=0.15, linewidth=0.4)

fig_ro.tight_layout()
fig_ro.savefig(f"{OUT_DIR}/rollout.png", dpi=300, bbox_inches="tight")
fig_ro.savefig(f"{OUT_DIR}/rollout.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_ro)
print(f"[PLOT] Saved {OUT_DIR}/rollout.png/pdf")

# ---- Plot 6: relu_modulation.png ---- How ReLU modulates different terms
fig_rm, axes_rm = plt.subplots(2, 2, figsize=(10, 8))

# Panel 1: ReLU(x-theta) raw
ax = axes_rm[0, 0]
for ep_idx, (feats_ep, t_ep) in enumerate([
    (feats_1, t_1), (feats_2, t_2), (feats_3, t_3)
], 1):
    ax.plot(t_ep, feats_ep[:, 5], lw=0.8, alpha=0.8, label=f"Ep{ep_idx}")
ax.set_ylabel(r"ReLU$(x - \theta)$")
ax.set_title("ReLU activation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15, linewidth=0.4)

# Panel 2: ReLU*y (modulated damping)
ax = axes_rm[0, 1]
for ep_idx, (feats_ep, t_ep) in enumerate([
    (feats_1, t_1), (feats_2, t_2), (feats_3, t_3)
], 1):
    ax.plot(t_ep, feats_ep[:, 6], lw=0.8, alpha=0.8, label=f"Ep{ep_idx}")
ax.set_ylabel(r"ReLU$(x-\theta) \cdot y$")
ax.set_title("ReLU-modulated damping")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15, linewidth=0.4)

# Panel 3: ReLU*x (modulated position)
ax = axes_rm[1, 0]
for ep_idx, (feats_ep, t_ep) in enumerate([
    (feats_1, t_1), (feats_2, t_2), (feats_3, t_3)
], 1):
    ax.plot(t_ep, feats_ep[:, 7], lw=0.8, alpha=0.8, label=f"Ep{ep_idx}")
ax.set_ylabel(r"ReLU$(x-\theta) \cdot x$")
ax.set_xlabel("Time (s)")
ax.set_title("ReLU-modulated position coupling")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15, linewidth=0.4)

# Panel 4: Time gate activation
ax = axes_rm[1, 1]
for ep_idx, (feats_ep, t_ep, onset_s) in enumerate([
    (feats_1, t_1, ONSET_TIMES[1]),
    (feats_2, t_2, ONSET_TIMES[2]),
    (feats_3, t_3, ONSET_TIMES[3]),
], 1):
    gate = feats_ep[:, 10]
    ax.plot(t_ep, gate, lw=1.2, alpha=0.8, label=f"Ep{ep_idx} (onset={onset_s:.1f}s)")
    ax.axvline(onset_s, color=f"C{ep_idx-1}", ls=":", lw=0.7, alpha=0.5)
ax.set_ylabel("Time gate")
ax.set_xlabel("Time (s)")
ax.set_title(f"Seizure gate (ramp={GATE_RAMP_S}s)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15, linewidth=0.4)

fig_rm.suptitle("ReLU-gated modulation of dynamics", fontsize=12)
fig_rm.tight_layout()
fig_rm.savefig(f"{OUT_DIR}/relu_modulation.png", dpi=300, bbox_inches="tight")
fig_rm.savefig(f"{OUT_DIR}/relu_modulation.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_rm)
print(f"[PLOT] Saved {OUT_DIR}/relu_modulation.png/pdf")

# ---- Plot 7: coefficient_summary.png ---- Bar chart
active_names = []
active_coeffs_list = []
for fname, info in composed_coeffs.items():
    active_names.append(fname)
    active_coeffs_list.append(info["linear_coeff"])

if active_names:
    fig_cs, ax_cs = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(active_names))
    colors_bar = ["#1f77b4" if c >= 0 else "#d62728" for c in active_coeffs_list]
    bars = ax_cs.bar(x_pos, active_coeffs_list, color=colors_bar, alpha=0.85,
                     edgecolor="black", linewidth=0.5)
    ax_cs.set_xticks(x_pos)
    ax_cs.set_xticklabels(active_names, fontsize=9, rotation=35, ha="right")
    ax_cs.set_ylabel("Coefficient (original space)", fontsize=10)
    ax_cs.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax_cs.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    ax_cs.spines["top"].set_visible(False)
    ax_cs.spines["right"].set_visible(False)

    for bar, val in zip(bars, active_coeffs_list):
        y_off = 0.01 * max(abs(v) for v in active_coeffs_list)
        ax_cs.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + np.sign(val) * y_off,
                   f"{val:.4f}", ha="center",
                   va="bottom" if val >= 0 else "top", fontsize=8)

    if abs(constant_total) > 1e-6:
        ax_cs.annotate(
            f"Constant: {constant_total:.6f}",
            xy=(0.02, 0.98), xycoords="axes fraction",
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    fig_cs.suptitle(
        r"Discovered coefficients: $\ddot{x}$ = const + $\sum c_i \cdot$feature$_i$"
        " (ReLU-gated model)",
        fontsize=11,
    )
    fig_cs.tight_layout()
    fig_cs.savefig(f"{OUT_DIR}/coefficient_summary.png", dpi=300, bbox_inches="tight")
    fig_cs.savefig(f"{OUT_DIR}/coefficient_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_cs)
    print(f"[PLOT] Saved {OUT_DIR}/coefficient_summary.png/pdf")

# ---- Plot 8: onestep.png ---- One-step prediction scatter
fig_os, axes_os = plt.subplots(1, 2, figsize=(10, 4.5))
axes_os[0].scatter(targets_all, pred_un, s=3, alpha=0.3, c="steelblue")
lims = [min(targets_all.min(), pred_un.min()),
        max(targets_all.max(), pred_un.max())]
axes_os[0].plot(lims, lims, "k--", lw=0.7, alpha=0.5)
axes_os[0].set_xlabel(r"True $\ddot{x}$")
axes_os[0].set_ylabel(r"Predicted $\ddot{x}$")
axes_os[0].set_title(f"KANDy ($R^2$={kandy_r2:.4f})")

pred_ols_all = features_ols @ coeffs_ols
axes_os[1].scatter(targets_all, pred_ols_all, s=3, alpha=0.3, c="coral")
axes_os[1].plot(lims, lims, "k--", lw=0.7, alpha=0.5)
axes_os[1].set_xlabel(r"True $\ddot{x}$")
axes_os[1].set_ylabel(r"Predicted $\ddot{x}$")
axes_os[1].set_title(f"OLS ($R^2$={ols_r2:.4f})")

fig_os.suptitle("One-step prediction: KANDy vs OLS (ReLU-gated)", fontsize=12)
fig_os.tight_layout()
fig_os.savefig(f"{OUT_DIR}/onestep.png", dpi=300, bbox_inches="tight")
fig_os.savefig(f"{OUT_DIR}/onestep.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_os)
print(f"[PLOT] Saved {OUT_DIR}/onestep.png/pdf")

# ---- Additional diagnostic: loss curves and edge activations via kandy.plotting
sym_input_full = torch.tensor(features_n[:N_SYM_PTS], dtype=torch.float32)
model.model_.save_act = True
with torch.no_grad():
    model.model_(sym_input_full)

fig_edges, axes_edges = plot_all_edges(
    model.model_,
    X=sym_input_full,
    fits=["linear"],
    in_var_names=FEATURE_NAMES,
    out_var_names=["x_ddot"],
    figsize_per_panel=(3.0, 2.2),
    save=f"{OUT_DIR}/edge_activations_kandy",
)
plt.close(fig_edges)
print(f"[PLOT] Saved {OUT_DIR}/edge_activations_kandy.png/pdf")

fig_loss, ax_loss = plot_loss_curves(
    model.train_results_,
    save=f"{OUT_DIR}/loss_curves",
)
plt.close(fig_loss)
print(f"[PLOT] Saved {OUT_DIR}/loss_curves.png/pdf")

# ===========================================================================
# 15. Summary
# ===========================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY: ReLU-Gated Duffing Oscillator for iEEG")
print(f"{'='*70}")
print(f"""
  Target model (ReLU-gated Duffing):
    x_ddot = mu*x - x^3 - alpha*y + eta*x^2 - kappa*x*y
             + beta1*ReLU(x-theta)
             + beta2*ReLU(x-theta)*y        (ReLU-gated damping)
             + beta3*ReLU(x-theta)*x        (ReLU-gated position)
             + gamma*sin(wt)*gate(t)         (gated sin forcing)
             + delta*cos(wt)*gate(t)         (gated cos forcing)
             + epsilon*gate(t)               (DC shift at seizure)
             + zeta*ReLU(x-theta)^2          (quadratic ReLU)

  Configuration:
    Data:         {N_total} Mode 0 samples (3 episodes, {fs_ds} Hz)
    Preprocessing: Alpha {ALPHA_LOW}-{ALPHA_HIGH} Hz, Hilbert, {SMOOTH_WIN_S}s avg, log(A+1)
    SVD mode 0:   {cumvar[0]:.1f}% variance
    Embedding:    2-jet (x, y=x_dot) -> target x_ddot
    Lift:         {FEATURE_NAMES}
    KAN:          [{N_FEAT}, 1], grid={GRID}, k={K_SPLINE}, lamb={LAMB}
    theta (ReLU): {THETA_EST:.4f}
    omega:        {OMEGA_EST:.4f} rad/s
    Gate ramp:    {GATE_RAMP_S}s (smooth ramp 0->1 at seizure onset)

  Results:
    OLS R^2:      {ols_r2:.4f}
    KANDy R^2:    {kandy_r2:.4f}
    Active edges: {n_active}/{N_FEAT}

  Rollout (Ep1):
    KANDy RMSE:   {rollout_rmse:.4f} (NRMSE={nrmse:.4f})
    OLS RMSE:     {ols_rollout_rmse:.4f} (NRMSE={ols_nrmse:.4f})
""")

print(f"  Discovered equation:")
print(f"    x_ddot = {full_eq}")
print()

print(f"  Spline symbolic fits:")
print(f"  {'Edge':20s} {'Type':12s} {'R^2':>6s} {'Equation'}")
print(f"  {'-'*20} {'-'*12} {'-'*6} {'-'*40}")
for i in range(N_FEAT):
    fit = spline_fits[i]
    eq_short = fit['equation'][:50]
    print(f"  {FEATURE_NAMES[i]:20s} {fit['fit_type']:12s} "
          f"{fit['r2']:6.3f} {eq_short}")

print(f"\n  Composed coefficients (original space):")
print(f"  {'Term':20s} {'Coefficient':>15s} {'Spline R^2':>10s}")
print(f"  {'-'*20} {'-'*15} {'-'*10}")
if abs(constant_total) > 1e-6:
    print(f"  {'constant':20s} {constant_total:+15.6f}")
for fname, info in composed_coeffs.items():
    print(f"  {fname:20s} {info['linear_coeff']:+15.6f} {info['fit_r2']:10.4f}")

print(f"\n  All results saved to: {OUT_DIR}/")
print(f"  Done.")
