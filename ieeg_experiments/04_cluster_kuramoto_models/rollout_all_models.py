#!/usr/bin/env python3
"""Rollout all iEEG KANDy models and plot predicted vs true trajectories.

For each model:
  - Train the KAN
  - RK4 rollout on held-out episode data
  - Plot predicted vs true trajectories per cluster
  - Compute RMSE / NRMSE

Models:
  A. 3-Cluster Amplitude Dynamics (alpha, [9,3])
  B. Envelope Phase Kuramoto ([15,3])
  C. Stuart-Landau OLS (dA/dt = mu*A - alpha*A^3)

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import numpy as np
import torch
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from scipy.integrate import solve_ivp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ieeg_experiments" / "03_clustering_and_supporting_analysis"))
sys.path.insert(0, str(ROOT / "src"))

from ieeg_utils import (
    load_episodes, extract_cluster_amplitudes, savgol_derivative,
    save_fig, setup_style,
    FS, ONSET_TIMES, CLUSTERS, CLUSTER_IDS, CLUSTER_NAMES,
    CLUSTER_COLORS, ALPHA_BAND,
)
from kandy import KANDy, CustomLift

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).resolve().parent
N_CLUSTERS = len(CLUSTER_IDS)
setup_style()

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("ROLLOUT ALL iEEG MODELS: PREDICTED vs TRUE")
print("=" * 70)
episodes = load_episodes()

# ============================================================
# Shared: extract amplitude envelopes at 5 Hz
# ============================================================
DS_A = 100
DT_A = DS_A / FS

amp_data = {}
for ep in [1, 2, 3]:
    modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
    amp_data[ep] = {cid: modes[cid][::DS_A] for cid in CLUSTER_IDS}

# ============================================================
# MODEL A: Amplitude Dynamics [9,3]
# ============================================================
print("\n" + "=" * 70)
print("MODEL A: 3-Cluster Amplitude Dynamics [9,3]")
print("=" * 70)

SG_WIN = 13
SG_POLY = 4
TRIM = SG_WIN // 2 + 3
TRAIN_START, TRAIN_END = 30.0, 97.0
PAIR_A = [(0, 1), (0, 2), (1, 2)]

# ReLU thresholds
thresholds_A = {}
for cid in CLUSTER_IDS:
    vals = []
    for ep in [1, 2, 3]:
        t = np.arange(len(amp_data[ep][cid])) * DT_A
        mask = (t >= ONSET_TIMES[ep] - 10) & (t < ONSET_TIMES[ep])
        if mask.sum() > 0:
            vals.append(amp_data[ep][cid][mask].mean())
    thresholds_A[cid] = np.mean(vals)

N_FEAT_A = 9
FEAT_A = ([f"x_{c}" for c in CLUSTER_IDS] +
          [f"x{CLUSTER_IDS[i]}*x{CLUSTER_IDS[j]}" for i, j in PAIR_A] +
          [f"ReLU(x{c})" for c in CLUSTER_IDS])


def build_lift_A(x_vec):
    """Build lift for single (3,) state vector."""
    feats = list(x_vec)
    for i, j in PAIR_A:
        feats.append(x_vec[i] * x_vec[j])
    for ci, cid in enumerate(CLUSTER_IDS):
        feats.append(max(x_vec[ci] - thresholds_A[cid], 0.0))
    return np.array(feats)[None, :]


# Build training data (all 3 episodes)
Phi_parts, dot_parts, x_parts = [], [], []
for ep in [1, 2, 3]:
    n = len(amp_data[ep][CLUSTER_IDS[0]])
    s, e = int(TRAIN_START / DT_A), min(int(TRAIN_END / DT_A), n)
    x_all = np.column_stack([amp_data[ep][cid][s:e] for cid in CLUSTER_IDS])
    x_dot = np.zeros_like(x_all)
    for ci in range(N_CLUSTERS):
        x_dot[:, ci] = savgol_filter(x_all[:, ci], SG_WIN, SG_POLY,
                                     deriv=1, delta=DT_A)
    v = slice(TRIM, len(x_all) - TRIM)
    x_v, xd_v = x_all[v], x_dot[v]
    feats = []
    for ci in range(N_CLUSTERS):
        feats.append(x_v[:, ci])
    for i, j in PAIR_A:
        feats.append(x_v[:, i] * x_v[:, j])
    for ci, cid in enumerate(CLUSTER_IDS):
        feats.append(np.maximum(x_v[:, ci] - thresholds_A[cid], 0.0))
    Phi_parts.append(np.column_stack(feats))
    dot_parts.append(xd_v)
    x_parts.append(x_v)

Phi_A = np.vstack(Phi_parts)
Dot_A = np.vstack(dot_parts)

# Train
lift_A = CustomLift(fn=lambda X: X, output_dim=N_FEAT_A, name="amp_id")
model_A = KANDy(lift=lift_A, grid=3, k=3, steps=200, seed=SEED, device="cpu")
model_A.fit(X=Phi_A, X_dot=Dot_A, val_frac=0.15, test_frac=0.15,
            lamb=0.0, patience=0, verbose=False)

# One-step R²
pred_os = model_A.predict(Phi_A[-200:])
true_os = Dot_A[-200:]
r2_A = 1 - np.sum((pred_os - true_os)**2) / np.sum((true_os - true_os.mean(0))**2)
print(f"  One-step R² = {r2_A:.4f}")

# Rollout on each episode
for ep in [1, 2, 3]:
    onset = ONSET_TIMES[ep]
    n = len(amp_data[ep][CLUSTER_IDS[0]])
    t_full = np.arange(n) * DT_A

    # Rollout window: 20s before onset to 15s after
    roll_start = max(onset - 20, TRAIN_START + TRIM * DT_A)
    roll_end = min(onset + 15, TRAIN_END - TRIM * DT_A)
    s_idx = int(roll_start / DT_A)
    e_idx = int(roll_end / DT_A)
    N_ROLL = e_idx - s_idx

    x_true = np.column_stack([amp_data[ep][cid][s_idx:e_idx] for cid in CLUSTER_IDS])
    t_roll = t_full[s_idx:e_idx]
    x0 = x_true[0].copy()

    # RK4 rollout
    traj = [x0.copy()]
    state = x0.copy()
    for step in range(N_ROLL - 1):
        def f_A(s):
            return model_A.predict(build_lift_A(s)).ravel()
        k1 = f_A(state)
        k2 = f_A(state + 0.5 * DT_A * k1)
        k3 = f_A(state + 0.5 * DT_A * k2)
        k4 = f_A(state + DT_A * k3)
        state = state + (DT_A / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state = np.clip(state, -20, 20)
        traj.append(state.copy())
    traj = np.array(traj)

    rmse = np.sqrt(np.mean((traj - x_true)**2))
    x_range = x_true.max() - x_true.min()
    nrmse = rmse / max(x_range, 1e-8)
    print(f"  Ep{ep} rollout: RMSE={rmse:.4f}, NRMSE={nrmse:.4f}")

    # Plot
    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS), sharex=True)
    fig.suptitle(f"Model A: Amplitude Dynamics [9,3] — Episode {ep}\n"
                 f"One-step R²={r2_A:.3f}, Rollout NRMSE={nrmse:.3f}", fontsize=12)
    for ci, cid in enumerate(CLUSTER_IDS):
        ax = axes[ci]
        ax.plot(t_roll, x_true[:, ci], color=CLUSTER_COLORS[cid], lw=1.4,
                label="True")
        ax.plot(t_roll, traj[:, ci], color=CLUSTER_COLORS[cid], lw=1.0, ls="--",
                label="KANDy rollout")
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6, label="Onset")
        ax.set_ylabel(f"$x_{{{cid}}}(t)$")
        ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"rollout_amplitude_ep{ep}")

# Combined one-step scatter: predicted derivative vs true derivative
fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(14, 4))
fig.suptitle("Model A: One-Step Prediction (all episodes)", fontsize=12)
all_pred = model_A.predict(Phi_A)
for ci, cid in enumerate(CLUSTER_IDS):
    ax = axes[ci]
    ax.scatter(Dot_A[:, ci], all_pred[:, ci], s=2, alpha=0.3,
              color=CLUSTER_COLORS[cid])
    lims = [min(Dot_A[:, ci].min(), all_pred[:, ci].min()),
            max(Dot_A[:, ci].max(), all_pred[:, ci].max())]
    ax.plot(lims, lims, "k--", lw=0.8)
    ss_res = np.sum((all_pred[:, ci] - Dot_A[:, ci])**2)
    ss_tot = np.sum((Dot_A[:, ci] - Dot_A[:, ci].mean())**2)
    r2_c = 1 - ss_res / max(ss_tot, 1e-15)
    ax.set_title(f"{CLUSTER_NAMES[cid]}\nR²={r2_c:.3f}", fontsize=10)
    ax.set_xlabel("True $dx/dt$")
    if ci == 0:
        ax.set_ylabel("Predicted $dx/dt$")
fig.tight_layout()
save_fig(fig, OUT_DIR / "onestep_amplitude")


# ============================================================
# MODEL B: Envelope Phase Kuramoto [15,3]
# ============================================================
print("\n" + "=" * 70)
print("MODEL B: Envelope Phase Kuramoto [15,3]")
print("=" * 70)

DS_B = 10
DT_B_env = DS_B / FS
FS_B_env = FS / DS_B
DS_B_phase = 5
DT_B = DT_B_env * DS_B_phase
SG_WIN_B = 15
TRIM_B = SG_WIN_B // 2 + 3
CLUSTER_PAIRS_B = [(0, 2), (0, 3), (2, 3)]
slow_lo, slow_hi = 0.03, 0.1
N_FEAT_B = 15

# Extract envelopes at 50 Hz, then slow phase at 10 Hz
env_data = {}
for ep in [1, 2, 3]:
    modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
    env_data[ep] = {cid: modes[cid][::DS_B] for cid in CLUSTER_IDS}

psi_data, r_data_B, dpsi_data = {}, {}, {}
for ep in [1, 2, 3]:
    psi_data[ep], r_data_B[ep], dpsi_data[ep] = {}, {}, {}
    for cid in CLUSTER_IDS:
        sig = env_data[ep][cid]
        nyq = FS_B_env / 2.0
        b, a = butter(3, [max(slow_lo/nyq, 0.001), min(slow_hi/nyq, 0.999)], btype="band")
        env_filt = filtfilt(b, a, sig)
        analytic = hilbert(env_filt)
        psi_data[ep][cid] = np.angle(analytic)[::DS_B_phase]
        r_data_B[ep][cid] = np.abs(analytic)[::DS_B_phase]
        psi_uw = np.unwrap(np.angle(analytic)[::DS_B_phase])
        dpsi_data[ep][cid] = savgol_derivative(psi_uw, DT_B, window=SG_WIN_B, polyorder=4)

# ReLU thresholds
r_thresh_B = {}
for cid in CLUSTER_IDS:
    vals = []
    for ep in [1, 2, 3]:
        t = np.arange(len(psi_data[ep][cid])) * DT_B
        mask = (t >= ONSET_TIMES[ep] - 15) & (t < ONSET_TIMES[ep])
        if mask.sum() > 0:
            vals.append(r_data_B[ep][cid][mask].mean())
    r_thresh_B[cid] = np.mean(vals) if vals else 0.0


def build_features_B(ep, idx=None):
    """Build features for episode ep. If idx is a slice/int, apply it."""
    T = len(psi_data[ep][CLUSTER_IDS[0]])
    feats = np.zeros((T, N_FEAT_B))
    col = 0
    for ci, cj in CLUSTER_PAIRS_B:
        diff = psi_data[ep][ci] - psi_data[ep][cj]
        feats[:, col] = np.sin(diff); feats[:, col+1] = np.cos(diff); col += 2
    for cid in CLUSTER_IDS:
        feats[:, col] = r_data_B[ep][cid]; col += 1
    relu_v = {}
    for cid in CLUSTER_IDS:
        relu = np.maximum(r_data_B[ep][cid] - r_thresh_B[cid], 0.0)
        feats[:, col] = relu; relu_v[cid] = relu; col += 1
    psi_mean = np.zeros(T)
    for cid in CLUSTER_IDS:
        psi_mean += psi_data[ep][cid]
    psi_mean /= N_CLUSTERS
    for cid in CLUSTER_IDS:
        feats[:, col] = relu_v[cid] * np.sin(psi_data[ep][cid] - psi_mean); col += 1
    if idx is not None:
        return feats[idx]
    return feats


def build_single_feat_B(psi_vec, r_vec):
    """Build features from state psi (3,) and r (3,) for one timestep."""
    feats = np.zeros(N_FEAT_B)
    col = 0
    psi_d = {cid: psi_vec[i] for i, cid in enumerate(CLUSTER_IDS)}
    r_d = {cid: r_vec[i] for i, cid in enumerate(CLUSTER_IDS)}
    for ci, cj in CLUSTER_PAIRS_B:
        diff = psi_d[ci] - psi_d[cj]
        feats[col] = np.sin(diff); feats[col+1] = np.cos(diff); col += 2
    for cid in CLUSTER_IDS:
        feats[col] = r_d[cid]; col += 1
    relu_v = {}
    for cid in CLUSTER_IDS:
        relu = max(r_d[cid] - r_thresh_B[cid], 0.0)
        feats[col] = relu; relu_v[cid] = relu; col += 1
    psi_mean = np.mean(list(psi_d.values()))
    for cid in CLUSTER_IDS:
        feats[col] = relu_v[cid] * np.sin(psi_d[cid] - psi_mean); col += 1
    return feats[None, :]


# Build training data
Phi_Bp, Dot_Bp = [], []
for ep in [1, 2, 3]:
    feats = build_features_B(ep)
    tgts = np.column_stack([dpsi_data[ep][cid] for cid in CLUSTER_IDS])
    Phi_Bp.append(feats[TRIM_B:-TRIM_B])
    Dot_Bp.append(tgts[TRIM_B:-TRIM_B])
Phi_B = np.vstack(Phi_Bp)
Dot_B = np.vstack(Dot_Bp)
n_B = min(len(Phi_B), len(Dot_B))
Phi_B, Dot_B = Phi_B[:n_B], Dot_B[:n_B]

# Train
lift_B = CustomLift(fn=lambda X: X, output_dim=N_FEAT_B, name="phase_id")
model_B = KANDy(lift=lift_B, grid=5, k=3, steps=200, seed=SEED, device="cpu")
model_B.fit(X=Phi_B, X_dot=Dot_B, val_frac=0.15, test_frac=0.15,
            lamb=0.0, patience=0, verbose=False)

# One-step R²
pred_os = model_B.predict(Phi_B[-200:])
true_os = Dot_B[-200:]
r2_B = 1 - np.sum((pred_os - true_os)**2) / np.sum((true_os - true_os.mean(0))**2)
print(f"  One-step R² = {r2_B:.4f}")

# Rollout per episode (use ground truth r for features, rollout psi only)
for ep in [1, 2, 3]:
    onset = ONSET_TIMES[ep]
    n_ph = len(psi_data[ep][CLUSTER_IDS[0]])
    t_ph = np.arange(n_ph) * DT_B

    roll_start_idx = max(TRIM_B, n_ph // 5)
    N_ROLL = min(300, n_ph - roll_start_idx - TRIM_B)

    psi0 = np.array([np.unwrap(psi_data[ep][cid])[roll_start_idx]
                      for cid in CLUSTER_IDS])
    r_ep = np.column_stack([r_data_B[ep][cid] for cid in CLUSTER_IDS])

    traj = [psi0.copy()]
    state = psi0.copy()
    for step in range(N_ROLL - 1):
        idx = min(roll_start_idx + step, n_ph - 1)
        r_now = r_ep[idx]
        def f_B(s, r_v=r_now):
            return model_B.predict(build_single_feat_B(s, r_v)).ravel()
        k1 = f_B(state)
        k2 = f_B(state + 0.5 * DT_B * k1)
        k3 = f_B(state + 0.5 * DT_B * k2)
        k4 = f_B(state + DT_B * k3)
        state = state + (DT_B / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(state.copy())
    traj = np.array(traj)

    true_psi = np.column_stack([
        np.unwrap(psi_data[ep][cid])[roll_start_idx:roll_start_idx + N_ROLL]
        for cid in CLUSTER_IDS
    ])
    nr = min(len(traj), len(true_psi))
    traj, true_psi = traj[:nr], true_psi[:nr]
    t_r = np.arange(nr) * DT_B + t_ph[roll_start_idx]

    rmse = np.sqrt(np.mean((traj - true_psi)**2))
    print(f"  Ep{ep} rollout RMSE={rmse:.4f}")

    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS), sharex=True)
    fig.suptitle(f"Model B: Envelope Phase [15,3] — Episode {ep}\n"
                 f"One-step R²={r2_B:.3f}, Rollout RMSE={rmse:.3f}", fontsize=12)
    for ci, cid in enumerate(CLUSTER_IDS):
        ax = axes[ci]
        ax.plot(t_r, true_psi[:, ci], color=CLUSTER_COLORS[cid], lw=1.4,
                label="True")
        ax.plot(t_r, traj[:, ci], color=CLUSTER_COLORS[cid], lw=1.0, ls="--",
                label="KANDy rollout")
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6, label="Onset")
        ax.set_ylabel(rf"$\psi_{{{cid}}}$ (rad)")
        ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"rollout_envelope_phase_ep{ep}")

# One-step scatter
fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(14, 4))
fig.suptitle("Model B: One-Step Prediction (all episodes)", fontsize=12)
all_pred_B = model_B.predict(Phi_B)
for ci, cid in enumerate(CLUSTER_IDS):
    ax = axes[ci]
    ax.scatter(Dot_B[:, ci], all_pred_B[:, ci], s=2, alpha=0.3,
              color=CLUSTER_COLORS[cid])
    lims = [min(Dot_B[:, ci].min(), all_pred_B[:, ci].min()),
            max(Dot_B[:, ci].max(), all_pred_B[:, ci].max())]
    ax.plot(lims, lims, "k--", lw=0.8)
    ss_r = np.sum((all_pred_B[:, ci] - Dot_B[:, ci])**2)
    ss_t = np.sum((Dot_B[:, ci] - Dot_B[:, ci].mean())**2)
    r2_c = 1 - ss_r / max(ss_t, 1e-15)
    ax.set_title(f"{CLUSTER_NAMES[cid]}\nR²={r2_c:.3f}", fontsize=10)
    ax.set_xlabel(r"True $d\psi/dt$")
    if ci == 0:
        ax.set_ylabel(r"Predicted $d\psi/dt$")
fig.tight_layout()
save_fig(fig, OUT_DIR / "onestep_envelope_phase")


# ============================================================
# MODEL C: Stuart-Landau OLS
# ============================================================
print("\n" + "=" * 70)
print("MODEL C: Stuart-Landau OLS (dA/dt = mu*A - alpha*A^3)")
print("=" * 70)

for ep in [1, 2, 3]:
    onset = ONSET_TIMES[ep]
    n = len(amp_data[ep][CLUSTER_IDS[0]])
    t_full = np.arange(n) * DT_A

    # Fit on seizure window for each cluster
    mu_fit, alpha_fit = {}, {}
    for cid in CLUSTER_IDS:
        A = amp_data[ep][cid]
        dAdt = savgol_filter(A, SG_WIN, SG_POLY, deriv=1, delta=DT_A)
        mask = (t_full >= onset) & (t_full < onset + 10)
        A_w, dA_w = A[mask], dAdt[mask]
        X_sl = np.column_stack([A_w, A_w**3])
        coeffs, _, _, _ = np.linalg.lstsq(X_sl, dA_w, rcond=None)
        mu_fit[cid] = coeffs[0]
        alpha_fit[cid] = -coeffs[1]
        print(f"  Ep{ep} {CLUSTER_NAMES[cid]}: dA/dt = {mu_fit[cid]:+.4f}*A "
              f"- {alpha_fit[cid]:.4f}*A^3")

    # Rollout: 20s before onset to 15s after
    roll_start = max(onset - 20, 30.0)
    roll_end = min(onset + 15, 97.0)
    s_idx = int(roll_start / DT_A)
    e_idx = int(roll_end / DT_A)
    N_ROLL = e_idx - s_idx
    t_roll = t_full[s_idx:e_idx]

    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS), sharex=True)
    fig.suptitle(f"Model C: Stuart-Landau OLS — Episode {ep}\n"
                 f"$dA/dt = \\mu A - \\alpha A^3$ (fit on seizure window)", fontsize=12)

    for ci, cid in enumerate(CLUSTER_IDS):
        A_true = amp_data[ep][cid][s_idx:e_idx]
        A0 = A_true[0]
        mu, alpha = mu_fit[cid], alpha_fit[cid]

        # Solve ODE: dA/dt = mu*A - alpha*A^3
        def sl_rhs(t, A, _mu=mu, _alpha=alpha):
            return _mu * A - _alpha * A**3

        def blowup_event(t, A, _mu=mu, _alpha=alpha):
            return 50.0 - abs(A[0])
        blowup_event.terminal = True

        sol = solve_ivp(sl_rhs, [0, (N_ROLL - 1) * DT_A], [A0],
                       t_eval=np.arange(N_ROLL) * DT_A,
                       method="RK45", rtol=1e-8, atol=1e-10,
                       events=blowup_event)
        A_pred = sol.y[0]
        # Pad to full length if solver stopped early
        if len(A_pred) < N_ROLL:
            A_pred = np.pad(A_pred, (0, N_ROLL - len(A_pred)),
                           constant_values=A_pred[-1])

        rmse_c = np.sqrt(np.mean((A_pred - A_true)**2))

        ax = axes[ci]
        ax.plot(t_roll, A_true, color=CLUSTER_COLORS[cid], lw=1.4,
                label="True")
        ax.plot(t_roll[:len(A_pred)], A_pred, color=CLUSTER_COLORS[cid],
                lw=1.0, ls="--",
                label=f"SL: $\\mu$={mu:.3f}, $\\alpha$={alpha:.3f}")
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6, label="Onset")
        ax.set_ylabel(f"$A_{{{cid}}}(t)$")
        ax.set_title(f"{CLUSTER_NAMES[cid]} (RMSE={rmse_c:.3f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"rollout_stuart_landau_ep{ep}")

    # Also fit on PRE-seizure and plot
    mu_pre, alpha_pre = {}, {}
    for cid in CLUSTER_IDS:
        A = amp_data[ep][cid]
        dAdt = savgol_filter(A, SG_WIN, SG_POLY, deriv=1, delta=DT_A)
        mask = (t_full >= onset - 20) & (t_full < onset)
        A_w, dA_w = A[mask], dAdt[mask]
        X_sl = np.column_stack([A_w, A_w**3])
        coeffs, _, _, _ = np.linalg.lstsq(X_sl, dA_w, rcond=None)
        mu_pre[cid] = coeffs[0]
        alpha_pre[cid] = -coeffs[1]

    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS), sharex=True)
    fig.suptitle(f"Model C: Stuart-Landau — Episode {ep}\n"
                 f"Pre-seizure fit vs Seizure fit", fontsize=12)

    for ci, cid in enumerate(CLUSTER_IDS):
        A_true = amp_data[ep][cid][s_idx:e_idx]
        A0 = A_true[0]

        # Pre-seizure fit rollout
        _mu_p, _al_p = mu_pre[cid], alpha_pre[cid]
        def sl_pre(t, A, _m=_mu_p, _a=_al_p):
            return _m * A - _a * A**3
        def ev_pre(t, A, _m=_mu_p, _a=_al_p):
            return 50.0 - abs(A[0])
        ev_pre.terminal = True
        sol_pre = solve_ivp(sl_pre, [0, (N_ROLL-1)*DT_A], [A0],
                           t_eval=np.arange(N_ROLL)*DT_A, method="RK45",
                           rtol=1e-8, atol=1e-10, events=ev_pre)
        A_pre = sol_pre.y[0]
        if len(A_pre) < N_ROLL:
            A_pre = np.pad(A_pre, (0, N_ROLL-len(A_pre)),
                          constant_values=A_pre[-1])

        # Seizure fit rollout
        _mu_s, _al_s = mu_fit[cid], alpha_fit[cid]
        def sl_sez(t, A, _m=_mu_s, _a=_al_s):
            return _m * A - _a * A**3
        def ev_sez(t, A, _m=_mu_s, _a=_al_s):
            return 50.0 - abs(A[0])
        ev_sez.terminal = True
        sol_sez = solve_ivp(sl_sez, [0, (N_ROLL-1)*DT_A], [A0],
                           t_eval=np.arange(N_ROLL)*DT_A, method="RK45",
                           rtol=1e-8, atol=1e-10, events=ev_sez)
        A_sez = sol_sez.y[0]
        if len(A_sez) < N_ROLL:
            A_sez = np.pad(A_sez, (0, N_ROLL-len(A_sez)),
                          constant_values=A_sez[-1])

        ax = axes[ci]
        ax.plot(t_roll, A_true, color="k", lw=1.4, label="True")
        ax.plot(t_roll[:len(A_pre)], A_pre, color="steelblue", lw=1.0, ls="--",
                label=f"Pre-sez: $\\mu$={mu_pre[cid]:.3f}")
        ax.plot(t_roll[:len(A_sez)], A_sez, color="tomato", lw=1.0, ls=":",
                label=f"Seizure: $\\mu$={mu_fit[cid]:.3f}")
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6, label="Onset")
        ax.set_ylabel(f"$A_{{{cid}}}(t)$")
        ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"rollout_stuart_landau_comparison_ep{ep}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL ROLLOUT PLOTS SAVED")
print("=" * 70)
print(f"  Model A (Amplitude [9,3]):")
print(f"    rollout_amplitude_ep1/2/3.png — trajectory rollout")
print(f"    onestep_amplitude.png — one-step scatter")
print(f"  Model B (Envelope Phase [15,3]):")
print(f"    rollout_envelope_phase_ep1/2/3.png — trajectory rollout")
print(f"    onestep_envelope_phase.png — one-step scatter")
print(f"  Model C (Stuart-Landau OLS):")
print(f"    rollout_stuart_landau_ep1/2/3.png — seizure-fit rollout")
print(f"    rollout_stuart_landau_comparison_ep1/2/3.png — pre vs seizure fit")
print(f"\n  All in {OUT_DIR}/")
