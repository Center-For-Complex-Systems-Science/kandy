#!/usr/bin/env python3
"""Global SVD on all 120 channels → KANDy dynamics on top K modes.

Instead of cluster-first-then-SVD, do SVD on all 120 channels at once:
  1. All 120 ch → alpha bandpass → Hilbert envelope → log → center/standardize → single SVD
  2. Take top K modes (wherever cumulative variance hits ~90%)
  3. State = [z_0(t), ..., z_K(t)]
  4. Lift: modes + pairwise products z_i*z_j + ReLU gating
  5. KANDy [N_feat, K], per-episode fits on ±15s around onset

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import numpy as np
import torch
import sympy as sp
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ieeg_experiments" / "03_clustering_and_supporting_analysis"))
sys.path.insert(0, str(ROOT / "src"))

from ieeg_utils import (
    load_episodes, save_fig, setup_style,
    FS, ONSET_TIMES, CLUSTER_IDS, CLUSTER_NAMES, CLUSTER_COLORS, ALPHA_BAND,
    CLUSTERS,
)
from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
DS = 100           # 500 Hz -> 5 Hz
DT = DS / FS       # 0.2s
FS_DS = 1.0 / DT   # 5 Hz

SMOOTH_WIN_S = 4.0
SMOOTH_SAMPLES = int(SMOOTH_WIN_S * FS)
smooth_kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES

BAND_LO, BAND_HI = ALPHA_BAND

# SVD
CUMVAR_THRESHOLD = 0.90   # keep modes until 90% variance explained

# Derivatives
SG_WIN = 13
SG_POLY = 4
TRIM = SG_WIN // 2 + 3

# Training window: ±15s around seizure onset
WINDOW_BEFORE = 15.0   # seconds before onset
WINDOW_AFTER = 15.0    # seconds after onset

# KANDy
GRID = 5
K_SPLINE = 3
STEPS = 200
LAMB = 0.0

LINEAR_LIB = make_symbolic_lib({
    "x": (lambda x: x, lambda x: x, 1),
    "0": (lambda x: x * 0, lambda x: x * 0, 0),
})


# ============================================================
# Step 0: Load data
# ============================================================
print("=" * 70)
print("GLOBAL SVD → KANDy: ALL 120 CHANNELS")
print("=" * 70)

episodes = load_episodes()
setup_style()


# ============================================================
# Step 1: Extract alpha-band log-Hilbert envelopes for ALL channels
# ============================================================
print("\nStep 1: Alpha-band envelope extraction (all channels)...")

nyq = FS / 2.0
b_alpha, a_alpha = butter(4, [BAND_LO / nyq, BAND_HI / nyq], btype="band")

envelopes = {}  # envelopes[ep] = (n_ds, n_ch) log-amplitude at 5 Hz
for ep in [1, 2, 3]:
    X_raw = episodes[ep]
    n_samples, n_ch = X_raw.shape
    amps = np.zeros((n_samples, n_ch))

    for ch in range(n_ch):
        sig_filt = filtfilt(b_alpha, a_alpha, X_raw[:, ch])
        analytic = hilbert(sig_filt)
        inst_amp = np.abs(analytic)
        amp_smooth = np.convolve(inst_amp, smooth_kernel, mode="same")
        amps[:, ch] = np.log(amp_smooth + 1.0)

    # Downsample
    envelopes[ep] = amps[::DS]
    print(f"  Episode {ep}: {X_raw.shape} -> envelopes {envelopes[ep].shape} at {FS_DS:.0f} Hz")


# ============================================================
# Step 2: Joint SVD across all 3 episodes
# ============================================================
print("\nStep 2: Joint SVD across all episodes...")

# Truncate to minimum channel count (Ep3 has 119 channels, others 120)
min_ch = min(envelopes[ep].shape[1] for ep in [1, 2, 3])
for ep in [1, 2, 3]:
    envelopes[ep] = envelopes[ep][:, :min_ch]
print(f"  Using {min_ch} channels (min across episodes)")

# Concatenate all episodes for joint SVD
all_env = np.vstack([envelopes[ep] for ep in [1, 2, 3]])
print(f"  Combined: {all_env.shape}")

# Center and standardize per channel
env_mean = all_env.mean(axis=0)
env_std = all_env.std(axis=0)
env_std[env_std < 1e-10] = 1.0
all_centered = (all_env - env_mean) / env_std

# SVD
U, S, Vt = np.linalg.svd(all_centered, full_matrices=False)
cumvar = np.cumsum(S**2) / np.sum(S**2)

# Determine K
K = np.searchsorted(cumvar, CUMVAR_THRESHOLD) + 1
K = max(K, 3)  # at least 3
K = min(K, 5)   # cap at 5 to keep KAN tractable (N_feat = K + K*(K-1)/2 + K)
print(f"  Singular values: {S[:15].round(2)}")
print(f"  Cumulative variance: {cumvar[:15].round(4)}")
print(f"  K = {K} modes (cumvar = {cumvar[K-1]:.4f})")

# Project each episode into mode space
modes = {}  # modes[ep] = (n_ds, K)
boundaries = [0]
for ep in [1, 2, 3]:
    env_c = (envelopes[ep] - env_mean) / env_std
    modes[ep] = env_c @ Vt[:K].T  # (n_ds, K)
    boundaries.append(boundaries[-1] + len(modes[ep]))
    print(f"  Episode {ep}: modes shape {modes[ep].shape}")


# ============================================================
# Step 2b: Analyze spatial loadings — which channels drive each mode
# ============================================================
print("\nStep 2b: Spatial loading analysis...")

# SOZ channels (0-indexed)
SOZ_CHANNELS = set()
for cid in CLUSTERS:
    SOZ_CHANNELS.update(CLUSTERS[cid])

# Plot: SVD spectrum + spatial loadings
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Spectrum
ax = axes[0]
ax.bar(range(min(20, len(S))), S[:20]**2 / np.sum(S**2), color="steelblue",
       edgecolor="k", lw=0.5)
ax.axvline(K - 0.5, color="red", ls="--", lw=1, label=f"K={K}")
ax2 = ax.twinx()
ax2.plot(range(min(20, len(cumvar))), cumvar[:20], "r.-", lw=1)
ax2.axhline(CUMVAR_THRESHOLD, color="red", ls=":", lw=0.5)
ax2.set_ylabel("Cumulative variance", color="red")
ax.set_xlabel("Mode index")
ax.set_ylabel("Fraction of variance")
ax.set_title(f"SVD Spectrum (K={K}, cumvar={cumvar[K-1]:.3f})")
ax.legend()

# Spatial loadings for top K modes
ax = axes[1]
for k in range(min(K, 5)):
    loading = Vt[k, :]
    ax.plot(range(len(loading)), loading, lw=0.8, label=f"Mode {k}")
# Mark SOZ channels
for ch in SOZ_CHANNELS:
    if ch < Vt.shape[1]:
        ax.axvline(ch, color="gray", ls=":", lw=0.3, alpha=0.5)
ax.axvspan(min(SOZ_CHANNELS), max(SOZ_CHANNELS), alpha=0.1, color="red",
           label="SOZ channels")
ax.set_xlabel("Channel index")
ax.set_ylabel("Loading weight")
ax.set_title("Spatial Loadings (V vectors)")
ax.legend(fontsize=7, loc="upper right")

fig.tight_layout()
save_fig(fig, OUT_DIR / "svd_spectrum_loadings")


# ============================================================
# Step 3: Build lift features
# ============================================================
print(f"\nStep 3: Building lift features (K={K} modes)...")

# Feature list:
# - K raw modes: z_0, ..., z_{K-1}
# - K*(K-1)/2 pairwise products: z_i * z_j for i<j
# - K ReLU gating: ReLU(z_k - theta_k)

N_PAIRS = K * (K - 1) // 2
N_FEAT = K + N_PAIRS + K
PAIRS = [(i, j) for i in range(K) for j in range(i + 1, K)]

feat_names = [f"z_{k}" for k in range(K)]
feat_names += [f"z{i}*z{j}" for i, j in PAIRS]
feat_names += [f"ReLU(z_{k})" for k in range(K)]

out_names = [f"dz_{k}/dt" for k in range(K)]

print(f"  Features ({N_FEAT}): {feat_names}")
print(f"  Outputs ({K}): {out_names}")


# ============================================================
# Step 4: Per-episode fits on ±15s around onset
# ============================================================
print(f"\nStep 4: Per-episode KANDy fits (±{WINDOW_BEFORE}s around onset)")
print("=" * 70)

ep_results = {}

for ep in [1, 2, 3]:
    onset = ONSET_TIMES[ep]
    n_ds = len(modes[ep])
    t = np.arange(n_ds) * DT

    print(f"\n{'~' * 60}")
    print(f"  EPISODE {ep} (onset = {onset}s)")
    print(f"{'~' * 60}")

    # Training window
    t_start = onset - WINDOW_BEFORE
    t_end = onset + WINDOW_AFTER
    s_idx = max(int(t_start / DT), TRIM)
    e_idx = min(int(t_end / DT), n_ds - TRIM)

    z_win = modes[ep][s_idx:e_idx]  # (T_win, K)
    t_win = t[s_idx:e_idx]
    T_win = len(z_win)
    print(f"  Window: [{t_win[0]:.1f}s, {t_win[-1]:.1f}s], {T_win} samples")

    # Compute derivatives
    z_dot = np.zeros_like(z_win)
    for k in range(K):
        z_dot[:, k] = savgol_filter(z_win[:, k], SG_WIN, SG_POLY,
                                    deriv=1, delta=DT)

    # Trim SG boundaries
    z_inner = z_win[TRIM:-TRIM]
    z_dot_inner = z_dot[TRIM:-TRIM]
    t_inner = t_win[TRIM:-TRIM]
    T_inner = len(z_inner)

    # ReLU thresholds: pre-seizure mean amplitude per mode
    pre_mask = t_inner < onset
    thresholds = np.zeros(K)
    if pre_mask.sum() > 5:
        thresholds = z_inner[pre_mask].mean(axis=0)
    print(f"  ReLU thresholds: {thresholds.round(3)}")

    # Build features
    feats = []
    # Raw modes
    for k in range(K):
        feats.append(z_inner[:, k])
    # Pairwise products
    for i, j in PAIRS:
        feats.append(z_inner[:, i] * z_inner[:, j])
    # ReLU gating
    for k in range(K):
        feats.append(np.maximum(z_inner[:, k] - thresholds[k], 0.0))

    Phi = np.column_stack(feats)  # (T_inner, N_FEAT)
    targets = z_dot_inner          # (T_inner, K)

    print(f"  Training data: Phi={Phi.shape}, targets={targets.shape}")

    # Check for degenerate data
    feat_stds = Phi.std(axis=0)
    dead_feats = np.sum(feat_stds < 1e-8)
    if dead_feats > 0:
        print(f"  WARNING: {dead_feats} features have near-zero variance")

    # --- Train KANDy ---
    lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT,
                      name=f"global_svd_ep{ep}")
    model = KANDy(lift=lift, grid=GRID, k=K_SPLINE, steps=STEPS,
                  seed=SEED, device="cpu")

    print(f"  Training KAN [{N_FEAT}, {K}], grid={GRID}, steps={STEPS}...")
    model.fit(X=Phi, X_dot=targets, val_frac=0.15, test_frac=0.15,
              lamb=LAMB, patience=0, verbose=False)

    # --- One-step R² ---
    n_test = min(200, T_inner // 5)
    pred_os = model.predict(Phi[-n_test:])
    true_os = targets[-n_test:]
    ss_res = np.sum((pred_os - true_os)**2, axis=0)
    ss_tot = np.sum((true_os - true_os.mean(axis=0))**2, axis=0)
    r2_per_mode = 1 - ss_res / np.maximum(ss_tot, 1e-15)
    r2_overall = 1 - np.sum(ss_res) / np.sum(ss_tot)

    print(f"\n  One-step R² overall: {r2_overall:.4f}")
    for k in range(K):
        print(f"    Mode {k}: R² = {r2_per_mode[k]:.4f}")

    # --- Rollout (RK4) ---
    print(f"  Rollout...")

    def build_single_lift(z_vec, thresholds_=thresholds):
        feats = list(z_vec)
        for i, j in PAIRS:
            feats.append(z_vec[i] * z_vec[j])
        for k in range(K):
            feats.append(max(z_vec[k] - thresholds_[k], 0.0))
        return np.array(feats)[None, :]

    # Start rollout from beginning of window
    z0 = z_inner[0].copy()
    N_ROLL = T_inner
    traj = [z0.copy()]
    state = z0.copy()
    for step in range(N_ROLL - 1):
        def f(s):
            return model.predict(build_single_lift(s)).ravel()
        k1 = f(state)
        k2 = f(state + 0.5 * DT * k1)
        k3 = f(state + 0.5 * DT * k2)
        k4 = f(state + DT * k3)
        state = state + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state = np.clip(state, -30, 30)
        traj.append(state.copy())
    traj = np.array(traj)

    rmse_roll = np.sqrt(np.mean((traj - z_inner)**2))
    z_range = z_inner.max() - z_inner.min()
    nrmse_roll = rmse_roll / max(z_range, 1e-8)
    print(f"  Rollout RMSE: {rmse_roll:.4f}, NRMSE: {nrmse_roll:.4f}")

    per_mode_rmse = np.sqrt(np.mean((traj - z_inner)**2, axis=0))
    for k in range(K):
        r = z_inner[:, k].max() - z_inner[:, k].min()
        print(f"    Mode {k}: RMSE={per_mode_rmse[k]:.4f}, "
              f"NRMSE={per_mode_rmse[k]/max(r, 1e-8):.4f}")

    # --- Plot: Rollout trajectories ---
    n_plot = min(K, 6)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    fig.suptitle(f"Episode {ep}: Global SVD Rollout [{N_FEAT},{K}]\n"
                 f"R²={r2_overall:.3f}, Rollout NRMSE={nrmse_roll:.3f}",
                 fontsize=12)
    colors = plt.cm.tab10.colors
    for k in range(n_plot):
        ax = axes[k]
        ax.plot(t_inner, z_inner[:, k], color=colors[k], lw=1.4, label="True")
        ax.plot(t_inner, traj[:, k], color=colors[k], lw=1.0, ls="--",
                label="KANDy")
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6, label="Onset")
        ax.set_ylabel(f"$z_{{{k}}}$")
        ax.set_title(f"Mode {k} (R²={r2_per_mode[k]:.3f})", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"rollout_ep{ep}")

    # --- Plot: One-step scatter ---
    all_pred = model.predict(Phi)
    n_plot_os = min(K, 6)
    fig, axes = plt.subplots(1, n_plot_os, figsize=(3 * n_plot_os, 3))
    if n_plot_os == 1:
        axes = [axes]
    fig.suptitle(f"Episode {ep}: One-Step Prediction", fontsize=12)
    for k in range(n_plot_os):
        ax = axes[k]
        ax.scatter(targets[:, k], all_pred[:, k], s=3, alpha=0.4,
                  color=colors[k])
        lims = [min(targets[:, k].min(), all_pred[:, k].min()),
                max(targets[:, k].max(), all_pred[:, k].max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_title(f"Mode {k}\nR²={r2_per_mode[k]:.3f}", fontsize=9)
        ax.set_xlabel("True")
        if k == 0:
            ax.set_ylabel("Predicted")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / f"onestep_ep{ep}")

    # --- Edge activations ---
    n_sub = min(2000, int(T_inner * 0.7))
    sub_idx = np.random.choice(int(T_inner * 0.7), n_sub, replace=False)
    train_t = torch.tensor(Phi[sub_idx], dtype=torch.float32)
    try:
        fig_e, _ = plot_all_edges(
            model.model_, X=train_t,
            in_var_names=feat_names, out_var_names=out_names,
            save=str(OUT_DIR / f"edge_activations_ep{ep}"),
        )
        plt.close(fig_e)
    except Exception as e:
        print(f"  [WARN] Edge plot failed: {e}")

    # --- Loss curves ---
    if hasattr(model, "train_results_") and model.train_results_ is not None:
        try:
            fig_l, _ = plot_loss_curves(
                model.train_results_,
                save=str(OUT_DIR / f"loss_curves_ep{ep}"),
            )
            plt.close(fig_l)
        except Exception as e:
            print(f"  [WARN] Loss plot failed: {e}")

    # --- model.plot() ---
    print(f"  Generating model.plot()...")
    model.model_.save_act = True
    sym_t = torch.tensor(Phi[:min(2048, T_inner)], dtype=torch.float32)
    with torch.no_grad():
        model.model_(sym_t)
    try:
        in_vars = [rf"${n}$" for n in feat_names]
        out_vars = [rf"${n}$" for n in out_names]
        model.model_.plot(in_vars=in_vars, out_vars=out_vars,
                         title=f"Episode {ep}")
        plt.savefig(str(OUT_DIR / f"model_plot_ep{ep}.png"),
                    dpi=300, bbox_inches="tight")
        plt.savefig(str(OUT_DIR / f"model_plot_ep{ep}.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved model_plot_ep{ep}.png")
    except Exception as e:
        print(f"    [WARN] model.plot() failed: {e}")

    # --- Symbolic extraction (in-place, no deepcopy — avoids torch tensor issue) ---
    print(f"  Symbolic extraction...")
    model.model_.save_act = True
    with torch.no_grad():
        model.model_(sym_t)
    try:
        robust_auto_symbolic(
            model.model_, lib=LINEAR_LIB,
            r2_threshold=0.80, weight_simple=0.8,
            topk_edges=N_FEAT * K,
        )
        exprs, vars_ = model.model_.symbolic_formula()
        sub_map = {sp.Symbol(str(v)): sp.Symbol(n)
                   for v, n in zip(vars_, feat_names)}

        print(f"\n  ┌─────────────────────────────────────────────")
        print(f"  │ DISCOVERED EQUATIONS: Episode {ep}")
        print(f"  │ (R² = {r2_overall:.4f}, NRMSE = {nrmse_roll:.4f})")
        print(f"  ├─────────────────────────────────────────────")
        formulas = []
        for k in range(K):
            sym = sp.sympify(exprs[k]).xreplace(sub_map)
            sym = sp.expand(sym).xreplace(
                {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
            )
            formulas.append(sym)
            print(f"  │  dz_{k}/dt = {sym}")
        print(f"  └─────────────────────────────────────────────\n")
    except Exception as e:
        print(f"    [WARN] Symbolic extraction failed: {e}")
        formulas = None

    ep_results[ep] = {
        "model": model,
        "r2_overall": r2_overall,
        "r2_per_mode": r2_per_mode,
        "rmse_roll": rmse_roll,
        "nrmse_roll": nrmse_roll,
        "formulas": formulas,
        "thresholds": thresholds,
    }


# ============================================================
# Step 5: Cross-episode comparison
# ============================================================
print("\n" + "=" * 70)
print("CROSS-EPISODE COMPARISON")
print("=" * 70)

# Summary table
print(f"\n{'Ep':>4s} | {'R² overall':>10s} | {'NRMSE':>8s} | " +
      " | ".join([f"R² z{k}" for k in range(min(K, 6))]))
print("-" * (30 + 10 * min(K, 6)))
for ep in [1, 2, 3]:
    res = ep_results[ep]
    r2s = " | ".join([f"{res['r2_per_mode'][k]:6.3f}" for k in range(min(K, 6))])
    print(f"  {ep:2d} | {res['r2_overall']:10.4f} | {res['nrmse_roll']:8.4f} | {r2s}")

# Plot: comparison of mode dynamics across episodes
n_plot = min(K, 4)
fig, axes = plt.subplots(n_plot, 3, figsize=(15, 3 * n_plot), sharex=False)
fig.suptitle("Global SVD Modes: All Episodes", fontsize=13)
colors = plt.cm.tab10.colors
for ep_idx, ep in enumerate([1, 2, 3]):
    onset = ONSET_TIMES[ep]
    n_ds = len(modes[ep])
    t = np.arange(n_ds) * DT
    for k in range(n_plot):
        ax = axes[k, ep_idx] if n_plot > 1 else axes[ep_idx]
        ax.plot(t, modes[ep][:, k], color=colors[k], lw=0.8)
        ax.axvline(onset, color="k", ls=":", lw=0.8, alpha=0.6)
        if ep_idx == 0:
            ax.set_ylabel(f"$z_{{{k}}}$")
        if k == 0:
            ax.set_title(f"Episode {ep}", fontsize=10)
        if k == n_plot - 1:
            ax.set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, OUT_DIR / "mode_traces_all_episodes")

# Plot: spatial loading map — which channels contribute to each mode
fig, axes = plt.subplots(min(K, 5), 1, figsize=(14, 2.5 * min(K, 5)), sharex=True)
if min(K, 5) == 1:
    axes = [axes]
fig.suptitle("SVD Spatial Loadings (which channels drive each mode)", fontsize=13)
n_ch = Vt.shape[1]
for k in range(min(K, 5)):
    ax = axes[k]
    loading = np.abs(Vt[k, :n_ch])
    bar_colors = ["red" if ch in SOZ_CHANNELS else "steelblue"
                  for ch in range(n_ch)]
    ax.bar(range(n_ch), loading, color=bar_colors, width=1.0, edgecolor="none")
    ax.set_ylabel(f"Mode {k}\n|loading|")
    ax.set_xlim(-0.5, n_ch - 0.5)
# Legend
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="red", label="SOZ"),
                        Patch(color="steelblue", label="Non-SOZ")],
               fontsize=8, loc="upper right")
axes[-1].set_xlabel("Channel index")
fig.tight_layout()
save_fig(fig, OUT_DIR / "spatial_loadings")


# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  SVD: K={K} modes, cumvar={cumvar[K-1]:.4f}")
print(f"  Lift: {N_FEAT} features = {K} modes + {N_PAIRS} products + {K} ReLU")
print(f"  Window: ±{WINDOW_BEFORE}s around onset")
print(f"  Per-episode fits:")
for ep in [1, 2, 3]:
    res = ep_results[ep]
    print(f"    Ep{ep}: R²={res['r2_overall']:.4f}, "
          f"NRMSE={res['nrmse_roll']:.4f}")

best_ep = max(ep_results, key=lambda e: ep_results[e]["r2_overall"])
print(f"\n  Best episode: {best_ep} "
      f"(R²={ep_results[best_ep]['r2_overall']:.4f})")

print(f"\n  All plots saved to {OUT_DIR}/")
print("  Done.")
