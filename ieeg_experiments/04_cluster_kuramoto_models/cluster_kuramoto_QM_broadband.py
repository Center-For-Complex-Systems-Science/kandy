#!/usr/bin/env python3
"""KANDy experiment: Cluster-Reduced Kuramoto from PatientQM_02Clean data.

Same pipeline as kandy_cluster_kuramoto.py but with:
  - Data: PatientQM_02Clean.txt (preprocessed, 49971 x 120, 500 Hz)
  - Drop C1 (bulk) → 3 clusters only (C0, C2, C3)
  - Mean-subtract targets: model learns frequency FLUCTUATIONS, not ω≈60 rad/s
  - Include r_k * sin(θ_k - θ_mean) features for amplitude-phase coupling
  - Separate pre-seizure vs seizure fits
  - 50 Hz sampling (DS=10) for plenty of data

Pipeline:
  1. Differential embedding → spectral clustering (k=4, seizure window)
  2. Drop C1 (bulk) → keep C0, C2, C3 (3 SOZ subclusters)
  3. Alpha bandpass (8-13 Hz) → Hilbert → cluster mean phases θ_k, order params r_k
  4. Smooth + downsample to 50 Hz, central-difference derivatives
  5. Subtract per-cluster mean dθ/dt → targets = frequency residuals δω_k
  6. Lift: sin(Δθ), cos(Δθ), r_k, r_k*sin(θ_k - θ_mean) → 15 features
  7. KANDy [15, 3], grid=3, per-window fits (pre-seizure, seizure, full)
  8. Symbolic extraction + rollout + plots

Author: KANDy Researcher Agent
Date: 2026-03-30
"""

import sys
import copy
import numpy as np
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from sklearn.cluster import SpectralClustering
import torch
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- KANDy imports ---
sys.path.insert(0, str(ROOT / "src"))
from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ============================================================
# Parameters
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = ROOT / "data" / "QM" / "PatientQM_02Clean.txt"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
ONSET_TIME = 88.25  # Episode 2 onset from prior work

# Differential embedding
SG_WINDOW = 51
SG_POLY = 5

# Bandpass: broadband (full seizure dynamics)
BAND_LO, BAND_HI = 1.0, 50.0

# Clustering
CLUSTER_K = 4  # initial clustering, then drop bulk
SOZ_CHANNELS = list(range(21, 31))

# Phase smoothing
PHASE_SG_WIN = 51
PHASE_SG_POLY = 3

# Downsample
DS = 10        # 500 Hz → 50 Hz
DT_DS = DS / FS  # 0.02s

# KANDy
GRID = 3
K_SPLINE = 3
STEPS = 200
LAMB = 0.0

# Training windows (relative to onset)
WINDOWS = {
    "full":       (30.0, 97.0),
    "pre_seizure": (30.0, ONSET_TIME),
}

LINEAR_LIB = make_symbolic_lib({
    "x": (lambda x: x, lambda x: x, 1),
    "0": (lambda x: x * 0, lambda x: x * 0, 0),
})


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# STEP 0: Load data
# ============================================================
print("=" * 70)
print("CLUSTER-REDUCED KURAMOTO: PatientQM_02Clean (BROADBAND 1-50 Hz)")
print("=" * 70)

print("\nLoading data...")
X = np.loadtxt(DATA_PATH)
n_samples, n_ch = X.shape
t_axis = np.arange(n_samples) / FS
print(f"  Shape: {X.shape}, duration: {n_samples / FS:.1f}s, channels: {n_ch}")
ONSET_SAMPLE = int(ONSET_TIME * FS)

# ============================================================
# STEP 1: Differential embedding clustering
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Differential embedding clustering (seizure window)")
print("=" * 70)


def compute_embedding(data_segment, fs):
    T, C = data_segment.shape
    dt = 1.0 / fs
    embeddings = {}
    for ch in range(C):
        x = data_segment[:, ch].copy()
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-12:
            sigma = 1.0
        x = (x - mu) / sigma
        x_smooth = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=0)
        x_dot = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=1, delta=dt)
        x_ddot = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=2, delta=dt)
        embeddings[ch] = np.column_stack([x_smooth, x_dot, x_ddot])
    return embeddings


sez_start = ONSET_SAMPLE
sez_end = min(n_samples, sez_start + int(10.0 * FS))
data_sez = X[sez_start:sez_end, :]

embs_sez = compute_embedding(data_sez, FS)
covs_sez = {}
for ch, emb in embs_sez.items():
    C = np.cov(emb, rowvar=False)
    C += 1e-10 * np.eye(3)
    covs_sez[ch] = C

D_sez = np.zeros((n_ch, n_ch))
for i in range(n_ch):
    for j in range(i + 1, n_ch):
        d = np.linalg.norm(covs_sez[i] - covs_sez[j], "fro")
        D_sez[i, j] = d
        D_sez[j, i] = d

sigma_aff = np.median(D_sez[D_sez > 0])
affinity = np.exp(-D_sez ** 2 / (2 * sigma_aff ** 2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

# Relabel: SOZ-dominant cluster → 0
soz_labels = labels[SOZ_CHANNELS]
soz_cluster_orig = np.bincount(soz_labels).argmax()
cluster_map = {soz_cluster_orig: 0}
next_label = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = next_label
        next_label += 1
labels = np.array([cluster_map[l] for l in labels])

clusters_all = {}
for c in range(CLUSTER_K):
    clusters_all[c] = np.where(labels == c)[0].tolist()

CLUSTER_NAMES_ALL = {0: "C0 (SOZ core)", 1: "C1 (bulk)", 2: "C2 (pacemaker)", 3: "C3 (boundary)"}
for c in range(CLUSTER_K):
    n_soz = sum(1 for m in clusters_all[c] if m in SOZ_CHANNELS)
    soz_in = [m for m in clusters_all[c] if m in SOZ_CHANNELS]
    print(f"  {CLUSTER_NAMES_ALL[c]}: {len(clusters_all[c])} ch, "
          f"{n_soz} SOZ {soz_in if soz_in else ''}")

# ============================================================
# STEP 2: Drop C1 (bulk) → 3 SOZ subclusters
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Drop C1 (bulk) → 3 SOZ subclusters")
print("=" * 70)

# Identify bulk cluster (largest)
bulk_id = max(clusters_all, key=lambda c: len(clusters_all[c]))
KEEP_IDS = [c for c in sorted(clusters_all.keys()) if c != bulk_id]
print(f"  Bulk cluster: {CLUSTER_NAMES_ALL[bulk_id]} ({len(clusters_all[bulk_id])} ch) → DROPPED")
print(f"  Keeping: {[CLUSTER_NAMES_ALL[c] for c in KEEP_IDS]}")

clusters = {c: clusters_all[c] for c in KEEP_IDS}
N_CLUSTERS = len(KEEP_IDS)
CLUSTER_NAMES = {c: CLUSTER_NAMES_ALL[c] for c in KEEP_IDS}
CLUSTER_COLORS = {KEEP_IDS[0]: "#1f77b4", KEEP_IDS[1]: "#d62728", KEEP_IDS[2]: "#2ca02c"}

for c in KEEP_IDS:
    print(f"  {CLUSTER_NAMES[c]}: {len(clusters[c])} ch, "
          f"SOZ: {[m for m in clusters[c] if m in SOZ_CHANNELS]}")


# ============================================================
# STEP 3: Phase extraction (alpha band Hilbert)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 3: Phase extraction ({BAND_LO}-{BAND_HI} Hz)")
print("=" * 70)

nyq = FS / 2.0
b, a = butter(4, [BAND_LO / nyq, BAND_HI / nyq], btype="band")
X_filt = filtfilt(b, a, X, axis=0)

phases_all = np.zeros((n_samples, n_ch))
for ch in range(n_ch):
    analytic = hilbert(X_filt[:, ch])
    phases_all[:, ch] = np.angle(analytic)
print(f"  Phases: {phases_all.shape}")


def cluster_order_parameter(phases, ch_idx):
    z = np.exp(1j * phases[:, ch_idx]).mean(axis=1)
    return np.angle(z), np.abs(z)


psi_raw = np.zeros((n_samples, N_CLUSTERS))
r_raw = np.zeros((n_samples, N_CLUSTERS))
for ci, c in enumerate(KEEP_IDS):
    psi_raw[:, ci], r_raw[:, ci] = cluster_order_parameter(phases_all, clusters[c])

# Unwrap + smooth
for ci in range(N_CLUSTERS):
    psi_raw[:, ci] = np.unwrap(psi_raw[:, ci])
psi_smooth = np.zeros_like(psi_raw)
for ci in range(N_CLUSTERS):
    psi_smooth[:, ci] = savgol_filter(psi_raw[:, ci], PHASE_SG_WIN, PHASE_SG_POLY)

# Downsample
psi_ds = psi_smooth[::DS]
r_ds = r_raw[::DS]
t_ds = np.arange(len(psi_ds)) * DT_DS
n_ds = len(psi_ds)

print(f"  Downsampled: {n_ds} samples at {1/DT_DS:.0f} Hz")
for ci, c in enumerate(KEEP_IDS):
    freq = np.diff(psi_ds[:, ci]).mean() / (2 * np.pi * DT_DS)
    print(f"  {CLUSTER_NAMES[c]}: r={r_ds[:, ci].mean():.3f}±{r_ds[:, ci].std():.3f}, "
          f"freq={freq:.1f} Hz")


# ============================================================
# STEP 4: Lift design
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Koopman lift")
print("=" * 70)

PAIRS = [(i, j) for i in range(N_CLUSTERS) for j in range(i + 1, N_CLUSTERS)]
N_PAIRS = len(PAIRS)

FEAT_NAMES = []
# sin(Δθ) for unique pairs (3)
for i, j in PAIRS:
    FEAT_NAMES.append(f"sin(t{KEEP_IDS[i]}-t{KEEP_IDS[j]})")
# cos(Δθ) for unique pairs (3)
for i, j in PAIRS:
    FEAT_NAMES.append(f"cos(t{KEEP_IDS[i]}-t{KEEP_IDS[j]})")
# r_k (3)
for ci, c in enumerate(KEEP_IDS):
    FEAT_NAMES.append(f"r{c}")
# r_k * sin(θ_k - θ_mean) (3) — amplitude-phase coupling
for ci, c in enumerate(KEEP_IDS):
    FEAT_NAMES.append(f"r{c}*sin(t{c}-tm)")
# r_k * cos(θ_k - θ_mean) (3) — amplitude-phase coupling
for ci, c in enumerate(KEEP_IDS):
    FEAT_NAMES.append(f"r{c}*cos(t{c}-tm)")

N_FEAT = len(FEAT_NAMES)
OUT_NAMES = [f"dw{c}" for c in KEEP_IDS]
print(f"  {N_FEAT} features: {FEAT_NAMES}")
print(f"  KAN: [{N_FEAT}, {N_CLUSTERS}]")


def build_lift(psi_arr, r_arr):
    """(T, 3) phases + (T, 3) order params → (T, N_FEAT)."""
    feats = []
    # sin(Δθ)
    for i, j in PAIRS:
        feats.append(np.sin(psi_arr[:, i] - psi_arr[:, j]))
    # cos(Δθ)
    for i, j in PAIRS:
        feats.append(np.cos(psi_arr[:, i] - psi_arr[:, j]))
    # r_k
    for ci in range(N_CLUSTERS):
        feats.append(r_arr[:, ci])
    # r_k * sin(θ_k - θ_mean)
    theta_mean = psi_arr.mean(axis=1)
    for ci in range(N_CLUSTERS):
        feats.append(r_arr[:, ci] * np.sin(psi_arr[:, ci] - theta_mean))
    # r_k * cos(θ_k - θ_mean)
    for ci in range(N_CLUSTERS):
        feats.append(r_arr[:, ci] * np.cos(psi_arr[:, ci] - theta_mean))
    return np.column_stack(feats)


def build_lift_single(psi_vec, r_vec):
    return build_lift(psi_vec[None, :], r_vec[None, :])


# ============================================================
# STEP 5: Per-window KANDy fits
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Per-window KANDy fits (mean-subtracted targets)")
print("=" * 70)

use_pub_style()
window_results = {}

for win_name, (t_start, t_end) in WINDOWS.items():
    print(f"\n{'~' * 60}")
    print(f"  WINDOW: {win_name} [{t_start:.1f}s, {t_end:.1f}s]")
    print(f"{'~' * 60}")

    s_idx = int(t_start / DT_DS)
    e_idx = min(int(t_end / DT_DS), n_ds)

    psi_win = psi_ds[s_idx:e_idx]
    r_win = r_ds[s_idx:e_idx]
    T_win = len(psi_win)

    # Derivatives via central differences
    psi_dot = (psi_win[2:] - psi_win[:-2]) / (2.0 * DT_DS)
    psi_inner = psi_win[1:-1]
    r_inner = r_win[1:-1]
    T_inner = len(psi_inner)

    # MEAN-SUBTRACT targets: remove constant ω per cluster
    omega_mean = psi_dot.mean(axis=0)
    delta_omega = psi_dot - omega_mean  # frequency residuals
    print(f"  Samples: {T_inner}")
    for ci, c in enumerate(KEEP_IDS):
        print(f"    {CLUSTER_NAMES[c]}: ω_mean={omega_mean[ci]:.2f} rad/s "
              f"({omega_mean[ci]/(2*np.pi):.2f} Hz), "
              f"residual std={delta_omega[:, ci].std():.2f} rad/s")

    # Build lift
    Phi = build_lift(psi_inner, r_inner)
    print(f"  Lift: {Phi.shape}, targets: {delta_omega.shape}")
    print(f"  Samples/features ratio: {T_inner/N_FEAT:.0f}:1")

    # Train
    lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT,
                      name=f"kuramoto_{win_name}")
    model = KANDy(lift=lift, grid=GRID, k=K_SPLINE, steps=STEPS,
                  seed=SEED, device="cpu")
    model.fit(X=Phi, X_dot=delta_omega, val_frac=0.15, test_frac=0.15,
              lamb=LAMB, patience=0, verbose=False)

    # One-step R²
    n_test = min(500, T_inner // 5)
    pred = model.predict(Phi[-n_test:])
    true = delta_omega[-n_test:]
    ss_res = np.sum((pred - true) ** 2, axis=0)
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    r2_per = 1 - ss_res / np.maximum(ss_tot, 1e-15)
    r2_overall = 1 - np.sum(ss_res) / np.sum(ss_tot)
    print(f"\n  One-step R² overall: {r2_overall:.4f}")
    for ci, c in enumerate(KEEP_IDS):
        print(f"    {CLUSTER_NAMES[c]}: R² = {r2_per[ci]:.4f}")

    # Rollout (around onset, for full and seizure windows)
    if win_name in ("full", "seizure"):
        roll_start_t = max(ONSET_TIME - 5, t_start + 1)
        roll_end_t = min(ONSET_TIME + 10, t_end - 1)
    else:
        roll_start_t = max(t_end - 15, t_start + 1)
        roll_end_t = t_end - 1

    roll_s = int(roll_start_t / DT_DS) - s_idx - 1  # relative to inner
    roll_e = int(roll_end_t / DT_DS) - s_idx - 1
    roll_s = max(0, roll_s)
    roll_e = min(T_inner, roll_e)
    N_ROLL = roll_e - roll_s

    if N_ROLL > 20:
        psi0 = psi_inner[roll_s].copy()
        r_roll = r_inner[roll_s:roll_e]
        psi_true_roll = psi_inner[roll_s:roll_e]
        t_roll = t_ds[s_idx + 1 + roll_s:s_idx + 1 + roll_e]

        traj = [psi0.copy()]
        state = psi0.copy()
        for step in range(N_ROLL - 1):
            r_now = r_roll[min(step, len(r_roll) - 1)]
            def f(ps, r_v=r_now):
                phi = build_lift_single(ps, r_v)
                pred_delta = model.predict(phi).ravel()
                return pred_delta + omega_mean  # add back mean ω
            k1 = f(state)
            k2 = f(state + 0.5 * DT_DS * k1)
            k3 = f(state + 0.5 * DT_DS * k2)
            k4 = f(state + DT_DS * k3)
            state = state + (DT_DS / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(state.copy())
        traj = np.array(traj)

        rmse_roll = np.sqrt(np.mean((traj - psi_true_roll) ** 2))
        print(f"\n  Rollout [{roll_start_t:.1f}s, {roll_end_t:.1f}s]: "
              f"RMSE={rmse_roll:.4f} ({N_ROLL} steps)")
        for ci, c in enumerate(KEEP_IDS):
            true_f = np.diff(psi_true_roll[:, ci]).mean() / (2*np.pi*DT_DS)
            pred_f = np.diff(traj[:, ci]).mean() / (2*np.pi*DT_DS)
            print(f"    {CLUSTER_NAMES[c]}: true={true_f:.2f} Hz, pred={pred_f:.2f} Hz")

        # Rollout plot
        fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3*N_CLUSTERS), sharex=True)
        fig.suptitle(f"Rollout: {win_name} (R²={r2_overall:.3f}, RMSE={rmse_roll:.3f})",
                     fontsize=12)
        for ci, c in enumerate(KEEP_IDS):
            ax = axes[ci]
            ax.plot(t_roll, psi_true_roll[:, ci], color=CLUSTER_COLORS[c],
                    lw=1.4, label="True")
            ax.plot(t_roll[:len(traj)], traj[:, ci], color=CLUSTER_COLORS[c],
                    lw=1.0, ls="--", label="KANDy")
            ax.axvline(ONSET_TIME, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.set_ylabel(rf"$\theta_{{{c}}}$")
            ax.set_title(CLUSTER_NAMES[c], fontsize=10)
            if ci == 0:
                ax.legend(fontsize=7)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        save_fig(fig, f"rollout_{win_name}")

    # Edge activations
    n_sub = min(3000, int(T_inner * 0.7))
    sub_idx = np.random.choice(int(T_inner * 0.7), n_sub, replace=False)
    train_t = torch.tensor(Phi[sub_idx], dtype=torch.float32)
    try:
        fig_e, _ = plot_all_edges(
            model.model_, X=train_t,
            in_var_names=FEAT_NAMES, out_var_names=OUT_NAMES,
            save=str(OUT_DIR / f"edges_{win_name}"),
        )
        plt.close(fig_e)
    except Exception as e:
        print(f"  [WARN] Edge plot failed: {e}")

    # Loss curves
    if hasattr(model, "train_results_") and model.train_results_ is not None:
        try:
            fig_l, _ = plot_loss_curves(model.train_results_,
                                        save=str(OUT_DIR / f"loss_{win_name}"))
            plt.close(fig_l)
        except Exception as e:
            print(f"  [WARN] Loss plot failed: {e}")

    # model.plot()
    model.model_.save_act = True
    sym_t = torch.tensor(Phi[:min(2048, T_inner)], dtype=torch.float32)
    with torch.no_grad():
        model.model_(sym_t)
    try:
        model.model_.plot(title=f"KAN: {win_name}")
        plt.savefig(str(OUT_DIR / f"model_plot_{win_name}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(str(OUT_DIR / f"model_plot_{win_name}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved model_plot_{win_name}")
    except Exception as e:
        print(f"  [WARN] model.plot failed: {e}")

    # Symbolic extraction (in-place, no deepcopy)
    print(f"  Symbolic extraction...")
    try:
        robust_auto_symbolic(
            model.model_, lib=LINEAR_LIB,
            r2_threshold=0.80, weight_simple=0.8,
            topk_edges=N_FEAT * N_CLUSTERS,
        )
        exprs, vars_ = model.model_.symbolic_formula()
        sub_map = {sp.Symbol(str(v)): sp.Symbol(n)
                   for v, n in zip(vars_, FEAT_NAMES)}

        print(f"\n  ┌─────────────────────────────────────────────")
        print(f"  │ EQUATIONS: {win_name} (R²={r2_overall:.4f})")
        print(f"  │ (targets = dθ/dt − ω_mean, i.e. frequency residuals)")
        print(f"  ├─────────────────────────────────────────────")
        formulas = []
        for ci, c in enumerate(KEEP_IDS):
            sym = sp.sympify(exprs[ci]).xreplace(sub_map)
            sym = sp.expand(sym).xreplace(
                {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
            )
            formulas.append(sym)
            omega_hz = omega_mean[ci] / (2 * np.pi)
            print(f"  │  dθ_{c}/dt = {omega_mean[ci]:.2f} + ({sym})")
            print(f"  │           = {omega_hz:.2f} Hz + residual")
        print(f"  └─────────────────────────────────────────────\n")
    except Exception as e:
        print(f"  [WARN] Symbolic failed: {e}")
        formulas = None

    window_results[win_name] = {
        "model": model,
        "r2_overall": r2_overall,
        "r2_per": r2_per,
        "omega_mean": omega_mean,
        "formulas": formulas,
    }

# One-step scatter (full window)
if "full" in window_results:
    model_full = window_results["full"]["model"]
    s_idx = int(WINDOWS["full"][0] / DT_DS)
    e_idx = min(int(WINDOWS["full"][1] / DT_DS), n_ds)
    psi_w = psi_ds[s_idx:e_idx]
    r_w = r_ds[s_idx:e_idx]
    psi_dot_w = (psi_w[2:] - psi_w[:-2]) / (2.0 * DT_DS)
    omega_m = window_results["full"]["omega_mean"]
    delta_w = psi_dot_w - omega_m
    Phi_w = build_lift(psi_w[1:-1], r_w[1:-1])
    pred_w = model_full.predict(Phi_w)

    fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(4*N_CLUSTERS, 4))
    fig.suptitle("One-Step: True vs Predicted δω (full window)", fontsize=12)
    for ci, c in enumerate(KEEP_IDS):
        ax = axes[ci]
        ax.scatter(delta_w[:, ci], pred_w[:, ci], s=2, alpha=0.3,
                  color=CLUSTER_COLORS[c])
        lims = [min(delta_w[:, ci].min(), pred_w[:, ci].min()),
                max(delta_w[:, ci].max(), pred_w[:, ci].max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_title(f"{CLUSTER_NAMES[c]}\nR²={window_results['full']['r2_per'][ci]:.3f}",
                     fontsize=10)
        ax.set_xlabel(r"True $\delta\omega$")
        if ci == 0:
            ax.set_ylabel(r"Predicted $\delta\omega$")
    fig.tight_layout()
    save_fig(fig, "onestep_full")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'Window':>15s} | {'R² overall':>10s} | " +
      " | ".join([f"R² {CLUSTER_NAMES[c].split()[0]}" for c in KEEP_IDS]))
print("-" * 65)
for win_name in WINDOWS:
    if win_name in window_results:
        res = window_results[win_name]
        r2s = " | ".join([f"{res['r2_per'][ci]:12.4f}" for ci in range(N_CLUSTERS)])
        print(f"{win_name:>15s} | {res['r2_overall']:10.4f} | {r2s}")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
