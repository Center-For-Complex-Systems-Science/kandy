#!/usr/bin/env python3
"""KANDy experiment: 3-Oscillator Cluster-Reduced Dynamics with ReLU-Gated
Switching and Multi-Frequency Band Analysis.

Combines the 3-cluster structure (SOZ core, pacemaker, boundary) with:
1. Per-cluster SVD amplitude envelopes (not raw phases or order parameters)
2. ReLU-gated features for seizure-onset switching
3. Multi-frequency band analysis (5 bands)
4. Inter-cluster amplitude coupling

Previous failed approaches:
- Phase-based Kuramoto (alpha band): constant rotation dominates, NRMSE=0.90
- Order parameter dynamics: derivatives too noisy (SNR~0.02), R^2 negative

This approach models the AMPLITUDE ENVELOPE dynamics following the successful
ieeg_relu_gated.py pattern, but extended to 3 clusters. For each cluster:
  x_k(t) = Mode 0 of SVD(log(Hilbert_envelope)) within the cluster
  x_k_dot(t) computed via Savitzky-Golay

Lift (per band, 9 features):
  x_0, x_2, x_3                — 3 cluster amplitudes
  x_0*x_2, x_0*x_3, x_2*x_3   — 3 inter-cluster coupling products
  ReLU(x_0-th), ReLU(x_2-th),
    ReLU(x_3-th)               — 3 ReLU gating

KAN: [9, 3], grid=3, predicting [dx_0/dt, dx_2/dt, dx_3/dt]

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import os
import sys
import copy
import io
import contextlib
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
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
from kandy.plotting import (
    get_edge_activation,
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ============================================================
# Parameters
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = ROOT / "data" / "E3Data.mat"
OUT_DIR = ROOT / "results" / "iEEG" / "kuramoto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500  # Hz
ONSET_TIMES = {1: 80.25, 2: 88.25, 3: 87.00}

# Clusters: drop C1 (bulk, 109 channels, low coherence)
# Keep 3 SOZ subclusters
CLUSTERS = {
    0: [12, 15, 24, 28, 29, 30, 33],  # C0: SOZ core (7 ch)
    2: [25, 26],                        # C2: pacemaker pair (2 ch)
    3: [23, 27],                        # C3: boundary pair (2 ch)
}
CLUSTER_IDS = sorted(CLUSTERS.keys())  # [0, 2, 3]
N_CLUSTERS = len(CLUSTER_IDS)  # 3
CLUSTER_NAMES = {
    0: "C0 (SOZ core)",
    2: "C2 (pacemaker)",
    3: "C3 (boundary)",
}

# Frequency bands
BANDS = {
    "delta":     (1.0,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "beta":      (13.0, 30.0),
    "low_gamma": (30.0, 50.0),
}

# Amplitude envelope: Hilbert + smoothing + log + SVD (like ieeg_relu_gated.py)
SMOOTH_WIN_S = 4.0       # 4s running average for envelope
DS = 100                  # 500 Hz -> 5 Hz (same as ieeg_relu_gated.py)
DT_DS = DS / FS           # 0.2 s
FS_DS = 1.0 / DT_DS      # 5 Hz

# Training: full episode (30-97s as in ieeg_relu_gated.py)
TRAIN_START_T = 30.0
TRAIN_END_T = 97.0

# KANDy hyperparameters
GRID = 3
K_SPLINE = 3
STEPS = 200
LAMB = 0.0

# Savitzky-Golay derivatives (at 5 Hz)
SG_DERIV_WIN = 13
SG_DERIV_POLY = 4

# Gate ramp duration (seconds)
GATE_RAMP_S = 5.0


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# STEP 0: Load data
# ============================================================
print("=" * 70)
print("3-CLUSTER AMPLITUDE DYNAMICS + ReLU GATING (MULTI-FREQUENCY)")
print("=" * 70)

print("\nLoading data...")
mat = sio.loadmat(DATA_PATH)
X1 = mat["X1"].astype(np.float64)  # (49972, 120) Episode 1
X2 = mat["X2"].astype(np.float64)  # (49971, 120) Episode 2
X3 = mat["X3"].astype(np.float64)  # (49971, 119) Episode 3
print(f"  X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}")

EPISODES = {1: X1, 2: X2, 3: X3}


# ============================================================
# STEP 1: Multi-band amplitude extraction per cluster (SVD mode 0)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Multi-band amplitude extraction (Hilbert + smoothing + SVD)")
print("=" * 70)

SMOOTH_SAMPLES = int(SMOOTH_WIN_S * FS)
smooth_kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES


def extract_cluster_amplitudes(X_raw, clusters, band_lo, band_hi, fs):
    """Extract per-cluster SVD mode 0 of log(Hilbert envelope) for one band.

    Pipeline per channel:
      bandpass -> Hilbert -> |analytic| -> 4s running avg -> log(A+1)
    Then per cluster: SVD mode 0 of the (T, n_ch_in_cluster) matrix.

    Returns
    -------
    modes : dict {cluster_id: (T,) amplitude mode 0}
    """
    n_samples, n_ch = X_raw.shape
    nyq = fs / 2.0
    lo = max(band_lo / nyq, 0.001)
    hi = min(band_hi / nyq, 0.999)
    b, a = butter(4, [lo, hi], btype="band")

    modes = {}
    for cid in sorted(clusters.keys()):
        ch_list = [ch for ch in clusters[cid] if ch < n_ch]
        if len(ch_list) == 0:
            modes[cid] = np.zeros(n_samples)
            continue

        amps = np.zeros((n_samples, len(ch_list)))
        for j, ch in enumerate(ch_list):
            sig_filt = filtfilt(b, a, X_raw[:, ch])
            analytic = hilbert(sig_filt)
            inst_amp = np.abs(analytic)
            amp_smooth = np.convolve(inst_amp, smooth_kernel, mode="same")
            amps[:, j] = np.log(amp_smooth + 1.0)

        # SVD: take mode 0
        amp_mean = amps.mean(axis=0)
        amp_std = amps.std(axis=0)
        amp_std[amp_std < 1e-10] = 1.0
        amps_centered = (amps - amp_mean) / amp_std

        if amps_centered.shape[1] == 1:
            # Only 1 channel effectively — just use centered signal
            modes[cid] = amps_centered[:, 0]
        else:
            U, S, Vt = np.linalg.svd(amps_centered, full_matrices=False)
            cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
            modes[cid] = U[:, 0] * S[0]  # mode 0 with amplitude

    return modes


print(f"Extracting per-cluster amplitude mode 0...")
print(f"Smoothing: {SMOOTH_WIN_S}s running average")
print(f"Downsample: {FS} Hz -> {FS_DS:.0f} Hz (DS={DS})")

# data_amp[band][episode] = {cluster_id: (T_ds,) mode 0 amplitude, downsampled}
data_amp = {}
for band_name, (blo, bhi) in BANDS.items():
    data_amp[band_name] = {}
    for ep_num, X_raw in EPISODES.items():
        modes = extract_cluster_amplitudes(X_raw, CLUSTERS, blo, bhi, FS)
        # Downsample
        modes_ds = {cid: m[::DS] for cid, m in modes.items()}
        data_amp[band_name][ep_num] = modes_ds

    # Summary
    ep1_modes = data_amp[band_name][1]
    print(f"\n  {band_name:12s} ({blo:.0f}-{bhi:.0f} Hz):")
    for cid in CLUSTER_IDS:
        m = ep1_modes[cid]
        print(f"    {CLUSTER_NAMES[cid]:20s}: mean={m.mean():.4f}, "
              f"std={m.std():.4f}, range=[{m.min():.4f}, {m.max():.4f}]")


# ============================================================
# STEP 2: Build training data for each band
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Build training data (all 3 episodes, per band)")
print("=" * 70)


def estimate_theta(data_amp_band, onset_times, pre_window_s=10.0):
    """Compute per-cluster ReLU threshold from pre-seizure amplitude."""
    thresholds = {}
    for cid in CLUSTER_IDS:
        vals = []
        for ep_num, onset_s in onset_times.items():
            modes_ds = data_amp_band[ep_num]
            t_ds = np.arange(len(modes_ds[cid])) * DT_DS
            pre_mask = (t_ds >= onset_s - pre_window_s) & (t_ds < onset_s)
            if pre_mask.sum() > 0:
                vals.append(modes_ds[cid][pre_mask].mean())
        thresholds[cid] = np.mean(vals) if vals else 0.0
    return thresholds


def build_training_data_amplitude(data_amp_band, onset_times):
    """Build lift features and targets for one frequency band.

    Lift: [x_0, x_2, x_3, x_0*x_2, x_0*x_3, x_2*x_3,
           ReLU(x_0-th), ReLU(x_2-th), ReLU(x_3-th)]

    Returns: Phi, x_dot, x_inner, ep_boundaries, thresholds
    """
    thresholds = estimate_theta(data_amp_band, onset_times)

    Phi_parts = []
    x_dot_parts = []
    x_inner_parts = []
    ep_boundaries = [0]

    for ep_num in sorted(data_amp_band.keys()):
        modes_ds = data_amp_band[ep_num]
        n_ds = len(modes_ds[CLUSTER_IDS[0]])
        t_ds = np.arange(n_ds) * DT_DS

        # Training window
        train_s = int(TRAIN_START_T / DT_DS)
        train_e = min(int(TRAIN_END_T / DT_DS), n_ds)

        # Stack cluster modes: (T, 3)
        x_all = np.column_stack([modes_ds[cid][train_s:train_e]
                                 for cid in CLUSTER_IDS])
        t_win = t_ds[train_s:train_e]

        if len(x_all) < 20:
            continue

        # Compute derivatives via SG
        x_dot = np.zeros_like(x_all)
        for ci in range(N_CLUSTERS):
            x_dot[:, ci] = savgol_filter(
                x_all[:, ci], SG_DERIV_WIN, SG_DERIV_POLY,
                deriv=1, delta=DT_DS)

        # Trim SG boundary
        trim = SG_DERIV_WIN // 2 + 3
        valid = slice(trim, len(x_all) - trim)

        x_v = x_all[valid]
        x_dot_v = x_dot[valid]

        # Build lift features
        feats = []

        # Raw cluster amplitudes (3)
        for ci in range(N_CLUSTERS):
            feats.append(x_v[:, ci])

        # Inter-cluster coupling products (3 pairs)
        pair_indices = [(0, 1), (0, 2), (1, 2)]  # indices into N_CLUSTERS
        for (i, j) in pair_indices:
            feats.append(x_v[:, i] * x_v[:, j])

        # ReLU gating (3)
        for ci, cid in enumerate(CLUSTER_IDS):
            feats.append(np.maximum(x_v[:, ci] - thresholds[cid], 0.0))

        Phi = np.column_stack(feats)
        Phi_parts.append(Phi)
        x_dot_parts.append(x_dot_v)
        x_inner_parts.append(x_v)
        ep_boundaries.append(ep_boundaries[-1] + len(Phi))

    Phi_all = np.vstack(Phi_parts)
    x_dot_all = np.vstack(x_dot_parts)
    x_inner_all = np.vstack(x_inner_parts)

    return Phi_all, x_dot_all, x_inner_all, ep_boundaries, thresholds


# Feature names
FEAT_NAMES = [f"x_{cid}" for cid in CLUSTER_IDS]
pair_labels = [(CLUSTER_IDS[0], CLUSTER_IDS[1]),
               (CLUSTER_IDS[0], CLUSTER_IDS[2]),
               (CLUSTER_IDS[1], CLUSTER_IDS[2])]
FEAT_NAMES += [f"x{ci}*x{cj}" for (ci, cj) in pair_labels]
FEAT_NAMES += [f"ReLU(x{cid})" for cid in CLUSTER_IDS]

N_FEAT = len(FEAT_NAMES)
print(f"  Features ({N_FEAT}): {FEAT_NAMES}")
print(f"  KAN: [{N_FEAT}, {N_CLUSTERS}]")
print(f"  Training window: {TRAIN_START_T}-{TRAIN_END_T}s")

# Build data for each band
training_data = {}
for band_name in BANDS:
    Phi, x_dot, x_inner, ep_bounds, thresholds = build_training_data_amplitude(
        data_amp[band_name], ONSET_TIMES)
    training_data[band_name] = {
        "Phi": Phi,
        "x_dot": x_dot,
        "x_inner": x_inner,
        "ep_bounds": ep_bounds,
        "thresholds": thresholds,
    }
    n_samp = len(Phi)
    print(f"\n  {band_name:12s}: {n_samp} samples, "
          f"ratio={n_samp/N_FEAT:.0f}:1")
    th = thresholds
    print(f"    ReLU thresholds: {[f'{th[cid]:.4f}' for cid in CLUSTER_IDS]}")
    for ci, cid in enumerate(CLUSTER_IDS):
        dx = x_dot[:, ci]
        print(f"    dx_{cid}/dt: mean={dx.mean():.6f}, std={dx.std():.6f}")


# ============================================================
# STEP 3: Train KANDy per band (Stage 1: find the right band)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Per-band KANDy training")
print("=" * 70)

band_results = {}

for band_name, (blo, bhi) in BANDS.items():
    print(f"\n{'~'*50}")
    print(f"  Band: {band_name} ({blo:.0f}-{bhi:.0f} Hz)")
    print(f"{'~'*50}")

    td = training_data[band_name]
    Phi = td["Phi"]
    x_dot = td["x_dot"]
    n_samp = len(Phi)

    if n_samp < 50:
        print(f"  SKIP: too few samples ({n_samp})")
        band_results[band_name] = {"r2_per_cluster": {}, "r2_overall": -1}
        continue

    # Check for NaN
    if np.any(np.isnan(Phi)) or np.any(np.isnan(x_dot)):
        print(f"  SKIP: NaN in data")
        band_results[band_name] = {"r2_per_cluster": {}, "r2_overall": -1}
        continue

    # Feature statistics
    print(f"  Feature stats:")
    for fi, fn in enumerate(FEAT_NAMES):
        col = Phi[:, fi]
        print(f"    {fn:20s}: mean={col.mean():+.4f}, std={col.std():.4f}, "
              f"range=[{col.min():.3f}, {col.max():.3f}]")

    # Train KANDy
    lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="identity")
    model = KANDy(
        lift=lift,
        grid=GRID,
        k=K_SPLINE,
        steps=STEPS,
        seed=SEED,
        device="cpu",
    )

    print(f"  Training KAN [{N_FEAT}, {N_CLUSTERS}], grid={GRID}, "
          f"steps={STEPS}, lamb={LAMB} ...")

    model.fit(
        X=Phi,
        X_dot=x_dot,
        val_frac=0.15,
        test_frac=0.15,
        lamb=LAMB,
        patience=0,
        verbose=True,
    )

    # One-step evaluation
    n_test = min(500, n_samp // 5)
    test_phi = Phi[-n_test:]
    test_dot = x_dot[-n_test:]
    pred_dot = model.predict(test_phi)

    # Per-cluster R^2
    r2_per_cluster = {}
    for ci, cid in enumerate(CLUSTER_IDS):
        ss_res = np.sum((pred_dot[:, ci] - test_dot[:, ci]) ** 2)
        ss_tot = np.sum((test_dot[:, ci] - test_dot[:, ci].mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        r2_per_cluster[cid] = r2

    # Overall R^2
    ss_res_all = np.sum((pred_dot - test_dot) ** 2)
    ss_tot_all = np.sum((test_dot - test_dot.mean(axis=0)) ** 2)
    r2_overall = 1 - ss_res_all / max(ss_tot_all, 1e-12)

    # RMSE
    rmse = np.sqrt(np.mean((pred_dot - test_dot) ** 2))
    nrmse = rmse / np.std(x_dot) if np.std(x_dot) > 1e-12 else float("inf")

    print(f"\n  Results for {band_name}:")
    print(f"    Overall R^2:  {r2_overall:.4f}")
    print(f"    RMSE:         {rmse:.6f}")
    print(f"    NRMSE:        {nrmse:.4f}")
    for cid in CLUSTER_IDS:
        print(f"    {CLUSTER_NAMES[cid]:20s}: R^2 = {r2_per_cluster[cid]:.4f}")

    # Count active edges
    n_sub = min(2000, int(n_samp * 0.7))
    sub_idx = np.random.choice(int(n_samp * 0.7), n_sub, replace=False)
    sym_input = torch.tensor(Phi[sub_idx], dtype=torch.float32)
    model.model_.save_act = True
    with torch.no_grad():
        model.model_(sym_input)

    edge_ranges = {}
    for fi in range(N_FEAT):
        for ci in range(N_CLUSTERS):
            x_a, y_a = get_edge_activation(model.model_, l=0, i=fi, j=ci)
            rng = np.max(y_a) - np.min(y_a)
            edge_ranges[(fi, ci)] = rng

    max_rng = max(edge_ranges.values())
    thr = 0.05 * max_rng if max_rng > 1e-8 else 1e-8
    n_active = sum(1 for r in edge_ranges.values() if r > thr)
    relu_active = []
    for ci in range(N_CLUSTERS):
        # ReLU features are at indices 6, 7, 8 (after 3 amplitudes + 3 products)
        relu_idx = 6 + ci
        relu_active.append(edge_ranges.get((relu_idx, ci), 0) > thr)

    print(f"    Active edges: {n_active}/{N_FEAT * N_CLUSTERS}")
    print(f"    ReLU active: {['ON' if a else 'off' for a in relu_active]}")

    band_results[band_name] = {
        "r2_per_cluster": r2_per_cluster,
        "r2_overall": r2_overall,
        "rmse": rmse,
        "nrmse": nrmse,
        "n_active": n_active,
        "relu_active": relu_active,
        "model": model,
        "sym_input": sym_input,
        "edge_ranges": edge_ranges,
        "train_loss": model.train_results_["train_loss"][-1] if model.train_results_ else None,
    }


# ============================================================
# STEP 4: Band comparison summary
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Band Comparison Summary")
print("=" * 70)

use_pub_style()

print(f"\n{'Band':>12s} | {'R2 overall':>10s} | {'NRMSE':>8s} | {'Active':>8s} | "
      f"{'ReLU':>12s} | {'R2 C0':>8s} | {'R2 C2':>8s} | {'R2 C3':>8s}")
print("-" * 95)

best_band = None
best_r2 = -999

for band_name in BANDS:
    br = band_results[band_name]
    if br["r2_overall"] == -1:
        print(f"{band_name:>12s} | {'SKIP':>10s}")
        continue

    r2o = br["r2_overall"]
    nrmse = br["nrmse"]
    n_act = br["n_active"]
    relu = br["relu_active"]
    relu_str = ",".join(["Y" if a else "n" for a in relu])
    r2c = br["r2_per_cluster"]

    print(f"{band_name:>12s} | {r2o:10.4f} | {nrmse:8.4f} | "
          f"{n_act:>4d}/{N_FEAT*N_CLUSTERS:<3d} | {relu_str:>12s} | "
          f"{r2c.get(0, -1):8.4f} | {r2c.get(2, -1):8.4f} | {r2c.get(3, -1):8.4f}")

    if r2o > best_r2:
        best_r2 = r2o
        best_band = band_name

print(f"\nBest band: {best_band} (R2={best_r2:.4f})")


# ============================================================
# STEP 5: Detailed analysis of best band(s)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 5: Detailed analysis — best band: {best_band}")
print("=" * 70)

# Also analyze any band with R2 > 0.1 (meaningful structure)
good_bands = [bn for bn in BANDS
              if band_results[bn]["r2_overall"] > max(0.1, best_r2 * 0.5)]
if best_band not in good_bands:
    good_bands = [best_band]

print(f"  Bands with meaningful structure: {good_bands}")

for band_name in good_bands:
    br = band_results[band_name]
    model = br["model"]
    sym_input = br["sym_input"]
    td = training_data[band_name]
    Phi = td["Phi"]
    x_dot = td["x_dot"]
    x_inner = td["x_inner"]
    thresholds = td["thresholds"]
    n_samp = len(Phi)

    # ---- Edge activation plots ----
    print(f"\n  Edge activations for {band_name}...")
    try:
        fig, axes = plot_all_edges(
            model.model_,
            X=sym_input,
            in_var_names=FEAT_NAMES,
            out_var_names=[f"dx_{cid}/dt" for cid in CLUSTER_IDS],
            save=str(OUT_DIR / f"multifreq_{band_name}_edges"),
        )
        plt.close(fig)
        print(f"    Saved multifreq_{band_name}_edges")
    except Exception as e:
        print(f"    [WARN] Edge plot failed: {e}")

    # ---- Loss curves ----
    if hasattr(model, "train_results_") and model.train_results_ is not None:
        try:
            fig, ax = plot_loss_curves(
                model.train_results_,
                save=str(OUT_DIR / f"multifreq_{band_name}_loss"),
            )
            plt.close(fig)
        except Exception as e:
            print(f"    [WARN] Loss plot failed: {e}")

    # ---- Edge analysis table ----
    print(f"\n  Edge Analysis ({band_name}):")
    print(f"  {'Edge':>20s} | {'Range':>8s} | {'Active':>6s}")
    print(f"  " + "-" * 42)
    edge_ranges = br["edge_ranges"]
    max_rng = max(edge_ranges.values())
    thr = 0.05 * max_rng if max_rng > 1e-8 else 1e-8
    for fi in range(N_FEAT):
        for ci in range(N_CLUSTERS):
            rng = edge_ranges[(fi, ci)]
            active = "YES" if rng > thr else "---"
            if rng > thr:
                cid = CLUSTER_IDS[ci]
                print(f"  {FEAT_NAMES[fi]:>16s}->dx{cid} | {rng:8.4f} | {active:>6s}")

    # ---- Symbolic extraction ----
    print(f"\n  Symbolic extraction for {band_name}...")
    LINEAR_LIB = make_symbolic_lib({
        "x":   (lambda x: x,           lambda x: x,           1),
        "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
    })

    model_copy = copy.deepcopy(model)
    sym_subset = torch.tensor(Phi[:min(2048, n_samp)], dtype=torch.float32)
    model_copy.model_.save_act = True
    with torch.no_grad():
        model_copy.model_(sym_subset)

    try:
        robust_auto_symbolic(
            model_copy.model_,
            lib=LINEAR_LIB,
            r2_threshold=0.85,
            weight_simple=0.8,
            topk_edges=15,
        )

        exprs, vars_ = model_copy.model_.symbolic_formula()
        sub_map = {sp.Symbol(str(v)): sp.Symbol(n) for v, n in zip(vars_, FEAT_NAMES)}

        COEFF_TOL = 3
        formulas = []
        for expr_str in exprs:
            sym = sp.sympify(expr_str).xreplace(sub_map)
            sym = sp.expand(sym).xreplace(
                {n: round(float(n), COEFF_TOL) for n in sym.atoms(sp.Number)}
            )
            formulas.append(sym)

        print(f"\n  Discovered equations ({band_name}):")
        for ci, cid in enumerate(CLUSTER_IDS):
            print(f"    dx_{cid}/dt = {formulas[ci]}")
    except Exception as e:
        print(f"    [WARN] Symbolic extraction failed: {e}")
        formulas = None

    # ---- Rollout (if R2 is decent) ----
    if br["r2_overall"] > 0.05:
        print(f"\n  Rollout for {band_name}...")

        # Use episode 3 data for rollout (seizure at 87s)
        modes_ep3 = data_amp[band_name][3]
        n_ds_ep3 = len(modes_ep3[CLUSTER_IDS[0]])
        t_ds_ep3 = np.arange(n_ds_ep3) * DT_DS

        rollout_start_t = 80.0
        rollout_end_t = 95.0
        roll_s = int(rollout_start_t / DT_DS)
        roll_e = min(int(rollout_end_t / DT_DS), n_ds_ep3)

        # Stack cluster modes for rollout
        x_true_roll = np.column_stack(
            [modes_ep3[cid][roll_s:roll_e] for cid in CLUSTER_IDS])
        t_roll = t_ds_ep3[roll_s:roll_e]
        N_ROLLOUT = len(x_true_roll)

        if N_ROLLOUT > 10:
            def build_single_lift_amp(x_vec, thresholds_):
                """Build lift for single time step. x_vec: (3,)."""
                feats = list(x_vec)  # 3 amplitudes
                # Cross products
                feats.append(x_vec[0] * x_vec[1])
                feats.append(x_vec[0] * x_vec[2])
                feats.append(x_vec[1] * x_vec[2])
                # ReLU
                for ci, cid in enumerate(CLUSTER_IDS):
                    feats.append(max(x_vec[ci] - thresholds_[cid], 0.0))
                return np.array(feats)[None, :]

            def rollout_amplitude(x0, n_steps, dt, thresholds_):
                """RK4 rollout of dx/dt."""
                x_cur = x0.copy().astype(np.float64)
                traj = [x_cur.copy()]

                for step in range(1, n_steps):
                    def f(x):
                        phi = build_single_lift_amp(x, thresholds_)
                        return model.predict(phi).ravel()

                    k1 = f(x_cur)
                    k2 = f(x_cur + 0.5 * dt * k1)
                    k3 = f(x_cur + 0.5 * dt * k2)
                    k4 = f(x_cur + dt * k3)
                    x_cur = x_cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    traj.append(x_cur.copy())

                return np.array(traj)

            x0 = x_true_roll[0]
            x_pred_roll = rollout_amplitude(
                x0, N_ROLLOUT, DT_DS, thresholds)

            rmse_roll = np.sqrt(np.mean((x_pred_roll - x_true_roll) ** 2))
            x_range = x_true_roll.max() - x_true_roll.min()
            nrmse_roll = rmse_roll / max(x_range, 1e-8)

            print(f"    Rollout RMSE:  {rmse_roll:.6f}")
            print(f"    Rollout NRMSE: {nrmse_roll:.4f}")
            for ci, cid in enumerate(CLUSTER_IDS):
                rmse_c = np.sqrt(np.mean(
                    (x_pred_roll[:, ci] - x_true_roll[:, ci]) ** 2))
                print(f"    {CLUSTER_NAMES[cid]:20s}: RMSE={rmse_c:.6f}")

            # ---- Rollout plot ----
            fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(10, 3 * N_CLUSTERS),
                                     sharex=True)
            colors = ["#2166ac", "#d6604d", "#4dac26"]
            for ci, cid in enumerate(CLUSTER_IDS):
                ax = axes[ci]
                ax.plot(t_roll, x_true_roll[:, ci], color=colors[ci], lw=1.2,
                        label="True $x(t)$")
                ax.plot(t_roll, x_pred_roll[:, ci], color=colors[ci], lw=1.0,
                        ls="--", label="KANDy $x(t)$")
                ax.axvline(x=ONSET_TIMES[3], color="red", ls=":", alpha=0.5,
                           label="Seizure onset")
                ax.set_ylabel(f"$x_{{{cid}}}(t)$")
                ax.set_title(f"{CLUSTER_NAMES[cid]}")
                ax.legend(fontsize=7, loc="upper left")
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Rollout: {band_name} band (RMSE={rmse_roll:.4f})",
                         fontsize=12)
            fig.tight_layout()
            save_fig(fig, f"multifreq_{band_name}_rollout")

            band_results[band_name]["rollout_rmse"] = rmse_roll
            band_results[band_name]["rollout_nrmse"] = nrmse_roll
        else:
            print(f"    SKIP: too few rollout samples ({N_ROLLOUT})")


# ============================================================
# STEP 6: Cross-band comparison plot
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Cross-band comparison plots")
print("=" * 70)

# 6a. Per-band R^2 bar chart
fig, ax = plt.subplots(figsize=(8, 4))
band_names_plot = list(BANDS.keys())
r2_vals = [band_results[bn]["r2_overall"] for bn in band_names_plot]
colors_bar = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

bars = ax.bar(range(len(band_names_plot)), r2_vals, color=colors_bar, edgecolor="k",
              linewidth=0.5)
ax.set_xticks(range(len(band_names_plot)))
ax.set_xticklabels([f"{bn}\n({BANDS[bn][0]:.0f}-{BANDS[bn][1]:.0f} Hz)"
                     for bn in band_names_plot], fontsize=9)
ax.set_ylabel("$R^2$ (one-step)")
ax.set_title("Per-Band One-Step $R^2$ for Cluster Amplitude Dynamics")
ax.axhline(y=0, color="k", lw=0.5)
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
save_fig(fig, "multifreq_band_comparison_r2")

# 6b. Per-band per-cluster R^2 grouped bar chart
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(band_names_plot))
width = 0.25
cluster_colors = ["#2166ac", "#d6604d", "#4dac26"]

for ci, cid in enumerate(CLUSTER_IDS):
    r2_cluster = [band_results[bn]["r2_per_cluster"].get(cid, -1)
                  for bn in band_names_plot]
    ax.bar(x + ci * width, r2_cluster, width, label=CLUSTER_NAMES[cid],
           color=cluster_colors[ci], edgecolor="k", linewidth=0.5)

ax.set_xticks(x + width)
ax.set_xticklabels([f"{bn}\n({BANDS[bn][0]:.0f}-{BANDS[bn][1]:.0f} Hz)"
                     for bn in band_names_plot], fontsize=9)
ax.set_ylabel("$R^2$ (one-step)")
ax.set_title("Per-Cluster $R^2$ by Frequency Band")
ax.legend(fontsize=8)
ax.axhline(y=0, color="k", lw=0.5)
fig.tight_layout()
save_fig(fig, "multifreq_band_cluster_r2")

# 6c. Cluster amplitude time series for best band (all 3 episodes overlaid)
if best_band is not None:
    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS),
                             sharex=True)
    colors_ep = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ci, cid in enumerate(CLUSTER_IDS):
        ax = axes[ci]
        for ep_num in sorted(EPISODES.keys()):
            modes_ds = data_amp[best_band][ep_num]
            x_sig = modes_ds[cid]
            t_ds = np.arange(len(x_sig)) * DT_DS
            ax.plot(t_ds, x_sig, color=colors_ep[ep_num - 1], lw=0.8,
                    label=f"Ep{ep_num}", alpha=0.8)
            ax.axvline(x=ONSET_TIMES[ep_num], color=colors_ep[ep_num - 1],
                       ls=":", alpha=0.5)

        ax.set_ylabel(f"$x_{{{cid}}}(t)$")
        ax.set_title(f"{CLUSTER_NAMES[cid]} -- {best_band} band")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Cluster Amplitude Mode 0: {best_band} band (3 episodes)",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, f"multifreq_{best_band}_amplitudes")

# 6d. ReLU feature activation visualization for best band
if best_band is not None:
    td = training_data[best_band]
    thresholds = td["thresholds"]

    fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(12, 3 * N_CLUSTERS),
                             sharex=True)

    # Use episode 3 data
    modes_ep3 = data_amp[best_band][3]
    for ci, cid in enumerate(CLUSTER_IDS):
        ax = axes[ci]
        x_sig = modes_ep3[cid]
        t_ds = np.arange(len(x_sig)) * DT_DS
        relu_val = np.maximum(x_sig - thresholds[cid], 0.0)
        ax.plot(t_ds, x_sig, color="steelblue", lw=0.8,
                label=f"$x_{{{cid}}}(t)$")
        ax.fill_between(t_ds, 0, relu_val, color="tomato", alpha=0.3,
                        label=f"ReLU($x_{{{cid}}} - \\theta$)")
        ax.axhline(y=thresholds[cid], color="gray", ls="--", lw=0.8,
                   label=f"$\\theta={thresholds[cid]:.3f}$")
        ax.axvline(x=ONSET_TIMES[3], color="red", ls=":", alpha=0.5,
                   label="Seizure onset")
        ax.set_ylabel(f"$x_{{{cid}}}(t)$")
        ax.set_title(f"{CLUSTER_NAMES[cid]}")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ReLU Gating: {best_band} band (Episode 3)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, f"multifreq_{best_band}_relu_gating")


# ============================================================
# STEP 7: Final summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n{'Band':>12s} | {'R2':>8s} | {'NRMSE':>8s} | {'Active':>8s} | "
      f"{'ReLU':>12s} | {'Roll RMSE':>10s}")
print("-" * 75)

for band_name in BANDS:
    br = band_results[band_name]
    if br["r2_overall"] == -1:
        print(f"{band_name:>12s} | {'SKIP':>8s}")
        continue

    r2o = br["r2_overall"]
    nrmse = br["nrmse"]
    n_act = br["n_active"]
    relu = br["relu_active"]
    relu_str = ",".join(["Y" if a else "n" for a in relu])
    roll_rmse = br.get("rollout_rmse", None)
    roll_str = f"{roll_rmse:.6f}" if roll_rmse is not None else "N/A"

    print(f"{band_name:>12s} | {r2o:8.4f} | {nrmse:8.4f} | "
          f"{n_act:>4d}/{N_FEAT*N_CLUSTERS:<3d} | {relu_str:>12s} | {roll_str:>10s}")

print(f"\nBest band: {best_band}")
print(f"Best R^2: {best_r2:.4f}")

# Check if any ReLU features were active in the best band
if best_band and any(band_results[best_band].get("relu_active", [])):
    print("\nReLU SWITCHING DETECTED: Seizure-onset coherence gating is active!")
else:
    print("\nNo ReLU switching detected in best band. "
          "Coherence changes may be gradual, not threshold-like.")

print(f"\nAll outputs saved to: {OUT_DIR}/multifreq_*")
print("=" * 70)
