#!/usr/bin/env python3
"""Extract equations and generate model.plot() for ALL iEEG KANDy models.

Trains each model fresh and extracts:
  1. Symbolic equations via auto_symbolic
  2. PyKAN architecture plot via model.plot()
  3. One-step R² and edge analysis

Models:
  A. Multifreq 3-oscillator amplitude dynamics (alpha band, [9,3])
  B. Envelope-of-envelope slow phase Kuramoto ([15,3])
  C. Stuart-Landau OLS fits (not KANDy, but prints equations)

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import copy
import numpy as np
import torch
import sympy as sp
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# Add paths
sys.path.insert(0, str(ROOT / "ieeg_experiments" / "03_clustering_and_supporting_analysis"))
sys.path.insert(0, str(ROOT / "src"))

from ieeg_utils import (
    load_episodes, extract_cluster_amplitudes, extract_cluster_phases,
    savgol_derivative, save_fig, setup_style, bandpass,
    FS, ONSET_TIMES, CLUSTERS, CLUSTER_IDS, CLUSTER_NAMES,
    CLUSTER_COLORS, ALPHA_BAND,
)
from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import (
    get_edge_activation, plot_all_edges, plot_loss_curves, use_pub_style,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = len(CLUSTER_IDS)
LINEAR_LIB = make_symbolic_lib({
    "x": (lambda x: x, lambda x: x, 1),
    "0": (lambda x: x * 0, lambda x: x * 0, 0),
})


def extract_equations(model, Phi, feat_names, out_names, model_name):
    """Extract symbolic equations + model.plot() for any trained KANDy model."""
    n_samp = len(Phi)
    n_feat = len(feat_names)
    n_out = len(out_names)

    # --- Symbolic extraction ---
    print(f"\n  [{model_name}] Symbolic extraction...")
    model_copy = copy.deepcopy(model)
    sym_subset = torch.tensor(Phi[:min(2048, n_samp)], dtype=torch.float32)
    model_copy.model_.save_act = True
    with torch.no_grad():
        model_copy.model_(sym_subset)

    try:
        robust_auto_symbolic(
            model_copy.model_,
            lib=LINEAR_LIB,
            r2_threshold=0.80,
            weight_simple=0.8,
            topk_edges=n_feat * n_out,
        )

        exprs, vars_ = model_copy.model_.symbolic_formula()
        sub_map = {sp.Symbol(str(v)): sp.Symbol(n)
                   for v, n in zip(vars_, feat_names)}

        print(f"\n  ┌─────────────────────────────────────────────")
        print(f"  │ DISCOVERED EQUATIONS: {model_name}")
        print(f"  ├─────────────────────────────────────────────")
        formulas = []
        for i in range(n_out):
            sym = sp.sympify(exprs[i]).xreplace(sub_map)
            sym = sp.expand(sym).xreplace(
                {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
            )
            formulas.append(sym)
            print(f"  │  {out_names[i]} = {sym}")
        print(f"  └─────────────────────────────────────────────\n")
        return formulas
    except Exception as e:
        print(f"  [WARN] Symbolic extraction failed: {e}")
        return None


def do_model_plot(model, Phi, feat_names, out_names, model_name, save_path):
    """Call pykan model.plot() to visualize the KAN architecture."""
    print(f"  [{model_name}] Generating model.plot()...")
    # model.plot() requires save_act=True and a forward pass first
    model.model_.save_act = True
    sym_subset = torch.tensor(Phi[:min(2048, len(Phi))], dtype=torch.float32)
    with torch.no_grad():
        model.model_(sym_subset)

    try:
        in_vars = [rf"${n}$" for n in feat_names]
        out_vars = [rf"${n}$" for n in out_names]
        model.model_.plot(
            in_vars=in_vars,
            out_vars=out_vars,
            title=model_name,
        )
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved {Path(save_path).name}.png")
    except Exception as e:
        print(f"    [WARN] model.plot() failed: {e}")


# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("EQUATION EXTRACTION FROM ALL iEEG KANDy MODELS")
print("=" * 70)

episodes = load_episodes()


# ============================================================
# MODEL A: 3-Oscillator Amplitude Dynamics (alpha band)
# ============================================================
print("\n" + "=" * 70)
print("MODEL A: 3-Cluster Amplitude Dynamics (alpha, [9, 3])")
print("=" * 70)

# --- Extract amplitude envelopes ---
DS_A = 100  # 500 Hz -> 5 Hz
DT_A = DS_A / FS
FS_A = 1.0 / DT_A
SMOOTH_SAMPLES = int(4.0 * FS)
smooth_kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES

amp_data = {}
for ep in [1, 2, 3]:
    modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
    amp_data[ep] = {cid: modes[cid][::DS_A] for cid in CLUSTER_IDS}

# --- Build training data ---
SG_WIN_A = 13
SG_POLY_A = 4
TRAIN_START = 30.0
TRAIN_END = 97.0
TRIM_A = SG_WIN_A // 2 + 3

# ReLU thresholds
thresholds_A = {}
for cid in CLUSTER_IDS:
    vals = []
    for ep in [1, 2, 3]:
        t_ds = np.arange(len(amp_data[ep][cid])) * DT_A
        onset = ONSET_TIMES[ep]
        pre_mask = (t_ds >= onset - 10) & (t_ds < onset)
        if pre_mask.sum() > 0:
            vals.append(amp_data[ep][cid][pre_mask].mean())
    thresholds_A[cid] = np.mean(vals)

FEAT_A = [f"x_{c}" for c in CLUSTER_IDS]
PAIR_A = [(0, 1), (0, 2), (1, 2)]
FEAT_A += [f"x{CLUSTER_IDS[i]}*x{CLUSTER_IDS[j]}" for i, j in PAIR_A]
FEAT_A += [f"ReLU(x{c})" for c in CLUSTER_IDS]
N_FEAT_A = len(FEAT_A)

OUT_A = [f"dx_{c}/dt" for c in CLUSTER_IDS]

Phi_parts, dot_parts = [], []
for ep in [1, 2, 3]:
    n_ds = len(amp_data[ep][CLUSTER_IDS[0]])
    t_ds = np.arange(n_ds) * DT_A
    s = int(TRAIN_START / DT_A)
    e = min(int(TRAIN_END / DT_A), n_ds)

    x_all = np.column_stack([amp_data[ep][cid][s:e] for cid in CLUSTER_IDS])

    x_dot = np.zeros_like(x_all)
    for ci in range(N_CLUSTERS):
        x_dot[:, ci] = savgol_filter(x_all[:, ci], SG_WIN_A, SG_POLY_A,
                                     deriv=1, delta=DT_A)

    v = slice(TRIM_A, len(x_all) - TRIM_A)
    x_v = x_all[v]
    x_dot_v = x_dot[v]

    feats = []
    for ci in range(N_CLUSTERS):
        feats.append(x_v[:, ci])
    for i, j in PAIR_A:
        feats.append(x_v[:, i] * x_v[:, j])
    for ci, cid in enumerate(CLUSTER_IDS):
        feats.append(np.maximum(x_v[:, ci] - thresholds_A[cid], 0.0))

    Phi_parts.append(np.column_stack(feats))
    dot_parts.append(x_dot_v)

Phi_A = np.vstack(Phi_parts)
Dot_A = np.vstack(dot_parts)
print(f"  Training: X={Phi_A.shape}, Y={Dot_A.shape}")

# --- Train ---
lift_A = CustomLift(fn=lambda X: X, output_dim=N_FEAT_A, name="amp_identity")
model_A = KANDy(lift=lift_A, grid=3, k=3, steps=200, seed=SEED, device="cpu")
model_A.fit(X=Phi_A, X_dot=Dot_A, val_frac=0.15, test_frac=0.15,
            lamb=0.0, patience=0, verbose=False)

# --- R² ---
n_test = min(500, len(Phi_A) // 5)
pred = model_A.predict(Phi_A[-n_test:])
true = Dot_A[-n_test:]
ss_res = np.sum((pred - true) ** 2, axis=0)
ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
r2 = 1 - ss_res / np.maximum(ss_tot, 1e-15)
r2_overall = 1 - np.sum(ss_res) / np.sum(ss_tot)
print(f"  R² overall: {r2_overall:.4f}")
for i, cid in enumerate(CLUSTER_IDS):
    print(f"    {CLUSTER_NAMES[cid]}: R² = {r2[i]:.4f}")

# --- Equations + model.plot() ---
extract_equations(model_A, Phi_A, FEAT_A, OUT_A, "Amplitude [9,3]")
do_model_plot(model_A, Phi_A, FEAT_A, OUT_A, "Amplitude [9,3]",
              str(OUT_DIR / "model_plot_amplitude"))


# ============================================================
# MODEL B: Envelope Phase Kuramoto ([15, 3])
# ============================================================
print("\n" + "=" * 70)
print("MODEL B: Envelope Phase Kuramoto ([15, 3])")
print("=" * 70)

# --- Extract envelopes at 50 Hz ---
DS_B = 10  # 500 Hz -> 50 Hz
DT_B_env = DS_B / FS
FS_B_env = FS / DS_B

DS_B_phase = 5  # 50 Hz -> 10 Hz
DT_B = DT_B_env * DS_B_phase
FS_B = 1.0 / DT_B
SG_WIN_B = 15
SG_POLY_B = 4
TRIM_B = SG_WIN_B // 2 + 3

CLUSTER_PAIRS_B = [(0, 2), (0, 3), (2, 3)]

env_data = {}
for ep in [1, 2, 3]:
    modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
    env_data[ep] = {cid: modes[cid][::DS_B] for cid in CLUSTER_IDS}

# Slow bandpass (0.03-0.1 Hz)
slow_lo, slow_hi = 0.03, 0.1
psi_data = {}
r_data = {}
dpsi_data = {}
for ep in [1, 2, 3]:
    psi_data[ep] = {}
    r_data[ep] = {}
    dpsi_data[ep] = {}
    for cid in CLUSTER_IDS:
        sig = env_data[ep][cid]
        nyq = FS_B_env / 2.0
        b, a = butter(3, [max(slow_lo / nyq, 0.001), min(slow_hi / nyq, 0.999)],
                      btype="band")
        env_filt = filtfilt(b, a, sig)
        analytic = hilbert(env_filt)
        phase = np.angle(analytic)
        r = np.abs(analytic)

        psi_data[ep][cid] = phase[::DS_B_phase]
        r_data[ep][cid] = r[::DS_B_phase]
        psi_uw = np.unwrap(phase[::DS_B_phase])
        dpsi_data[ep][cid] = savgol_derivative(psi_uw, DT_B,
                                               window=SG_WIN_B, polyorder=SG_POLY_B)

# ReLU thresholds
r_thresholds = {}
for cid in CLUSTER_IDS:
    vals = []
    for ep in [1, 2, 3]:
        n = len(psi_data[ep][cid])
        t = np.arange(n) * DT_B
        onset = ONSET_TIMES[ep]
        mask = (t >= onset - 15) & (t < onset)
        if mask.sum() > 0:
            vals.append(r_data[ep][cid][mask].mean())
    r_thresholds[cid] = np.mean(vals) if vals else 0.0

# Feature names
FEAT_B = []
for ci, cj in CLUSTER_PAIRS_B:
    FEAT_B.append(f"sin(p{ci}-p{cj})")
    FEAT_B.append(f"cos(p{ci}-p{cj})")
for cid in CLUSTER_IDS:
    FEAT_B.append(f"r_{cid}")
for cid in CLUSTER_IDS:
    FEAT_B.append(f"ReLU(r_{cid})")
for cid in CLUSTER_IDS:
    FEAT_B.append(f"ReLU*sin_{cid}")
N_FEAT_B = len(FEAT_B)
OUT_B = [f"dpsi_{c}/dt" for c in CLUSTER_IDS]

# Build features
Phi_B_parts, dot_B_parts = [], []
for ep in [1, 2, 3]:
    T_full = len(psi_data[ep][CLUSTER_IDS[0]])
    feats = np.zeros((T_full, N_FEAT_B))
    col = 0
    for ci, cj in CLUSTER_PAIRS_B:
        diff = psi_data[ep][ci] - psi_data[ep][cj]
        feats[:, col] = np.sin(diff)
        feats[:, col + 1] = np.cos(diff)
        col += 2
    for cid in CLUSTER_IDS:
        feats[:, col] = r_data[ep][cid]
        col += 1
    relu_vals = {}
    for cid in CLUSTER_IDS:
        relu = np.maximum(r_data[ep][cid] - r_thresholds[cid], 0.0)
        feats[:, col] = relu
        relu_vals[cid] = relu
        col += 1
    psi_mean = np.zeros(T_full)
    for cid in CLUSTER_IDS:
        psi_mean += psi_data[ep][cid]
    psi_mean /= N_CLUSTERS
    for cid in CLUSTER_IDS:
        feats[:, col] = relu_vals[cid] * np.sin(psi_data[ep][cid] - psi_mean)
        col += 1

    tgts = np.column_stack([dpsi_data[ep][cid] for cid in CLUSTER_IDS])
    Phi_B_parts.append(feats[TRIM_B:-TRIM_B])
    dot_B_parts.append(tgts[TRIM_B:-TRIM_B])

Phi_B = np.vstack(Phi_B_parts)
Dot_B = np.vstack(dot_B_parts)
n_B = min(len(Phi_B), len(Dot_B))
Phi_B, Dot_B = Phi_B[:n_B], Dot_B[:n_B]
print(f"  Training: X={Phi_B.shape}, Y={Dot_B.shape}")

# --- Train ---
lift_B = CustomLift(fn=lambda X: X, output_dim=N_FEAT_B, name="phase_identity")
model_B = KANDy(lift=lift_B, grid=5, k=3, steps=200, seed=SEED, device="cpu")
model_B.fit(X=Phi_B, X_dot=Dot_B, val_frac=0.15, test_frac=0.15,
            lamb=0.0, patience=0, verbose=False)

# --- R² ---
n_test = min(500, len(Phi_B) // 5)
pred = model_B.predict(Phi_B[-n_test:])
true = Dot_B[-n_test:]
ss_res = np.sum((pred - true) ** 2, axis=0)
ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
r2 = 1 - ss_res / np.maximum(ss_tot, 1e-15)
r2_overall = 1 - np.sum(ss_res) / np.sum(ss_tot)
print(f"  R² overall: {r2_overall:.4f}")
for i, cid in enumerate(CLUSTER_IDS):
    print(f"    {CLUSTER_NAMES[cid]}: R² = {r2[i]:.4f}")

# --- Equations + model.plot() (regardless of R²) ---
extract_equations(model_B, Phi_B, FEAT_B, OUT_B, "Envelope Phase [15,3]")
do_model_plot(model_B, Phi_B, FEAT_B, OUT_B, "Envelope Phase [15,3]",
              str(OUT_DIR / "model_plot_envelope_phase"))


# ============================================================
# MODEL C: Stuart-Landau OLS (from early warning analysis)
# ============================================================
print("\n" + "=" * 70)
print("MODEL C: Stuart-Landau OLS Fits (dA/dt = mu*A - alpha*A^3)")
print("=" * 70)

# Extract amplitude and derivative at 5 Hz
for ep in [1, 2, 3]:
    onset = ONSET_TIMES[ep]
    print(f"\n  Episode {ep} (onset={onset}s):")
    for cid in CLUSTER_IDS:
        A = amp_data[ep][cid]
        dAdt = savgol_filter(A, SG_WIN_A, SG_POLY_A, deriv=1, delta=DT_A)
        t = np.arange(len(A)) * DT_A

        for win_name, t_lo, t_hi in [
            ("pre-seizure", onset - 20, onset),
            ("seizure", onset, onset + 10),
            ("full", 30, 97),
        ]:
            mask = (t >= t_lo) & (t < t_hi)
            if mask.sum() < 10:
                continue
            A_win = A[mask]
            dA_win = dAdt[mask]

            # OLS: dA/dt = mu*A - alpha*A^3
            X_sl = np.column_stack([A_win, A_win ** 3])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_sl, dA_win, rcond=None)
                mu = coeffs[0]
                alpha = -coeffs[1]
                pred_sl = X_sl @ coeffs
                ss_res = np.sum((dA_win - pred_sl) ** 2)
                ss_tot = np.sum((dA_win - dA_win.mean()) ** 2)
                r2_sl = 1.0 - ss_res / max(ss_tot, 1e-15)
            except:
                mu, alpha, r2_sl = 0, 0, 0

            print(f"    {CLUSTER_NAMES[cid]:20s} ({win_name:12s}): "
                  f"dA/dt = {mu:+.4f}*A - {alpha:.4f}*A³  (R²={r2_sl:.4f})")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ALL DISCOVERED EQUATIONS SUMMARY")
print("=" * 70)
print(f"\nPlots saved to {OUT_DIR}/:")
print(f"  model_plot_amplitude.png    — KAN architecture for amplitude model")
print(f"  model_plot_envelope_phase.png — KAN architecture for envelope phase model")
print("\nDone.")
