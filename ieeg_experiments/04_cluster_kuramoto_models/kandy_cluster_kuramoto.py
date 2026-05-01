#!/usr/bin/env python3
"""KANDy experiment: Cluster-Reduced Kuramoto model from iEEG data.

Reduces 120 iEEG channels to 4 cluster-level oscillators (using differential
embedding spectral clustering), then uses KANDy to discover inter-cluster
coupling equations in Kuramoto form.

Key design choices:
  - 8-13 Hz bandpass (alpha band) for clean oscillatory dynamics
  - Phase smoothing (Savitzky-Golay) before differentiation
  - 10x downsampling (50 Hz effective) -- well above Nyquist for 13 Hz
  - Focused lift: sin + cos pairs + order params (28 features)
    No 3-body/simplicial terms (speculative, would overfit with limited data)

Cluster assignments (Frobenius, Spectral, k=4, seizure window 87-97s):
  C0 (SOZ core):      7 channels (4 SOZ: 24, 28, 29, 30)
  C1 (bulk):           109 channels (2 SOZ: 21, 22)
  C2 (pacemaker pair): 2 channels (SOZ: 25, 26)
  C3 (boundary pair):  2 channels (SOZ: 23, 27)

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import os
import sys
import copy
import numpy as np
import scipy.io as sio
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

DATA_PATH = ROOT / "data" / "E3Data.mat"
OUT_DIR = ROOT / "results" / "iEEG" / "kuramoto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500  # Hz
N_CHANNELS = 120
SOZ_CHANNELS = list(range(21, 31))  # 0-indexed
ONSET_SAMPLE = 43500  # 87.0s
ONSET_TIME = 87.0

# Differential embedding params
SG_WINDOW = 51
SG_POLY = 5

# Bandpass for Hilbert transform: alpha band (8-13 Hz)
# Chosen because: clean derivatives, relevant to seizure onset, dominant oscillation
BAND_LO = 8.0
BAND_HI = 13.0

# Cluster config
CLUSTER_K = 4

# Phase smoothing (SG filter before differentiation)
PHASE_SG_WINDOW = 51  # ~100ms at 500 Hz
PHASE_SG_POLY = 3

# Downsampling factor
DS = 10  # 500 Hz -> 50 Hz (well above 2*13=26 Hz Nyquist for alpha)
DT_DS = DS / FS  # 0.02s

# Training window — use full pre-seizure + seizure for more data
TRAIN_START_T = 30.0
TRAIN_END_T = 97.0

# KANDy hyperparameters
GRID = 3       # small grid to reduce parameters (only expect linear edges)
K_SPLINE = 3
STEPS = 150
LAMB = 0.0

# Rollout
ROLLOUT_START_T = 80.0
ROLLOUT_END_T = 95.0


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# STEP 0: Load data
# ============================================================
print("=" * 70)
print("CLUSTER-REDUCED KURAMOTO KANDy EXPERIMENT")
print("=" * 70)

print("\nLoading data...")
mat = sio.loadmat(DATA_PATH)
X1 = mat["X1"]  # (49972, 120)
n_samples, n_ch = X1.shape
t_axis = np.arange(n_samples) / FS
print(f"  X1 shape: {X1.shape}, duration: {n_samples / FS:.1f}s")


# ============================================================
# STEP 1: Recompute cluster assignments
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
data_sez = X1[sez_start:sez_end, :]

embs_sez = compute_embedding(data_sez, FS)
covs_sez = {}
for ch, emb in embs_sez.items():
    C = np.cov(emb, rowvar=False)
    C += 1e-10 * np.eye(3)
    covs_sez[ch] = C
D_sez = np.zeros((N_CHANNELS, N_CHANNELS))
for i in range(N_CHANNELS):
    for j in range(i + 1, N_CHANNELS):
        d = np.linalg.norm(covs_sez[i] - covs_sez[j], "fro")
        D_sez[i, j] = d
        D_sez[j, i] = d

sigma = np.median(D_sez[D_sez > 0])
affinity = np.exp(-D_sez ** 2 / (2 * sigma ** 2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

# Relabel: SOZ = cluster 0
soz_labels = labels[SOZ_CHANNELS]
soz_cluster_orig = np.bincount(soz_labels).argmax()
cluster_map = {soz_cluster_orig: 0}
next_label = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = next_label
        next_label += 1
labels = np.array([cluster_map[l] for l in labels])
clusters = {}
for c in range(CLUSTER_K):
    clusters[c] = np.where(labels == c)[0].tolist()

CLUSTER_NAMES = {
    0: "C0 (SOZ core)",
    1: "C1 (bulk)",
    2: "C2 (pacemaker)",
    3: "C3 (boundary)",
}

for c in range(CLUSTER_K):
    n_soz = sum(1 for m in clusters[c] if m in SOZ_CHANNELS)
    soz_in = [m for m in clusters[c] if m in SOZ_CHANNELS]
    print(f"  {CLUSTER_NAMES[c]}: {len(clusters[c])} channels, "
          f"{n_soz} SOZ {soz_in if soz_in else ''}")


# ============================================================
# STEP 2: Phase extraction (8-13 Hz alpha band)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 2: Phase extraction (Hilbert, {BAND_LO}-{BAND_HI} Hz alpha band)")
print("=" * 70)

nyq = FS / 2.0
b, a = butter(4, [BAND_LO / nyq, BAND_HI / nyq], btype="band")
X_filt = filtfilt(b, a, X1, axis=0)

phases_all = np.zeros((n_samples, n_ch))
for ch in range(n_ch):
    analytic = hilbert(X_filt[:, ch])
    phases_all[:, ch] = np.angle(analytic)

print(f"  Phases shape: {phases_all.shape}")


# ============================================================
# STEP 3: Cluster-level state variables
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Cluster-level state variables")
print("=" * 70)


def cluster_order_parameter(phases, ch_idx):
    """Cluster mean phase (psi) and order parameter (r)."""
    z = np.exp(1j * phases[:, ch_idx]).mean(axis=1)
    return np.angle(z), np.abs(z)


psi_raw = np.zeros((n_samples, CLUSTER_K))
r_raw = np.zeros((n_samples, CLUSTER_K))
for c in range(CLUSTER_K):
    psi_raw[:, c], r_raw[:, c] = cluster_order_parameter(phases_all, clusters[c])

# Unwrap phases (continuous)
for c in range(CLUSTER_K):
    psi_raw[:, c] = np.unwrap(psi_raw[:, c])

# Smooth phases before differentiation
psi_smooth = np.zeros_like(psi_raw)
for c in range(CLUSTER_K):
    psi_smooth[:, c] = savgol_filter(psi_raw[:, c], PHASE_SG_WINDOW, PHASE_SG_POLY)

# Downsample (take every DS-th sample)
psi_ds = psi_smooth[::DS]
r_ds = r_raw[::DS]
t_ds = np.arange(len(psi_ds)) * DT_DS

n_ds = len(psi_ds)
print(f"  Downsampled: {n_ds} samples at {1/DT_DS:.0f} Hz (dt={DT_DS}s)")
for c in range(CLUSTER_K):
    freq_hz = np.diff(psi_ds[:, c]).mean() / (2 * np.pi * DT_DS)
    print(f"  {CLUSTER_NAMES[c]}: r mean={r_ds[:, c].mean():.3f}, "
          f"freq ~{freq_hz:.1f} Hz")


# ============================================================
# STEP 4: Build training data with Koopman lift
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Build Koopman lift and training data")
print("=" * 70)

# Training window in downsampled indices
train_s = int(TRAIN_START_T / DT_DS)
train_e = min(int(TRAIN_END_T / DT_DS), n_ds)

psi_train = psi_ds[train_s:train_e]
r_train = r_ds[train_s:train_e]
T_train = len(psi_train)
print(f"  Training window: {TRAIN_START_T}-{TRAIN_END_T}s ({T_train} samples at 50 Hz)")

# --- Lift design ---
# Lean library: standard Kuramoto + Sakaguchi + coherence
# sin(psi_i-psi_j): 6 unique pairs  -- standard Kuramoto coupling
# cos(psi_i-psi_j): 6 unique pairs  -- Sakaguchi phase lag
# r_k:              4 clusters       -- local coherence modulation
# Total: 16 features => KAN [16, 4] = 64 edges, grid=3 => ~384 params
# This is well-posed with ~3000+ training samples (30-97s at 50 Hz)
PAIRS = [(i, j) for i in range(CLUSTER_K) for j in range(i + 1, CLUSTER_K)]
N_PAIRS = len(PAIRS)  # 6

FEAT_NAMES = []
# Group 1: sin(psi_i - psi_j) for unique pairs (6)
for i, j in PAIRS:
    FEAT_NAMES.append(f"sin(p{i}-p{j})")
# Group 2: cos(psi_i - psi_j) for unique pairs (6)
for i, j in PAIRS:
    FEAT_NAMES.append(f"cos(p{i}-p{j})")
# Group 3: r_k for each cluster (4)
for k in range(CLUSTER_K):
    FEAT_NAMES.append(f"r{k}")

N_FEAT = len(FEAT_NAMES)
print(f"  Lift: {N_FEAT} features")
print(f"    sin(psi_i-psi_j):  {N_PAIRS}")
print(f"    cos(psi_i-psi_j):  {N_PAIRS}")
print(f"    r_k:               {CLUSTER_K}")
print(f"  KAN: [{N_FEAT}, {CLUSTER_K}]")


def build_lift(psi_arr, r_arr):
    """Compute lift features. psi_arr: (T, 4), r_arr: (T, 4) -> (T, N_FEAT)."""
    feats = []
    for i, j in PAIRS:
        feats.append(np.sin(psi_arr[:, i] - psi_arr[:, j]))
    for i, j in PAIRS:
        feats.append(np.cos(psi_arr[:, i] - psi_arr[:, j]))
    for k in range(CLUSTER_K):
        feats.append(r_arr[:, k])
    return np.column_stack(feats)


def build_lift_single(psi_vec, r_vec):
    """Lift a single state (4,) + (4,) -> (1, N_FEAT)."""
    return build_lift(psi_vec[None, :], r_vec[None, :])


# Compute lift features on training window
Phi_train = build_lift(psi_train, r_train)

# Derivatives via central differences
psi_dot_train = (psi_train[2:] - psi_train[:-2]) / (2.0 * DT_DS)
Phi_inner = Phi_train[1:-1]
psi_inner = psi_train[1:-1]
r_inner = r_train[1:-1]
T_inner = len(Phi_inner)

print(f"  Training samples: {T_inner}")
print(f"  Samples-to-features ratio: {T_inner / N_FEAT:.1f}:1")
for c in range(CLUSTER_K):
    pd = psi_dot_train[:, c]
    print(f"  {CLUSTER_NAMES[c]}: dpsi mean={pd.mean():.1f} rad/s, "
          f"std={pd.std():.1f}, freq~{pd.mean()/(2*np.pi):.1f} Hz")

# Check feature statistics
print("\n  Feature statistics (min/max/std):")
for idx, fn in enumerate(FEAT_NAMES):
    f = Phi_inner[:, idx]
    print(f"    {fn:25s}: [{f.min():+.3f}, {f.max():+.3f}], std={f.std():.3f}")

assert not np.any(np.isnan(Phi_inner)), "NaN in features"
assert not np.any(np.isnan(psi_dot_train)), "NaN in derivatives"
print("\n  Data integrity: OK")


# ============================================================
# STEP 5: Train KANDy
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Train KANDy model")
print("=" * 70)

lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="identity")

model = KANDy(
    lift=lift,
    grid=GRID,
    k=K_SPLINE,
    steps=STEPS,
    seed=SEED,
    device="cpu",
)

print(f"  KAN: [{N_FEAT}, {CLUSTER_K}], grid={GRID}, k={K_SPLINE}")
print(f"  Steps={STEPS}, lamb={LAMB}, patience=0")

model.fit(
    X=Phi_inner,
    X_dot=psi_dot_train,
    val_frac=0.15,
    test_frac=0.15,
    lamb=LAMB,
    patience=0,
    verbose=True,
)


# ============================================================
# STEP 6: One-step prediction
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: One-step prediction")
print("=" * 70)

n_test = min(500, T_inner // 5)
test_phi = Phi_inner[-n_test:]
test_dot = psi_dot_train[-n_test:]
pred_dot = model.predict(test_phi)

mse_onestep = np.mean((pred_dot - test_dot) ** 2)
rmse_onestep = np.sqrt(mse_onestep)

# Normalized RMSE (relative to total std of dpsi)
nrmse_onestep = rmse_onestep / np.std(psi_dot_train)

print(f"  Overall one-step MSE:   {mse_onestep:.4f}")
print(f"  Overall one-step RMSE:  {rmse_onestep:.4f}")
print(f"  Overall one-step NRMSE: {nrmse_onestep:.4f}")
for c in range(CLUSTER_K):
    rmse_c = np.sqrt(np.mean((pred_dot[:, c] - test_dot[:, c]) ** 2))
    ss_res = np.sum((pred_dot[:, c] - test_dot[:, c]) ** 2)
    ss_tot = np.sum((test_dot[:, c] - test_dot[:, c].mean()) ** 2)
    r2_c = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  {CLUSTER_NAMES[c]}: RMSE={rmse_c:.4f}, R2={r2_c:.4f}")


# ============================================================
# STEP 7: Rollout (BEFORE symbolic extraction)
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Autoregressive rollout")
print("=" * 70)

# Rollout in downsampled indices, relative to training window start
rollout_s_ds = int(ROLLOUT_START_T / DT_DS) - train_s
rollout_e_ds = int(ROLLOUT_END_T / DT_DS) - train_s
rollout_s_ds = max(1, rollout_s_ds)
rollout_e_ds = min(T_inner, rollout_e_ds)
N_ROLLOUT = rollout_e_ds - rollout_s_ds

print(f"  Rollout: {ROLLOUT_START_T}s to {ROLLOUT_END_T}s ({N_ROLLOUT} steps at dt={DT_DS}s)")


def rollout_cluster(psi0, r_series, n_steps, dt):
    """RK4 rollout. Uses observed r(t) from data (not predicted)."""
    psi_cur = psi0.copy().astype(np.float64)
    traj = [psi_cur.copy()]

    for step in range(1, n_steps):
        r_cur = r_series[min(step, len(r_series) - 1)]

        def f(ps):
            phi = build_lift_single(ps, r_cur)
            return model.predict(phi).ravel()

        k1 = f(psi_cur)
        k2 = f(psi_cur + 0.5 * dt * k1)
        k3 = f(psi_cur + 0.5 * dt * k2)
        k4 = f(psi_cur + dt * k3)
        psi_cur = psi_cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(psi_cur.copy())

    return np.array(traj)


psi_true_roll = psi_inner[rollout_s_ds:rollout_e_ds]
r_true_roll = r_inner[rollout_s_ds:rollout_e_ds]
psi0 = psi_inner[rollout_s_ds]

print(f"  Rolling out {N_ROLLOUT} steps ...")
psi_pred_roll = rollout_cluster(psi0, r_true_roll, N_ROLLOUT, DT_DS)

rmse_rollout = np.sqrt(np.mean((psi_pred_roll - psi_true_roll) ** 2))
print(f"  Rollout RMSE (all): {rmse_rollout:.4f}")
for c in range(CLUSTER_K):
    rmse_c = np.sqrt(np.mean((psi_pred_roll[:, c] - psi_true_roll[:, c]) ** 2))
    print(f"    {CLUSTER_NAMES[c]}: RMSE={rmse_c:.4f}")

# Frequency comparison
print("\n  Frequency comparison (rollout window):")
for c in range(CLUSTER_K):
    true_freq = np.diff(psi_true_roll[:, c]).mean() / (2 * np.pi * DT_DS)
    pred_freq = np.diff(psi_pred_roll[:, c]).mean() / (2 * np.pi * DT_DS)
    print(f"    {CLUSTER_NAMES[c]}: true={true_freq:.2f} Hz, pred={pred_freq:.2f} Hz")


# ============================================================
# STEP 8: Edge activation plots (BEFORE symbolic)
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Edge activation plots")
print("=" * 70)

use_pub_style()

n_sub = min(3000, int(T_inner * 0.70))
sub_idx = np.random.choice(int(T_inner * 0.70), n_sub, replace=False)
train_phi_t = torch.tensor(Phi_inner[sub_idx], dtype=torch.float32)

try:
    fig, axes = plot_all_edges(
        model.model_,
        X=train_phi_t,
        in_var_names=FEAT_NAMES,
        out_var_names=[f"dpsi{c}/dt" for c in range(CLUSTER_K)],
        save=str(OUT_DIR / "kandy_cluster_edge_activations"),
    )
    plt.close(fig)
    print("  Saved edge activation plots")
except Exception as e:
    print(f"  [WARN] Edge activation plot failed: {e}")


# ============================================================
# STEP 9: Symbolic extraction
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Symbolic extraction")
print("=" * 70)

LINEAR_LIB = make_symbolic_lib({
    "x":   (lambda x: x,       lambda x: x,       1),
    "0":   (lambda x: x * 0,   lambda x: x * 0,   0),
})

# Deep copy for symbolic (fix_symbolic destroys splines)
kan_sym = copy.deepcopy(model.model_).cpu()
kan_sym.save_act = True

sym_input = torch.tensor(Phi_inner[:min(5000, T_inner)], dtype=torch.float32)
with torch.no_grad():
    kan_sym(sym_input)

print("  Running robust_auto_symbolic ...")
robust_auto_symbolic(
    kan_sym,
    lib=LINEAR_LIB,
    r2_threshold=0.80,
    weight_simple=0.8,
    topk_edges=25,
    set_others_to_zero=True,
)

# Extract formulas
exprs, vars_ = kan_sym.symbolic_formula()
sub_map = {sp.Symbol(str(v)): sp.Symbol(n) for v, n in zip(vars_, FEAT_NAMES)}
formulas = []
COEFF_TOL = 0.005
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym)
    terms = []
    for term in sp.Add.make_args(sym):
        coeff, rest = term.as_coeff_Mul()
        if abs(float(coeff)) > COEFF_TOL:
            terms.append(sp.Float(round(float(coeff), 3)) * rest)
    formulas.append(sum(terms) if terms else sp.Float(0))

print("\n" + "-" * 70)
print("DISCOVERED EQUATIONS")
print("-" * 70)
for c in range(CLUSTER_K):
    print(f"\n  d(psi_{c})/dt = {formulas[c]}")
    print(f"    [{CLUSTER_NAMES[c]}]")
    n_terms = len(sp.Add.make_args(formulas[c]))
    print(f"    ({n_terms} terms)")


# ============================================================
# STEP 10: Edge activity analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Edge activity analysis")
print("=" * 70)

for c in range(CLUSTER_K):
    print(f"\n  {CLUSTER_NAMES[c]} (dpsi{c}/dt):")
    formula = formulas[c]
    for fn in FEAT_NAMES:
        feat_sym = sp.Symbol(fn)
        coeff = formula.coeff(feat_sym)
        if abs(float(coeff)) > COEFF_TOL:
            print(f"    {fn:25s}: {float(coeff):+.4f}")
    # Check for constant (bias = natural frequency offset)
    const_term = formula.as_coeff_Add()[0]
    if abs(float(const_term)) > COEFF_TOL:
        print(f"    {'(constant)':25s}: {float(const_term):+.4f}")

# Summary by feature category
print("\n  Summary by feature group:")
categories = {
    "sin(psi_i-psi_j)": [f for f in FEAT_NAMES if f.startswith("sin(p")],
    "cos(psi_i-psi_j)": [f for f in FEAT_NAMES if f.startswith("cos(p")],
    "r_k": [f for f in FEAT_NAMES if f.startswith("r") and "*" not in f],
}
for cat_name, cat_feats in categories.items():
    n_active = 0
    for c in range(CLUSTER_K):
        for fn in cat_feats:
            coeff = formulas[c].coeff(sp.Symbol(fn))
            if abs(float(coeff)) > COEFF_TOL:
                n_active += 1
    total = len(cat_feats) * CLUSTER_K
    print(f"    {cat_name:25s}: {n_active}/{total} active")


# ============================================================
# STEP 11: Plots
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Publication plots")
print("=" * 70)

use_pub_style()
colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
t_roll = np.arange(N_ROLLOUT) * DT_DS + ROLLOUT_START_T

# 11a. Phase rollout
fig, axes = plt.subplots(CLUSTER_K, 1, figsize=(10, 2.5 * CLUSTER_K), sharex=True)
for c, ax in enumerate(axes):
    ax.plot(t_roll, psi_true_roll[:, c], color=colors[c], lw=1.2, label="True")
    ax.plot(t_roll, psi_pred_roll[:, c], color=colors[c], lw=1.0, ls="--",
            alpha=0.8, label="KANDy")
    ax.set_ylabel(f"$\\psi_{c}$ (rad)", fontsize=10)
    ax.set_title(CLUSTER_NAMES[c], fontsize=10, loc="left")
    if c == 0:
        ax.legend(loc="upper right", fontsize=8)
    ax.axvline(ONSET_TIME, color="black", ls=":", lw=0.8, alpha=0.6)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Cluster Kuramoto: Phase Rollout", fontsize=12, y=1.01)
fig.tight_layout()
save_fig(fig, "kandy_cluster_rollout")

# 11b. Phase differences (SOZ vs others)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for idx, (c_other, ax) in enumerate(zip([1, 2, 3], axes)):
    dpsi_true = psi_true_roll[:, 0] - psi_true_roll[:, c_other]
    dpsi_pred = psi_pred_roll[:, 0] - psi_pred_roll[:, c_other]
    ax.plot(t_roll, dpsi_true, color=colors[c_other], lw=1.2, label="True")
    ax.plot(t_roll, dpsi_pred, color=colors[c_other], lw=1.0, ls="--",
            alpha=0.8, label="KANDy")
    ax.set_ylabel(f"$\\psi_0 - \\psi_{c_other}$")
    ax.set_title(f"SOZ vs {CLUSTER_NAMES[c_other]}", fontsize=10, loc="left")
    ax.axvline(ONSET_TIME, color="black", ls=":", lw=0.8, alpha=0.6)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Phase Differences: SOZ vs Other Clusters", fontsize=12, y=1.01)
fig.tight_layout()
save_fig(fig, "kandy_cluster_phase_diffs")

# 11c. Order parameters (from data, for context)
fig, ax = plt.subplots(figsize=(10, 3.5))
r_smooth_vis = np.zeros((n_ds, CLUSTER_K))
for c in range(CLUSTER_K):
    r_smooth_vis[:, c] = savgol_filter(r_ds[:, c],
                                        min(51, len(r_ds[:, c]) // 2 * 2 - 1), 3)
    ax.plot(t_ds, r_smooth_vis[:, c], color=colors[c], lw=1.2, label=CLUSTER_NAMES[c])
ax.axvline(ONSET_TIME, color="black", ls=":", lw=1.0, label="Seizure onset")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$r_k(t)$")
ax.set_title("Cluster Order Parameters", fontsize=11)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_ylim(0, 1.05)
ax.set_xlim(max(0, TRAIN_START_T - 5), TRAIN_END_T + 5)
fig.tight_layout()
save_fig(fig, "kandy_cluster_order_params")

# 11d. One-step scatter
fig, axes = plt.subplots(1, CLUSTER_K, figsize=(3.5 * CLUSTER_K, 3.5))
for c, ax in enumerate(axes):
    ax.scatter(test_dot[:, c], pred_dot[:, c], s=1, alpha=0.3, color=colors[c])
    lo = min(test_dot[:, c].min(), pred_dot[:, c].min())
    hi = max(test_dot[:, c].max(), pred_dot[:, c].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ss_res = np.sum((pred_dot[:, c] - test_dot[:, c]) ** 2)
    ss_tot = np.sum((test_dot[:, c] - test_dot[:, c].mean()) ** 2)
    r2_c = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    ax.set_title(f"{CLUSTER_NAMES[c]}\n$R^2$={r2_c:.3f}", fontsize=9)
    ax.set_xlabel("True $d\\psi/dt$ (rad/s)", fontsize=9)
    if c == 0:
        ax.set_ylabel("Predicted $d\\psi/dt$", fontsize=9)
fig.suptitle("One-Step Prediction", fontsize=11, y=1.02)
fig.tight_layout()
save_fig(fig, "kandy_cluster_onestep")

# 11e. Derivative time series comparison (subset)
fig, axes = plt.subplots(CLUSTER_K, 1, figsize=(10, 2.5 * CLUSTER_K), sharex=True)
# Use a small window around seizure onset for detail
vis_s = max(0, int((ONSET_TIME - 3) / DT_DS) - train_s - 1)
vis_e = min(T_inner, int((ONSET_TIME + 5) / DT_DS) - train_s - 1)
t_vis = np.arange(vis_s, vis_e) * DT_DS + TRAIN_START_T + DT_DS

phi_vis = Phi_inner[vis_s:vis_e]
dot_vis = psi_dot_train[vis_s:vis_e]
pred_vis = model.predict(phi_vis)

for c, ax in enumerate(axes):
    ax.plot(t_vis, dot_vis[:, c], color=colors[c], lw=1.0, label="True", alpha=0.8)
    ax.plot(t_vis, pred_vis[:, c], color="black", lw=0.8, ls="--", label="KANDy")
    ax.set_ylabel(f"$d\\psi_{c}/dt$")
    ax.set_title(CLUSTER_NAMES[c], fontsize=10, loc="left")
    ax.axvline(ONSET_TIME, color="gray", ls=":", lw=0.8)
    if c == 0:
        ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Derivative Prediction Around Seizure Onset", fontsize=12, y=1.01)
fig.tight_layout()
save_fig(fig, "kandy_cluster_deriv_detail")

# 11f. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    try:
        fig, ax = plot_loss_curves(
            model.train_results_,
            save=str(OUT_DIR / "kandy_cluster_loss"),
        )
        plt.close(fig)
        print("  Saved loss curves")
    except Exception as e:
        print(f"  [WARN] Loss curve plot failed: {e}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(f"""
System:    Cluster-Reduced Kuramoto from iEEG (Episode 3)
Band:      {BAND_LO}-{BAND_HI} Hz (alpha)
Clusters:  {CLUSTER_K} (Frobenius/Spectral/k=4)
State:     4 cluster mean phases psi_0..psi_3
Lift:      {N_FEAT} features (sin, cos, r)
KAN:       [{N_FEAT}, {CLUSTER_K}], grid={GRID}, k={K_SPLINE}
Training:  {TRAIN_START_T}-{TRAIN_END_T}s ({T_inner} samples at 50 Hz), {STEPS} LBFGS steps
Downsample: {DS}x (500 Hz -> 50 Hz)
Phase smoothing: SG({PHASE_SG_WINDOW}, {PHASE_SG_POLY})

One-step MSE:   {mse_onestep:.4f}
One-step RMSE:  {rmse_onestep:.4f}
One-step NRMSE: {nrmse_onestep:.4f}
Rollout RMSE:   {rmse_rollout:.4f}
Rollout window: {ROLLOUT_START_T}-{ROLLOUT_END_T}s ({N_ROLLOUT} steps)
""")

print("Discovered equations:")
for c in range(CLUSTER_K):
    print(f"  d(psi_{c})/dt = {formulas[c]}")
    print(f"    [{CLUSTER_NAMES[c]}]")

print(f"\nResults saved to: {OUT_DIR}/kandy_cluster_*")
print("Done.")
