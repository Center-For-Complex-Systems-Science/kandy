#!/usr/bin/env python3
"""Multiband Kuramoto: run cluster-reduced Kuramoto across ALL frequency bands.

Bands: delta (1-4), theta (4-8), alpha (8-13), beta (13-30), low_gamma (30-50)

For each band:
  - Extract cluster mean phases + order parameters
  - Mean-subtract targets
  - Train KANDy [15, 3] on full window
  - Extract symbolic equations
  - Rollout + plots

Clustering is done ONCE (differential embedding on seizure window).

Author: KANDy Researcher Agent
Date: 2026-03-30
"""

import sys
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
sys.path.insert(0, str(ROOT / "src"))
from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = ROOT / "data" / "QM" / "PatientQM_02Clean.txt"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500
ONSET_TIME = 88.25

# Differential embedding
SG_WINDOW = 51
SG_POLY = 5

# All frequency bands
BANDS = {
    "delta":      (1.0, 4.0),
    "theta":      (4.0, 8.0),
    "alpha":      (8.0, 13.0),
    "beta":       (13.0, 30.0),
    "low_gamma":  (30.0, 50.0),
}

CLUSTER_K = 4
SOZ_CHANNELS = list(range(21, 31))

PHASE_SG_WIN = 51
PHASE_SG_POLY = 3

DS = 10
DT_DS = DS / FS

GRID = 3
K_SPLINE = 3
STEPS = 200
LAMB = 0.0

TRAIN_START, TRAIN_END = 30.0, 97.0

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
# Load data + clustering (ONCE)
# ============================================================
print("=" * 70)
print("MULTIBAND CROSS-FREQUENCY COUPLED KURAMOTO MODEL")
print("=" * 70)

X = np.loadtxt(DATA_PATH)
n_samples, n_ch = X.shape
t_axis = np.arange(n_samples) / FS
ONSET_SAMPLE = int(ONSET_TIME * FS)
print(f"Data: {X.shape}, {n_samples/FS:.1f}s, {n_ch} ch")

# Differential embedding clustering on seizure window
print("\nClustering (seizure window)...")


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


sez_s = ONSET_SAMPLE
sez_e = min(n_samples, sez_s + int(10.0 * FS))
embs = compute_embedding(X[sez_s:sez_e], FS)
covs = {}
for ch, emb in embs.items():
    C = np.cov(emb, rowvar=False) + 1e-10 * np.eye(3)
    covs[ch] = C

D = np.zeros((n_ch, n_ch))
for i in range(n_ch):
    for j in range(i + 1, n_ch):
        d = np.linalg.norm(covs[i] - covs[j], "fro")
        D[i, j] = d
        D[j, i] = d

sigma_aff = np.median(D[D > 0])
affinity = np.exp(-D ** 2 / (2 * sigma_aff ** 2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

soz_labels = labels[SOZ_CHANNELS]
soz_cluster_orig = np.bincount(soz_labels).argmax()
cluster_map = {soz_cluster_orig: 0}
next_label = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = next_label
        next_label += 1
labels = np.array([cluster_map[l] for l in labels])

clusters_all = {c: np.where(labels == c)[0].tolist() for c in range(CLUSTER_K)}
NAMES_ALL = {0: "C0 (SOZ core)", 1: "C1 (bulk)", 2: "C2 (pacemaker)", 3: "C3 (boundary)"}

# Drop bulk
bulk_id = max(clusters_all, key=lambda c: len(clusters_all[c]))
KEEP = [c for c in sorted(clusters_all.keys()) if c != bulk_id]
clusters = {c: clusters_all[c] for c in KEEP}
N_CL = len(KEEP)
CL_NAMES = {c: NAMES_ALL[c] for c in KEEP}
CL_COLORS = {KEEP[0]: "#1f77b4", KEEP[1]: "#d62728", KEEP[2]: "#2ca02c"}

print(f"Dropped {NAMES_ALL[bulk_id]} ({len(clusters_all[bulk_id])} ch)")
for c in KEEP:
    soz = [m for m in clusters[c] if m in SOZ_CHANNELS]
    print(f"  {CL_NAMES[c]}: {len(clusters[c])} ch, SOZ: {soz}")

# Lift setup
PAIRS = [(i, j) for i in range(N_CL) for j in range(i + 1, N_CL)]
FEAT_NAMES = []
for i, j in PAIRS:
    FEAT_NAMES.append(f"sin(t{KEEP[i]}-t{KEEP[j]})")
for i, j in PAIRS:
    FEAT_NAMES.append(f"cos(t{KEEP[i]}-t{KEEP[j]})")
for c in KEEP:
    FEAT_NAMES.append(f"r{c}")
for c in KEEP:
    FEAT_NAMES.append(f"r{c}*sin(t{c}-tm)")
for c in KEEP:
    FEAT_NAMES.append(f"r{c}*cos(t{c}-tm)")
N_FEAT = len(FEAT_NAMES)
OUT_NAMES = [f"dw{c}" for c in KEEP]


def build_lift(psi_arr, r_arr):
    feats = []
    for i, j in PAIRS:
        feats.append(np.sin(psi_arr[:, i] - psi_arr[:, j]))
    for i, j in PAIRS:
        feats.append(np.cos(psi_arr[:, i] - psi_arr[:, j]))
    for ci in range(N_CL):
        feats.append(r_arr[:, ci])
    theta_mean = psi_arr.mean(axis=1)
    for ci in range(N_CL):
        feats.append(r_arr[:, ci] * np.sin(psi_arr[:, ci] - theta_mean))
    for ci in range(N_CL):
        feats.append(r_arr[:, ci] * np.cos(psi_arr[:, ci] - theta_mean))
    return np.column_stack(feats)


def build_lift_single(psi_vec, r_vec):
    return build_lift(psi_vec[None, :], r_vec[None, :])


# ============================================================
# Per-band analysis
# ============================================================
use_pub_style()
all_results = {}

for band_name, (blo, bhi) in BANDS.items():
    print(f"\n{'=' * 70}")
    print(f"BAND: {band_name.upper()} ({blo:.0f}-{bhi:.0f} Hz)")
    print(f"{'=' * 70}")

    # Bandpass + Hilbert
    nyq = FS / 2.0
    b, a = butter(4, [blo / nyq, min(bhi / nyq, 0.999)], btype="band")
    X_filt = filtfilt(b, a, X, axis=0)

    phases_raw = np.zeros((n_samples, n_ch))
    for ch in range(n_ch):
        analytic = hilbert(X_filt[:, ch])
        phases_raw[:, ch] = np.angle(analytic)

    # Cluster mean phase + order parameter
    psi_raw = np.zeros((n_samples, N_CL))
    r_raw = np.zeros((n_samples, N_CL))
    for ci, c in enumerate(KEEP):
        z = np.exp(1j * phases_raw[:, clusters[c]]).mean(axis=1)
        psi_raw[:, ci] = np.angle(z)
        r_raw[:, ci] = np.abs(z)

    for ci in range(N_CL):
        psi_raw[:, ci] = np.unwrap(psi_raw[:, ci])
    psi_smooth = np.zeros_like(psi_raw)
    for ci in range(N_CL):
        psi_smooth[:, ci] = savgol_filter(psi_raw[:, ci], PHASE_SG_WIN, PHASE_SG_POLY)

    psi_ds = psi_smooth[::DS]
    r_ds = r_raw[::DS]
    t_ds = np.arange(len(psi_ds)) * DT_DS
    n_ds = len(psi_ds)

    for ci, c in enumerate(KEEP):
        freq = np.diff(psi_ds[:, ci]).mean() / (2 * np.pi * DT_DS)
        print(f"  {CL_NAMES[c]}: r={r_ds[:, ci].mean():.3f}±{r_ds[:, ci].std():.3f}, "
              f"freq={freq:.2f} Hz")

    # --- Per-window fits: full + pre_seizure ---
    WINDOWS = {
        "full":        (TRAIN_START, TRAIN_END),
        "pre_seizure": (TRAIN_START, ONSET_TIME),
    }

    for win_name, (win_start, win_end) in WINDOWS.items():
        print(f"\n  --- {win_name} [{win_start:.1f}s, {win_end:.1f}s] ---")

        # Reset RNG for reproducibility (matches single-band scripts)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        s_idx = int(win_start / DT_DS)
        e_idx = min(int(win_end / DT_DS), n_ds)
        psi_win = psi_ds[s_idx:e_idx]
        r_win = r_ds[s_idx:e_idx]

        psi_dot = (psi_win[2:] - psi_win[:-2]) / (2.0 * DT_DS)
        psi_inner = psi_win[1:-1]
        r_inner = r_win[1:-1]
        T_inner = len(psi_inner)

        omega_mean = psi_dot.mean(axis=0)
        delta_omega = psi_dot - omega_mean

        print(f"    Samples: {T_inner}, ratio: {T_inner/N_FEAT:.0f}:1")
        for ci, c in enumerate(KEEP):
            print(f"      {CL_NAMES[c]}: ω={omega_mean[ci]:.2f} rad/s "
                  f"({omega_mean[ci]/(2*np.pi):.2f} Hz), "
                  f"δω std={delta_omega[:, ci].std():.2f}")

        Phi = build_lift(psi_inner, r_inner)

        # Train
        lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT,
                          name=f"kb_{band_name}_{win_name}")
        model = KANDy(lift=lift, grid=GRID, k=K_SPLINE, steps=STEPS,
                      seed=SEED, device="cpu")
        print(f"    Training KAN [{N_FEAT}, {N_CL}]...")
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
        print(f"    R² overall: {r2_overall:.4f}")
        for ci, c in enumerate(KEEP):
            print(f"      {CL_NAMES[c]}: R²={r2_per[ci]:.4f}")

        # Rollout
        if win_name == "full":
            roll_start_t = max(ONSET_TIME - 5, win_start + 1)
            roll_end_t = min(ONSET_TIME + 10, win_end - 1)
        else:
            roll_start_t = max(win_end - 15, win_start + 1)
            roll_end_t = win_end - 1
        roll_s = max(0, int(roll_start_t / DT_DS) - s_idx - 1)
        roll_e = min(T_inner, int(roll_end_t / DT_DS) - s_idx - 1)
        N_ROLL = roll_e - roll_s

        if N_ROLL > 20:
            psi0 = psi_inner[roll_s].copy()
            r_roll = r_inner[roll_s:roll_e]
            psi_true = psi_inner[roll_s:roll_e]
            t_roll = t_ds[s_idx + 1 + roll_s:s_idx + 1 + roll_e]

            traj = [psi0.copy()]
            state = psi0.copy()
            for step in range(N_ROLL - 1):
                r_now = r_roll[min(step, len(r_roll) - 1)]
                def f(ps, r_v=r_now, om=omega_mean):
                    return model.predict(build_lift_single(ps, r_v)).ravel() + om
                k1 = f(state)
                k2 = f(state + 0.5 * DT_DS * k1)
                k3 = f(state + 0.5 * DT_DS * k2)
                k4 = f(state + DT_DS * k3)
                state = state + (DT_DS / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                traj.append(state.copy())
            traj = np.array(traj)

            rmse = np.sqrt(np.mean((traj - psi_true) ** 2))
            print(f"    Rollout RMSE: {rmse:.4f}")

            fig, axes = plt.subplots(N_CL, 1, figsize=(12, 2.5*N_CL), sharex=True)
            fig.suptitle(f"{band_name} {win_name} ({blo:.0f}-{bhi:.0f} Hz): "
                         f"R²={r2_overall:.3f}", fontsize=12)
            for ci, c in enumerate(KEEP):
                ax = axes[ci]
                ax.plot(t_roll, psi_true[:, ci], color=CL_COLORS[c], lw=1.4, label="True")
                ax.plot(t_roll[:len(traj)], traj[:, ci], color=CL_COLORS[c],
                        lw=1.0, ls="--", label="KANDy")
                ax.axvline(ONSET_TIME, color="k", ls=":", lw=0.8, alpha=0.6)
                ax.set_ylabel(rf"$\theta_{{{c}}}$")
                ax.set_title(CL_NAMES[c], fontsize=9)
                if ci == 0:
                    ax.legend(fontsize=7)
            axes[-1].set_xlabel("Time (s)")
            fig.tight_layout()
            save_fig(fig, f"rollout_{band_name}_{win_name}")

        # Edge activations
        n_sub = min(3000, int(T_inner * 0.7))
        sub_idx = np.random.choice(int(T_inner * 0.7), n_sub, replace=False)
        train_t = torch.tensor(Phi[sub_idx], dtype=torch.float32)
        try:
            fig_e, _ = plot_all_edges(
                model.model_, X=train_t,
                in_var_names=FEAT_NAMES, out_var_names=OUT_NAMES,
                save=str(OUT_DIR / f"edges_{band_name}_{win_name}"),
            )
            plt.close(fig_e)
        except Exception as e:
            print(f"    [WARN] Edge plot: {e}")

        # model.plot()
        model.model_.save_act = True
        sym_t = torch.tensor(Phi[:min(2048, T_inner)], dtype=torch.float32)
        with torch.no_grad():
            model.model_(sym_t)
        try:
            model.model_.plot(title=f"{band_name} {win_name}")
            plt.savefig(str(OUT_DIR / f"model_plot_{band_name}_{win_name}.png"),
                        dpi=300, bbox_inches="tight")
            plt.savefig(str(OUT_DIR / f"model_plot_{band_name}_{win_name}.pdf"),
                        dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    [WARN] model.plot: {e}")

        # Symbolic extraction
        print(f"    Symbolic extraction...")
        formulas = None
        try:
            robust_auto_symbolic(
                model.model_, lib=LINEAR_LIB,
                r2_threshold=0.80, weight_simple=0.8,
                topk_edges=N_FEAT * N_CL,
            )
            exprs, vars_ = model.model_.symbolic_formula()
            sub_map = {sp.Symbol(str(v)): sp.Symbol(n)
                       for v, n in zip(vars_, FEAT_NAMES)}

            print(f"\n    ┌──────────────────────────────────────────────────")
            print(f"    │ {band_name.upper()} {win_name.upper()} "
                  f"({blo:.0f}-{bhi:.0f} Hz)  R²={r2_overall:.4f}")
            print(f"    ├──────────────────────────────────────────────────")
            formulas = []
            for ci, c in enumerate(KEEP):
                sym = sp.sympify(exprs[ci]).xreplace(sub_map)
                sym = sp.expand(sym).xreplace(
                    {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
                )
                formulas.append(sym)
                print(f"    │  dθ_{c}/dt = {omega_mean[ci]:.2f} + ({sym})")
            print(f"    └──────────────────────────────────────────────────\n")
        except Exception as e:
            print(f"    [WARN] Symbolic failed: {e}")

        key = f"{band_name}_{win_name}"
        all_results[key] = {
            "r2_overall": r2_overall,
            "r2_per": r2_per,
            "omega_mean": omega_mean,
            "formulas": formulas,
            "band": (blo, bhi),
            "window": win_name,
            "band_name": band_name,
        }


# ============================================================
# GRAND SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("GRAND SUMMARY: ALL BANDS")
print("=" * 70)

print(f"\n{'Key':>25s} | {'R²':>8s} | "
      + " | ".join([f"R² {CL_NAMES[c].split()[0]}" for c in KEEP]))
print("-" * 75)
for key in sorted(all_results.keys()):
    res = all_results[key]
    r2s = " | ".join([f"{res['r2_per'][ci]:10.4f}" for ci in range(N_CL)])
    print(f"{key:>25s} | {res['r2_overall']:8.4f} | {r2s}")

# Print all equations together
print(f"\n{'=' * 70}")
print("ALL DISCOVERED EQUATIONS")
print(f"{'=' * 70}")
for key in sorted(all_results.keys()):
    res = all_results[key]
    if res["formulas"] is None:
        continue
    blo, bhi = res["band"]
    print(f"\n  {res['band_name'].upper()} {res['window'].upper()} "
          f"({blo:.0f}-{bhi:.0f} Hz), R²={res['r2_overall']:.4f}:")
    for ci, c in enumerate(KEEP):
        print(f"    dθ_{c}/dt = {res['omega_mean'][ci]:.2f} + ({res['formulas'][ci]})")

# Coupling term inventory
print(f"\n{'=' * 70}")
print("COUPLING TERM INVENTORY (sin/cos phase coupling terms)")
print(f"{'=' * 70}")
coupling_terms = ["sin", "cos"]
for key in sorted(all_results.keys()):
    res = all_results[key]
    if res["formulas"] is None:
        continue
    has_coupling = False
    for ci, c in enumerate(KEEP):
        formula_str = str(res["formulas"][ci])
        for term in coupling_terms:
            if term + "(t" in formula_str:
                has_coupling = True
                print(f"  {key:>25s} {CL_NAMES[c]:20s}: {formula_str}")
                break
    if not has_coupling:
        print(f"  {key:>25s}: no sin/cos coupling terms")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
