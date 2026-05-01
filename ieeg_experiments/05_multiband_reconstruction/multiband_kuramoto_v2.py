#!/usr/bin/env python3
"""Multiband Kuramoto v2: richer lift with second-order parameter + amplitude-modulated coupling.

v2 over v1:
  - Second-order Kuramoto order parameter r2_k = |<exp(i*2*θ)>| per cluster
    (captures bimodal phase clustering / anti-phase groups)
  - Amplitude-modulated coupling: r_i * sin(θ_i - θ_j) — coupling strength
    depends on source cluster's coherence
  - Target-modulated coupling: r_j * sin(θ_i - θ_j) — coupling depends on
    receiver's coherence

Lift (24 features per band):
  sin(θ_i - θ_j)           — 3 pairs  (standard Kuramoto)
  cos(θ_i - θ_j)           — 3 pairs  (Sakaguchi phase lag)
  r1_k                     — 3        (first-order coherence)
  r2_k                     — 3        (second-order coherence)
  r1_k * sin(θ_k - θ_mean) — 3        (mean-field amplitude-phase)
  r1_k * cos(θ_k - θ_mean) — 3        (mean-field amplitude-phase)
  r1_i * sin(θ_i - θ_j)    — 3 pairs  (source-modulated coupling)
  r1_j * sin(θ_i - θ_j)    — 3 pairs  (target-modulated coupling)

KAN: [24, 3], grid=5, 200 steps, per band, full + pre_seizure windows

Author: KANDy Researcher Agent
Date: 2026-04-24
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

SG_WINDOW = 51
SG_POLY = 5

BANDS = {
    "delta":     (1.0,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "beta":      (13.0, 30.0),
    "low_gamma": (30.0, 50.0),
}
BAND_NAMES = list(BANDS.keys())
BAND_LABELS = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma_L$"]
N_BANDS = len(BANDS)

CLUSTER_K = 4
SOZ_CHANNELS = list(range(21, 31))

PHASE_SG_WIN = 51
PHASE_SG_POLY = 3

DS = 10
DT_DS = DS / FS

GRID = 3
K_SPLINE = 3
STEPS = 150
LAMB = 0.0

TRAIN_START, TRAIN_END = 30.0, 97.0

WINDOWS = {
    "full":        (TRAIN_START, TRAIN_END),
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
# Load + cluster (same as v1)
# ============================================================
print("=" * 70)
print("MULTIBAND KURAMOTO v2: SECOND-ORDER PARAMETER + RICH COUPLING")
print("=" * 70)

X = np.loadtxt(DATA_PATH)
n_samples, n_ch = X.shape
t_axis = np.arange(n_samples) / FS
ONSET_SAMPLE = int(ONSET_TIME * FS)
print(f"Data: {X.shape}, {n_samples/FS:.1f}s")

# Clustering
print("\nClustering...")
sez_s = ONSET_SAMPLE
sez_e = min(n_samples, sez_s + int(10 * FS))
D = np.zeros((n_ch, n_ch))
for ch in range(n_ch):
    seg = X[sez_s:sez_e, ch].copy()
    mu, sigma = seg.mean(), max(seg.std(), 1e-12)
    seg = (seg - mu) / sigma
    emb = np.column_stack([
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=0),
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=1, delta=1/FS),
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=2, delta=1/FS),
    ])
    cov_ch = np.cov(emb, rowvar=False) + 1e-10 * np.eye(3)
    for ch2 in range(ch):
        seg2 = X[sez_s:sez_e, ch2].copy()
        mu2, sigma2 = seg2.mean(), max(seg2.std(), 1e-12)
        seg2 = (seg2 - mu2) / sigma2
        emb2 = np.column_stack([
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=0),
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=1, delta=1/FS),
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=2, delta=1/FS),
        ])
        cov2 = np.cov(emb2, rowvar=False) + 1e-10 * np.eye(3)
        d = np.linalg.norm(cov_ch - cov2, "fro")
        D[ch, ch2] = d
        D[ch2, ch] = d

sigma_aff = np.median(D[D > 0])
affinity = np.exp(-D**2 / (2 * sigma_aff**2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

soz_cluster_orig = np.bincount(labels[SOZ_CHANNELS]).argmax()
cluster_map = {soz_cluster_orig: 0}
nxt = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = nxt
        nxt += 1
labels = np.array([cluster_map[l] for l in labels])
clusters_all = {c: np.where(labels == c)[0].tolist() for c in range(CLUSTER_K)}

bulk_id = max(clusters_all, key=lambda c: len(clusters_all[c]))
KEEP = [c for c in sorted(clusters_all.keys()) if c != bulk_id]
clusters = {c: clusters_all[c] for c in KEEP}
N_CL = len(KEEP)
CL_NAMES = {0: "C0 (SOZ)", 2: "C2 (pace)", 3: "C3 (bound)"}
CL_COLORS = {KEEP[0]: "#1f77b4", KEEP[1]: "#d62728", KEEP[2]: "#2ca02c"}

print(f"Dropped bulk ({len(clusters_all[bulk_id])} ch)")
for c in KEEP:
    soz = [m for m in clusters[c] if m in SOZ_CHANNELS]
    print(f"  {CL_NAMES[c]}: {len(clusters[c])} ch, SOZ: {soz}")

# ============================================================
# Lift design (24 features)
# ============================================================
PAIRS = [(i, j) for i in range(N_CL) for j in range(i + 1, N_CL)]
N_PAIRS = len(PAIRS)

FEAT_NAMES = []
# sin(Δθ) — 3
for i, j in PAIRS:
    FEAT_NAMES.append(f"sin(t{KEEP[i]}-t{KEEP[j]})")
# cos(Δθ) — 3
for i, j in PAIRS:
    FEAT_NAMES.append(f"cos(t{KEEP[i]}-t{KEEP[j]})")
# r1_k — 3
for c in KEEP:
    FEAT_NAMES.append(f"r1_{c}")
# r2_k (second-order) — 3
for c in KEEP:
    FEAT_NAMES.append(f"r2_{c}")
# r1_k * sin(θ_k - θ_mean) — 3
for c in KEEP:
    FEAT_NAMES.append(f"r1_{c}*sin(t{c}-tm)")
# r1_k * cos(θ_k - θ_mean) — 3
for c in KEEP:
    FEAT_NAMES.append(f"r1_{c}*cos(t{c}-tm)")
# r1_i * sin(θ_i - θ_j) — source-modulated coupling — 3
for i, j in PAIRS:
    FEAT_NAMES.append(f"r1_{KEEP[i]}*sin(t{KEEP[i]}-t{KEEP[j]})")
# r1_j * sin(θ_i - θ_j) — target-modulated coupling — 3
for i, j in PAIRS:
    FEAT_NAMES.append(f"r1_{KEEP[j]}*sin(t{KEEP[i]}-t{KEEP[j]})")

N_FEAT = len(FEAT_NAMES)
OUT_NAMES = [f"dw{c}" for c in KEEP]
print(f"\nLift: {N_FEAT} features")
print(f"  {FEAT_NAMES}")
print(f"KAN: [{N_FEAT}, {N_CL}], grid={GRID}")


def build_lift(psi_arr, r1_arr, r2_arr):
    """(T,3) phases + (T,3) r1 + (T,3) r2 → (T, N_FEAT)."""
    feats = []
    # sin(Δθ)
    for i, j in PAIRS:
        feats.append(np.sin(psi_arr[:, i] - psi_arr[:, j]))
    # cos(Δθ)
    for i, j in PAIRS:
        feats.append(np.cos(psi_arr[:, i] - psi_arr[:, j]))
    # r1_k
    for ci in range(N_CL):
        feats.append(r1_arr[:, ci])
    # r2_k
    for ci in range(N_CL):
        feats.append(r2_arr[:, ci])
    # r1_k * sin(θ_k - θ_mean)
    theta_mean = psi_arr.mean(axis=1)
    for ci in range(N_CL):
        feats.append(r1_arr[:, ci] * np.sin(psi_arr[:, ci] - theta_mean))
    # r1_k * cos(θ_k - θ_mean)
    for ci in range(N_CL):
        feats.append(r1_arr[:, ci] * np.cos(psi_arr[:, ci] - theta_mean))
    # r1_i * sin(θ_i - θ_j) — source modulated
    for i, j in PAIRS:
        feats.append(r1_arr[:, i] * np.sin(psi_arr[:, i] - psi_arr[:, j]))
    # r1_j * sin(θ_i - θ_j) — target modulated
    for i, j in PAIRS:
        feats.append(r1_arr[:, j] * np.sin(psi_arr[:, i] - psi_arr[:, j]))
    return np.column_stack(feats)


def build_lift_single(psi_vec, r1_vec, r2_vec):
    return build_lift(psi_vec[None, :], r1_vec[None, :], r2_vec[None, :])


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

    # Cluster mean phase + first and second order parameters
    psi_raw = np.zeros((n_samples, N_CL))
    r1_raw = np.zeros((n_samples, N_CL))
    r2_raw = np.zeros((n_samples, N_CL))
    for ci, c in enumerate(KEEP):
        ch_phases = phases_raw[:, clusters[c]]
        # First order: r1 = |<exp(iθ)>|
        z1 = np.exp(1j * ch_phases).mean(axis=1)
        psi_raw[:, ci] = np.angle(z1)
        r1_raw[:, ci] = np.abs(z1)
        # Second order: r2 = |<exp(i*2*θ)>|
        z2 = np.exp(2j * ch_phases).mean(axis=1)
        r2_raw[:, ci] = np.abs(z2)

    for ci in range(N_CL):
        psi_raw[:, ci] = np.unwrap(psi_raw[:, ci])
    psi_smooth = np.zeros_like(psi_raw)
    for ci in range(N_CL):
        psi_smooth[:, ci] = savgol_filter(psi_raw[:, ci], PHASE_SG_WIN, PHASE_SG_POLY)

    psi_ds = psi_smooth[::DS]
    r1_ds = r1_raw[::DS]
    r2_ds = r2_raw[::DS]
    t_ds = np.arange(len(psi_ds)) * DT_DS
    n_ds = len(psi_ds)

    for ci, c in enumerate(KEEP):
        freq = np.diff(psi_ds[:, ci]).mean() / (2 * np.pi * DT_DS)
        print(f"  {CL_NAMES[c]}: r1={r1_ds[:, ci].mean():.3f}, "
              f"r2={r2_ds[:, ci].mean():.3f}, freq={freq:.2f} Hz")

    # --- Per-window fits ---
    for win_name, (win_start, win_end) in WINDOWS.items():
        print(f"\n  --- {win_name} [{win_start:.1f}s, {win_end:.1f}s] ---")

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        s_idx = int(win_start / DT_DS)
        e_idx = min(int(win_end / DT_DS), n_ds)
        psi_win = psi_ds[s_idx:e_idx]
        r1_win = r1_ds[s_idx:e_idx]
        r2_win = r2_ds[s_idx:e_idx]

        psi_dot = (psi_win[2:] - psi_win[:-2]) / (2.0 * DT_DS)
        psi_inner = psi_win[1:-1]
        r1_inner = r1_win[1:-1]
        r2_inner = r2_win[1:-1]
        T_inner = len(psi_inner)

        omega_mean = psi_dot.mean(axis=0)
        delta_omega = psi_dot - omega_mean

        print(f"    Samples: {T_inner}, ratio: {T_inner/N_FEAT:.0f}:1")
        for ci, c in enumerate(KEEP):
            print(f"      {CL_NAMES[c]}: ω={omega_mean[ci]:.2f} rad/s "
                  f"({omega_mean[ci]/(2*np.pi):.2f} Hz), "
                  f"δω std={delta_omega[:, ci].std():.2f}")

        Phi = build_lift(psi_inner, r1_inner, r2_inner)

        # Train
        lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT,
                          name=f"v2_{band_name}_{win_name}")
        model = KANDy(lift=lift, grid=GRID, k=K_SPLINE, steps=STEPS,
                      seed=SEED, device="cpu")
        print(f"    Training KAN [{N_FEAT}, {N_CL}], grid={GRID}...")
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
            r1_roll = r1_inner[roll_s:roll_e]
            r2_roll = r2_inner[roll_s:roll_e]
            psi_true = psi_inner[roll_s:roll_e]
            t_roll = t_ds[s_idx + 1 + roll_s:s_idx + 1 + roll_e]

            traj = [psi0.copy()]
            state = psi0.copy()
            for step in range(N_ROLL - 1):
                r1_now = r1_roll[min(step, len(r1_roll) - 1)]
                r2_now = r2_roll[min(step, len(r2_roll) - 1)]
                def f(ps, r1v=r1_now, r2v=r2_now, om=omega_mean):
                    return model.predict(build_lift_single(ps, r1v, r2v)).ravel() + om
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

        # Symbolic extraction — fix_symbolic on active edges only (fast)
        print(f"    Symbolic extraction (active edges only)...")
        formulas = None
        try:
            from kandy.plotting import get_edge_activation
            # Identify active edges by activation range
            edge_ranges = {}
            for fi in range(N_FEAT):
                for ci_out in range(N_CL):
                    try:
                        x_a, y_a = get_edge_activation(model.model_, l=0, i=fi, j=ci_out)
                        edge_ranges[(fi, ci_out)] = np.max(y_a) - np.min(y_a)
                    except:
                        edge_ranges[(fi, ci_out)] = 0.0
            max_range = max(edge_ranges.values())
            threshold = 0.05 * max_range if max_range > 1e-8 else 1e-8

            # Fix symbolic on active edges only
            for fi in range(N_FEAT):
                for ci_out in range(N_CL):
                    if edge_ranges[(fi, ci_out)] > threshold:
                        model.model_.fix_symbolic(0, fi, ci_out, "x", verbose=False)
                    else:
                        model.model_.fix_symbolic(0, fi, ci_out, "0", verbose=False)

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
print("GRAND SUMMARY: ALL BANDS (v2 — with r2 + amplitude-modulated coupling)")
print("=" * 70)

print(f"\n{'Key':>25s} | {'R²':>8s} | "
      + " | ".join([f"R² {CL_NAMES[c].split()[0]}" for c in KEEP]))
print("-" * 75)
for key in sorted(all_results.keys()):
    res = all_results[key]
    r2s = " | ".join([f"{res['r2_per'][ci]:10.4f}" for ci in range(N_CL)])
    print(f"{key:>25s} | {res['r2_overall']:8.4f} | {r2s}")

# All equations
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

# Coupling inventory
print(f"\n{'=' * 70}")
print("COUPLING TERM INVENTORY")
print(f"{'=' * 70}")
coupling_markers = ["sin(t", "cos(t", "r1_", "r2_"]
for key in sorted(all_results.keys()):
    res = all_results[key]
    if res["formulas"] is None:
        continue
    found_any = False
    for ci, c in enumerate(KEEP):
        formula_str = str(res["formulas"][ci])
        # Check for any non-constant terms
        has_sin = "sin(t" in formula_str
        has_cos = "cos(t" in formula_str
        has_r2 = "r2_" in formula_str
        has_rsin = "r1_" in formula_str and "sin" in formula_str
        markers = []
        if has_sin: markers.append("sin")
        if has_cos: markers.append("cos")
        if has_r2: markers.append("r2")
        if has_rsin: markers.append("r·sin")
        if markers:
            found_any = True
            print(f"  {key:>25s} {CL_NAMES[c]:15s} [{','.join(markers)}]: {formula_str}")
    if not found_any:
        print(f"  {key:>25s}: coherence-only (r1 terms)")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
