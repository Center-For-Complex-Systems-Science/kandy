#!/usr/bin/env python3
"""SOZ/NSOZ Kuramoto-ReLU KANDy model for iEEG virtual surgery.

Simplest possible model: 2 nodes — SOZ (ch21-30) and NSOZ (rest).
Community phases from circular mean over channel groups.

Model:
    dφ_SOZ/dt  = ω_SOZ  + K_1 * sin(Δφ) + β_1 * ReLU(A_SOZ - θ)
    dφ_NSOZ/dt = ω_NSOZ + K_2 * sin(Δφ) + β_2 * ReLU(A_SOZ - θ)

where Δφ = φ_SOZ - φ_NSOZ.

Virtual surgery: zero the coupling sin(Δφ) and ReLU gating,
simulate NSOZ dynamics in isolation.
"""

import os
import sys
import numpy as np
import torch
import sympy as sp
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = ROOT / "data" / "E3Data.mat"
OUT_DIR = ROOT / "results" / "iEEG" / "kuramoto"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 500
ALPHA_LOW, ALPHA_HIGH = 8.0, 13.0
FILTER_ORDER = 4
DS = 10  # 500 -> 50 Hz
DT_DS = DS / FS

N_CH = 119
SEIZURE_CHS = list(range(21, 31))
NSOZ_CHS = [ch for ch in range(N_CH) if ch not in SEIZURE_CHS]
ONSET_TIMES = {1: 80.25, 2: 88.25, 3: 87.0}
N_COMM = 2  # SOZ=0, NSOZ=1

SG_WINDOW = 15
SG_POLY = 4
TRIM = 10

print(f"[CONFIG] N_CH={N_CH}, SOZ={len(SEIZURE_CHS)}ch, NSOZ={len(NSOZ_CHS)}ch, DS={DS} -> {FS//DS} Hz")

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
print("\n[DATA] Loading E3Data.mat ...")
mat = sio.loadmat(DATA_PATH)
episodes_raw = [mat["X1"][:, :N_CH], mat["X2"][:, :N_CH], mat["X3"][:, :N_CH]]
for i, ep in enumerate(episodes_raw):
    print(f"  Episode {i+1}: {ep.shape}")

# ---------------------------------------------------------------------------
# 2. Alpha-band phase + amplitude extraction
# ---------------------------------------------------------------------------
print("\n[PHASE] Extracting alpha-band phases ...")
nyq = FS / 2
b_filt, a_filt = butter(FILTER_ORDER, [ALPHA_LOW / nyq, ALPHA_HIGH / nyq], btype="band")

phases_raw = []  # wrapped phase per episode: (T_ds, N_CH)
phases_unwrap = []  # unwrapped phase
amps_raw = []  # amplitude envelope

for ep_idx, X_raw in enumerate(episodes_raw):
    X_alpha = filtfilt(b_filt, a_filt, X_raw, axis=0)
    analytic = hilbert(X_alpha, axis=0)
    phase_wrapped = np.angle(analytic)  # (-pi, pi]
    amplitude = np.abs(analytic)
    phase_uw = np.unwrap(phase_wrapped, axis=0)

    # Downsample
    phases_raw.append(phase_wrapped[::DS])
    phases_unwrap.append(phase_uw[::DS])
    amps_raw.append(amplitude[::DS])
    print(f"  Episode {ep_idx+1}: {X_raw.shape[0]} -> {phase_wrapped[::DS].shape[0]} samples")

# ---------------------------------------------------------------------------
# 3. Community assignment: SOZ vs NSOZ (known a priori)
# ---------------------------------------------------------------------------
print("\n[CLUSTER] Assigning communities (SOZ=0, NSOZ=1) ...")

# PLV(i,j) = |<exp(i*(phi_i - phi_j))>_t|, averaged across episodes (for viz only)
plv_sum = np.zeros((N_CH, N_CH))
for ph_wrap in phases_raw:
    z = np.exp(1j * ph_wrap)  # (T, N_CH)
    plv_ep = np.abs(z.T.conj() @ z) / ph_wrap.shape[0]  # (N_CH, N_CH)
    plv_sum += plv_ep
PLV = plv_sum / len(phases_raw)
np.fill_diagonal(PLV, 0)

# Direct assignment: SOZ = community 0, NSOZ = community 1
labels = np.ones(N_CH, dtype=int)
for ch in SEIZURE_CHS:
    labels[ch] = 0

communities = {0: SEIZURE_CHS, 1: NSOZ_CHS}
for k, members in communities.items():
    tag = " *SEIZURE*" if k == 0 else ""
    print(f"  Community {k}: {len(members)} channels{tag}")

# ---------------------------------------------------------------------------
# 4. Community phase and amplitude computation
# ---------------------------------------------------------------------------
print("\n[COMM] Computing community phases (circular mean) ...")


def circular_mean(phases_wrapped, members):
    """Circular mean of wrapped phases for a set of channel indices."""
    z = np.mean(np.exp(1j * phases_wrapped[:, members]), axis=1)
    return np.angle(z)


def community_amplitude(amps, members):
    """Mean amplitude across community members."""
    return np.mean(amps[:, members], axis=1)


# Compute community phases and amplitudes per episode
comm_phase_parts = []  # unwrapped community phase per episode
comm_amp_parts = []
comm_dot_parts = []
onset_sample_parts = []

for ep_idx in range(3):
    T_ep = phases_raw[ep_idx].shape[0]

    # Wrapped community phase -> unwrap for derivative
    comm_wrapped = np.column_stack(
        [circular_mean(phases_raw[ep_idx], communities[k]) for k in range(N_COMM)]
    )
    comm_unwrapped = np.unwrap(comm_wrapped, axis=0)

    # Community amplitude
    comm_amp = np.column_stack(
        [community_amplitude(amps_raw[ep_idx], communities[k]) for k in range(N_COMM)]
    )

    # Phase derivatives via Savitzky-Golay
    comm_dot = np.zeros_like(comm_unwrapped)
    for k in range(N_COMM):
        comm_dot[:, k] = savgol_filter(
            comm_unwrapped[:, k], SG_WINDOW, SG_POLY, deriv=1, delta=DT_DS
        )

    # Trim edges
    s, e = TRIM, T_ep - TRIM
    comm_phase_parts.append(comm_unwrapped[s:e])
    comm_amp_parts.append(comm_amp[s:e])
    comm_dot_parts.append(comm_dot[s:e])

    onset_s = ONSET_TIMES[ep_idx + 1]
    onset_sample = int(onset_s / DT_DS) - TRIM
    onset_sample_parts.append(onset_sample)
    print(f"  Episode {ep_idx+1}: {e-s} samples, onset at sample {onset_sample}")

# Concatenate
COMM_PHASE = np.vstack(comm_phase_parts)
COMM_DOT = np.vstack(comm_dot_parts)
COMM_AMP = np.vstack(comm_amp_parts)
T_TOTAL = COMM_PHASE.shape[0]
print(f"[DATA] Total: {T_TOTAL} samples x {N_COMM} communities")

# Check order parameter spread
r_all = np.abs(np.mean(np.exp(1j * COMM_PHASE), axis=1))
print(f"[DATA] Community order param: min={r_all.min():.3f}, max={r_all.max():.3f}, mean={r_all.mean():.3f}")

# ReLU threshold: seizure community amplitude at onset
onset_amps = []
for ep_idx in range(3):
    oi = onset_sample_parts[ep_idx]
    if 0 <= oi < comm_amp_parts[ep_idx].shape[0]:
        onset_amps.append(comm_amp_parts[ep_idx][oi, 0])  # community 0 = seizure
RELU_THRESHOLD = np.mean(onset_amps)
print(f"[DATA] ReLU threshold (seizure community amp at onset): {RELU_THRESHOLD:.4f}")

# ---------------------------------------------------------------------------
# 5. Feature assembly: Kuramoto-ReLU lift
# ---------------------------------------------------------------------------
PAIRS = [(i, j) for i in range(N_COMM) for j in range(i + 1, N_COMM)]
N_PAIRS = len(PAIRS)  # 15

FEATURE_NAMES = (
    [f"sin(c{i}-c{j})" for (i, j) in PAIRS]  # 15 sin features
    + ["A_seiz"]  # seizure community amplitude
    + ["ReLU(A)"]  # ReLU gating
    + [f"ReLU*sin(c0-c{j})" for j in range(1, N_COMM)]  # gated coupling (5)
)
N_FEAT = len(FEATURE_NAMES)
print(f"\n[LIFT] Features ({N_FEAT}): {FEATURE_NAMES}")
print(f"[LIFT] KAN: [{N_FEAT}, {N_COMM}]")


def build_features(comm_phase, comm_amp):
    """Build Kuramoto-ReLU features from community phases and amplitudes."""
    T = comm_phase.shape[0]
    # Sin of unique phase differences
    sin_feats = np.column_stack(
        [np.sin(comm_phase[:, i] - comm_phase[:, j]) for (i, j) in PAIRS]
    )
    # Seizure community amplitude
    a_seiz = comm_amp[:, 0].reshape(-1, 1)
    # ReLU gating
    relu_val = np.maximum(a_seiz - RELU_THRESHOLD, 0.0)
    # Gated coupling: ReLU * sin(seizure - other)
    gated = np.column_stack(
        [relu_val.ravel() * np.sin(comm_phase[:, 0] - comm_phase[:, j])
         for j in range(1, N_COMM)]
    )
    return np.hstack([sin_feats, a_seiz, relu_val, gated])


Phi = build_features(COMM_PHASE, COMM_AMP)
print(f"[LIFT] Feature matrix: {Phi.shape}")

# ---------------------------------------------------------------------------
# 6. KANDy model fitting
# ---------------------------------------------------------------------------
lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kuramoto_community_id")

model = KANDy(
    lift=lift,
    grid=5,
    k=3,
    steps=200,
    seed=SEED,
    device="cpu",
)

print("\n[TRAIN] Fitting KANDy ...")
model.fit(
    X=Phi,
    X_dot=COMM_DOT,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# One-step evaluation
n_test = min(1000, T_TOTAL // 5)
pred_dot = model.predict(Phi[-n_test:])
true_dot = COMM_DOT[-n_test:]
mse_onestep = np.mean((pred_dot - true_dot) ** 2)
r2_per_comm = []
for k in range(N_COMM):
    ss_res = np.sum((pred_dot[:, k] - true_dot[:, k]) ** 2)
    ss_tot = np.sum((true_dot[:, k] - true_dot[:, k].mean()) ** 2)
    r2_per_comm.append(1 - ss_res / (ss_tot + 1e-14))
print(f"\n[EVAL] One-step MSE: {mse_onestep:.6e}")
for k in range(N_COMM):
    tag = " *SEIZURE*" if k == 0 else ""
    print(f"  Community {k}: R² = {r2_per_comm[k]:.4f}{tag}")

# ---------------------------------------------------------------------------
# 7. Rollout — BEFORE symbolic extraction
# ---------------------------------------------------------------------------
print("\n[ROLLOUT] Autoregressive rollout on Episode 1 ...")

ep1_comm_phase = comm_phase_parts[0]
ep1_comm_amp = comm_amp_parts[0]
onset_ep1 = onset_sample_parts[0]

PRE, POST = 200, 500
roll_start = max(0, onset_ep1 - PRE)
roll_end = min(ep1_comm_phase.shape[0], onset_ep1 + POST)
N_ROLL = roll_end - roll_start


def build_lift_single(comm_ph, amp_seiz):
    """Lift a single community state -> (1, N_FEAT)."""
    sin_f = [np.sin(comm_ph[i] - comm_ph[j]) for (i, j) in PAIRS]
    relu_val = max(amp_seiz - RELU_THRESHOLD, 0.0)
    gated = [relu_val * np.sin(comm_ph[0] - comm_ph[j]) for j in range(1, N_COMM)]
    feats = np.array(sin_f + [amp_seiz, relu_val] + gated)
    return feats[None, :]


def rollout_kandy(phi0, amp_traj, n_steps, dt):
    """RK4 rollout using KANDy model."""
    phi = phi0.copy().astype(np.float64)
    traj = [phi.copy()]

    for t in range(n_steps - 1):
        amp_t = amp_traj[min(t, len(amp_traj) - 1), 0]  # seizure amp

        def f(ph):
            feat = build_lift_single(ph, amp_t)
            return model.predict(feat).ravel()

        k1 = f(phi)
        k2 = f(phi + 0.5 * dt * k1)
        k3 = f(phi + 0.5 * dt * k2)
        k4 = f(phi + dt * k3)
        phi = phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(phi.copy())

    return np.array(traj)


phi0 = ep1_comm_phase[roll_start]
true_comm = ep1_comm_phase[roll_start:roll_end]
amp_roll = ep1_comm_amp[roll_start:roll_end]

pred_comm = rollout_kandy(phi0, amp_roll, N_ROLL, DT_DS)

rmse_roll = np.sqrt(np.mean((pred_comm - true_comm) ** 2))
print(f"[EVAL] Rollout RMSE (phase): {rmse_roll:.4f}")

# Community order parameters
r_true = np.abs(np.mean(np.exp(1j * true_comm), axis=1))
r_pred = np.abs(np.mean(np.exp(1j * pred_comm), axis=1))
rmse_r = np.sqrt(np.mean((r_pred - r_true) ** 2))
print(f"[EVAL] Order parameter RMSE: {rmse_r:.4f}")

# ---------------------------------------------------------------------------
# 8. Virtual surgery: zero seizure-community features
# ---------------------------------------------------------------------------
print("\n[SURGERY] Virtual surgery rollout (seizure community disconnected) ...")


def build_lift_surgery(comm_ph, amp_seiz):
    """Lift with seizure community disconnected (all seizure coupling zeroed)."""
    sin_f = []
    for (i, j) in PAIRS:
        if i == 0 or j == 0:
            sin_f.append(0.0)  # zero out seizure coupling
        else:
            sin_f.append(np.sin(comm_ph[i] - comm_ph[j]))
    feats = np.array(sin_f + [0.0, 0.0] + [0.0] * (N_COMM - 1))  # zero all gating
    return feats[None, :]


def rollout_surgery(phi0, n_steps, dt):
    """RK4 rollout with seizure community disconnected."""
    phi = phi0.copy().astype(np.float64)
    traj = [phi.copy()]

    for t in range(n_steps - 1):
        def f(ph):
            feat = build_lift_surgery(ph, 0.0)
            return model.predict(feat).ravel()

        k1 = f(phi)
        k2 = f(phi + 0.5 * dt * k1)
        k3 = f(phi + 0.5 * dt * k2)
        k4 = f(phi + dt * k3)
        phi = phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(phi.copy())

    return np.array(traj)


pred_surgery = rollout_surgery(phi0, N_ROLL, DT_DS)
r_surgery = np.abs(np.mean(np.exp(1j * pred_surgery[:, 1:]), axis=1))  # non-seizure only
r_true_rest = np.abs(np.mean(np.exp(1j * true_comm[:, 1:]), axis=1))

# ---------------------------------------------------------------------------
# 9. Edge activation plots + symbolic extraction
# ---------------------------------------------------------------------------
print("\n[EDGES] Plotting edge activations ...")
use_pub_style()

n_sub = min(5000, int(T_TOTAL * 0.70))
sub_idx = np.random.choice(int(T_TOTAL * 0.70), n_sub, replace=False)
train_phi_t = torch.tensor(Phi[sub_idx], dtype=torch.float32)
try:
    fig, axes = plot_all_edges(
        model.model_,
        X=train_phi_t,
        in_var_names=FEATURE_NAMES,
        out_var_names=[f"dc{k}/dt" for k in range(N_COMM)],
        save=f"{OUT_DIR}/edge_activations",
    )
    plt.close(fig)
except Exception as e:
    print(f"  [WARN] Edge plot failed: {e}")

print("\n[SYMBOLIC] Extracting equations ...")
LINEAR_LIB = make_symbolic_lib({
    "x": (lambda x: x, lambda x: x, 1),
    "0": (lambda x: x * 0, lambda x: x * 0, 0),
})

sym_subset = torch.tensor(Phi[:2048], dtype=torch.float32)
model.model_.save_act = True
with torch.no_grad():
    model.model_(sym_subset)

robust_auto_symbolic(
    model.model_,
    lib=LINEAR_LIB,
    r2_threshold=0.85,
    weight_simple=0.8,
    topk_edges=40,
)

exprs, vars_ = model.model_.symbolic_formula()
sub_map = {sp.Symbol(str(v)): sp.Symbol(n) for v, n in zip(vars_, FEATURE_NAMES)}
formulas = []
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym).xreplace({n: round(float(n), 4) for n in sym.atoms(sp.Number)})
    formulas.append(sym)

print("\nDiscovered equations:")
for k in range(N_COMM):
    tag = " *SEIZURE*" if k == 0 else ""
    print(f"  dc{k}/dt = {formulas[k]}{tag}")

# ---------------------------------------------------------------------------
# 10. Visualization
# ---------------------------------------------------------------------------
print("\n[FIGS] Generating plots ...")
t_roll = (np.arange(N_ROLL) - PRE) * DT_DS

# 10a. PLV matrix with community boundaries
fig, ax = plt.subplots(figsize=(8, 7))
# Reorder channels by community for visualization
order = []
for k in range(N_COMM):
    order.extend(communities[k])
PLV_ordered = PLV[np.ix_(order, order)]
im = ax.imshow(PLV_ordered, cmap="hot", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="PLV")
# Draw community boundaries
cum = 0
for k in range(N_COMM):
    sz = len(communities[k])
    ax.axhline(cum - 0.5, color="cyan", lw=0.8)
    ax.axvline(cum - 0.5, color="cyan", lw=0.8)
    ax.text(cum + sz / 2, cum + sz / 2, f"C{k}", ha="center", va="center",
            color="cyan", fontsize=9, fontweight="bold")
    cum += sz
ax.set_title(f"Phase-Locking Value Matrix ({N_COMM} communities)")
ax.set_xlabel("Channel (reordered)")
ax.set_ylabel("Channel (reordered)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/plv_matrix.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/plv_matrix.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10b. Community phase trajectories
fig, axes = plt.subplots(N_COMM, 1, figsize=(10, 2 * N_COMM), sharex=True)
colors = plt.cm.tab10.colors
for k, ax in enumerate(axes):
    ax.plot(t_roll, np.sin(true_comm[:, k]), color=colors[k], lw=1.0, label="True")
    ax.plot(t_roll, np.sin(pred_comm[:, k]), color=colors[k], lw=0.8, ls="--", label="KANDy")
    tag = " (seizure)" if k == 0 else ""
    ax.set_ylabel(f"sin($\\phi_{k}$){tag}")
    if k == 0:
        ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("Time relative to onset (s)")
fig.suptitle("Community Phase Rollout", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/community_rollout.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/community_rollout.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10c. Virtual surgery: order parameter
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t_roll, r_true, "k-", lw=1.2, label="True")
axes[0].plot(t_roll, r_pred, "b--", lw=1.0, label="KANDy full")
axes[0].set_ylabel("$r(t)$ all communities")
axes[0].legend(fontsize=8)
axes[0].set_title("Virtual Surgery: Community Order Parameter")

axes[1].plot(t_roll, r_true_rest, "k-", lw=1.2, label="True (non-seizure)")
axes[1].plot(t_roll, r_surgery, "r--", lw=1.0, label="Surgery (seizure removed)")
axes[1].set_ylabel("$r(t)$ non-seizure")
axes[1].set_xlabel("Time relative to onset (s)")
axes[1].legend(fontsize=8)

for ax in axes:
    ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
    ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/virtual_surgery.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/virtual_surgery.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10d. Per-community R²
fig, ax = plt.subplots(figsize=(8, 4))
colors_r2 = ["tomato" if k == 0 else "steelblue" for k in range(N_COMM)]
bars = ax.bar(range(N_COMM), r2_per_comm, color=colors_r2, width=0.6)
ax.set_xlabel("Community")
ax.set_ylabel("$R^2$ (one-step)")
ax.set_title("Per-Community Prediction Accuracy")
ax.set_xticks(range(N_COMM))
ax.set_xticklabels([f"C{k}" + (" (seiz)" if k == 0 else "") for k in range(N_COMM)])
for bar, r2 in zip(bars, r2_per_comm):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{r2:.3f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/community_r2.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/community_r2.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10e. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(model.train_results_, save=f"{OUT_DIR}/loss_curves")
    plt.close(fig)

# 10f. Community assignment map
fig, ax = plt.subplots(figsize=(14, 3))
comm_colors = plt.cm.Set2.colors
bar_colors = [comm_colors[labels[ch] % len(comm_colors)] for ch in range(N_CH)]
ax.bar(range(N_CH), np.ones(N_CH), color=bar_colors, width=1.0, edgecolor="none")
ax.set_xlabel("Channel")
ax.set_ylabel("")
ax.set_yticks([])
ax.set_title("Community Assignment (colored by community)")
ax.set_xlim(-0.5, N_CH - 0.5)
# Seizure zone overlay
for ch in SEIZURE_CHS:
    ax.axvline(ch, color="red", lw=0.5, alpha=0.5)
# Legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=comm_colors[k % len(comm_colors)],
                 label=f"C{k}" + (" (seizure)" if k == 0 else ""))
           for k in range(N_COMM)]
ax.legend(handles=handles, fontsize=7, ncol=N_COMM, loc="upper center")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/community_map.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/community_map.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("COMMUNITY KURAMOTO-ReLU KANDy SUMMARY")
print(f"{'='*60}")
print(f"  Communities: {N_COMM} (from {N_CH} channels)")
print(f"  KAN: [{N_FEAT}, {N_COMM}]")
print(f"  Features: {N_FEAT} ({N_PAIRS} sin pairs + 2 gating + {N_COMM-1} gated coupling)")
print(f"  Samples: {T_TOTAL}")
print(f"  One-step MSE: {mse_onestep:.6e}")
print(f"  Per-community R²: {[f'{r:.4f}' for r in r2_per_comm]}")
print(f"  Mean R²: {np.mean(r2_per_comm):.4f}")
print(f"  Rollout RMSE: {rmse_roll:.4f}")
print(f"  Order param RMSE: {rmse_r:.4f}")
print(f"  ReLU threshold: {RELU_THRESHOLD:.4f}")
print()
print("  Discovered equations:")
for k in range(N_COMM):
    print(f"    dc{k}/dt = {formulas[k]}")
print(f"\n  Figures saved to: {OUT_DIR}/")
print(f"{'='*60}")

np.savez(
    f"{OUT_DIR}/kuramoto_community_results.npz",
    communities=communities,
    labels=labels,
    PLV=PLV,
    r2_per_comm=r2_per_comm,
    formulas=[str(f) for f in formulas],
    relu_threshold=RELU_THRESHOLD,
)
print(f"[SAVE] Results saved to {OUT_DIR}/kuramoto_community_results.npz")
