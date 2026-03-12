#!/usr/bin/env python3
"""KANDy example: Standard Kuramoto oscillators.

N coupled phase oscillators with fixed global coupling:
    dθ_i/dt = ω_i + (K/N) Σ_{j≠i} sin(θ_j - θ_i)

This is the simplest Kuramoto model — no adaptive coupling, no phase lag.
It serves as a diagnostic stepping stone for the adaptive variant: if symbolic
extraction succeeds here, the failure on the adaptive model is likely caused
by the additional κ_ij features, not the sin/cos phase-difference lift.

Parameters:
    N   = 5 oscillators
    K   = 1.0  (near critical coupling — interesting partially-synchronized dynamics)
    ω_i ~ Uniform(-1, 1)  (wider spread to avoid trivial synchronisation)

Koopman lift (using θ_i - θ_j convention, sin-only — no cos to avoid degeneracy):
    phi(θ) = [sin(θ_i - θ_j) for i≠j]   N*(N-1) = 20 features

KAN: width = [N*(N-1), N]  →  [20, 5]
     base_fun = torch.sin

Key improvements over v1:
  1. K=1.0 with wider ω spread → partial synchronisation (r < 1)
  2. Pruning before symbolic extraction (prune_edge removes weak connections)
  3. Linear-only symbolic library: {x, 0} — lift already encodes sin/cos,
     so KAN activations should be linear (identity). Using sin in the library
     causes nested sin(sin(...)) compositions that break the rollout.
"""

import os
import numpy as np
import torch
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N = 5            # number of oscillators
K = 1.0          # near-critical coupling (Kc ≈ 2/π ≈ 0.64 for uniform dist)

# Wider frequency spread for more interesting dynamics
rng = np.random.default_rng(SEED)
OMEGA = rng.uniform(-1.0, 1.0, size=N)

# Off-diagonal index pairs
OD_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
N_OD = len(OD_PAIRS)       # N*(N-1) = 20
N_FEAT = N_OD              # sin only = 20 (no cos — same KS lesson)

print(f"[PARAMS] N={N}, K={K}, N_OD={N_OD}, N_FEAT={N_FEAT}")
print(f"         KAN: [{N_FEAT}, {N}]")
print(f"         omega = {OMEGA.round(4)}")
print(f"         K/N = {K/N:.4f}")

# ---------------------------------------------------------------------------
# 1. Simulation — multiple ICs concatenated for diverse training data
# ---------------------------------------------------------------------------

def kuramoto_rhs(t, theta):
    """RHS of standard Kuramoto model."""
    dtheta = OMEGA.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                dtheta[i] += (K / N) * np.sin(theta[j] - theta[i])
    return dtheta


def simulate(T_end=100.0, dt=0.05, seed=SEED):
    """Simulate standard Kuramoto; return (t, theta_traj)."""
    rng0 = np.random.default_rng(seed)
    theta0 = rng0.uniform(0, 2 * np.pi, size=N)
    t_eval = np.arange(0, T_end, dt)
    sol = solve_ivp(kuramoto_rhs, [0, T_end], theta0,
                    t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T   # (T, N)


# Generate multiple trajectories with different ICs for diversity
print("[DATA]  Simulating Kuramoto (3 ICs) ...")
all_theta = []
for ic_seed in [SEED, SEED + 1, SEED + 2]:
    t_sim, theta_traj = simulate(T_end=100.0, dt=0.05, seed=ic_seed)
    all_theta.append(theta_traj)

THETA = np.vstack(all_theta)  # (3*T, N)
T_snap = len(THETA)
DT_SIM = 0.05
print(f"[DATA]  T={T_snap} total snapshots, dt={DT_SIM:.3f}")

# Check order parameter spread
r_all = np.abs(np.mean(np.exp(1j * THETA), axis=1))
print(f"[DATA]  Order parameter: min={r_all.min():.3f}, max={r_all.max():.3f}, "
      f"mean={r_all.mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Koopman lift: sin(θ_i - θ_j) only  (no cos — avoids library degeneracy)
# ---------------------------------------------------------------------------
SIN_NAMES = [f"sin(t{i}-t{j})" for (i, j) in OD_PAIRS]
FEATURE_NAMES = SIN_NAMES


def build_lift(theta):
    """Lift (T, N) → (T, N_OD).  Sin-only: cos excluded to avoid degeneracy."""
    sin_feats = np.column_stack(
        [np.sin(theta[:, i] - theta[:, j]) for (i, j) in OD_PAIRS]
    )
    return sin_feats


Phi = build_lift(THETA)   # (T, N_FEAT)

# Targets: phase velocities via central differences
# Process each trajectory separately to avoid cross-boundary artifacts
Phi_parts = []
THETA_dot_parts = []
THETA_inner_parts = []
traj_len = T_snap // 3
for k in range(3):
    s, e = k * traj_len, (k + 1) * traj_len
    Phi_parts.append(Phi[s+1:e-1])
    THETA_dot_parts.append(
        (THETA[s+2:e] - THETA[s:e-2]) / (2.0 * DT_SIM)
    )
    THETA_inner_parts.append(THETA[s+1:e-1])

Phi_inner = np.vstack(Phi_parts)
THETA_dot = np.vstack(THETA_dot_parts)
THETA_inner = np.vstack(THETA_inner_parts)

T_INNER = len(Phi_inner)
print(f"[DATA]  Training samples: {T_INNER}")

# ---------------------------------------------------------------------------
# 3. KANDy model: width=[40, 5]
# ---------------------------------------------------------------------------
lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kuramoto_identity")

model = KANDy(
    lift=lift,
    grid=5,
    k=3,
    steps=200,
    seed=SEED,
    base_fun=torch.sin,
    device="cpu",
)

print("\n[TRAIN] Fitting KAN ...")
model.fit(
    X=Phi_inner,
    X_dot=THETA_dot,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 4. Pruning + Symbolic extraction with restricted library
# ---------------------------------------------------------------------------
import sys
print("\n[PRUNE] Pruning weak edges ...", flush=True)
sym_subset = torch.tensor(Phi_inner[:2048], dtype=torch.float32)
model.model_.save_act = True
print("[PRUNE] Running forward pass for activations ...", flush=True)
with torch.no_grad():
    model.model_(sym_subset)

# Compute edge attribution scores, then prune weak edges
print("[PRUNE] Computing edge attributions ...", flush=True)
model.model_.attribute()
print("[PRUNE] Pruning edges with threshold=5e-2 ...", flush=True)
model.model_.prune_edge(threshold=5e-2)

# Count surviving edges
mask = model.model_.act_fun[0].mask.data
n_alive = (mask.abs() > 0).sum().item()
n_total_edges = mask.numel()
print(f"[PRUNE] {n_alive}/{n_total_edges} edges survive (pruned {n_total_edges - n_alive})")

# Re-run forward pass after pruning so activations reflect pruned state
with torch.no_grad():
    model.model_(sym_subset)

# Linear-only library: lift already encodes sin/cos, so activations must be linear
LINEAR_LIB = make_symbolic_lib({
    "x":   (lambda x: x,           lambda x: x,           1),
    "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
})

print("\n[SYMBOLIC] Extracting with linear-only library {x, 0} ...", flush=True)
print("  (lift already contains sin/cos — activations should be linear)", flush=True)
model.model_.auto_symbolic(lib=LINEAR_LIB, weight_simple=0.8)
print("[SYMBOLIC] Done.", flush=True)

# Get formulas
exprs, vars_ = model.model_.symbolic_formula()
sub_map = {sp.Symbol(str(v)): sp.Symbol(n) for v, n in zip(vars_, FEATURE_NAMES)}
formulas = []
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym).xreplace(
        {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
    )
    formulas.append(sym)

print(f"\nTrue: dth_i/dt = w_i + (K/N) sum_j sin(th_j - th_i)", flush=True)
print(f"      = w_i - (K/N) sum_j sin(th_i - th_j)", flush=True)
print(f"      K/N = {K/N:.4f}\n", flush=True)
for i in range(N):
    print(f"  dth{i}/dt = {formulas[i]}", flush=True)
    true_terms = [f"{OMEGA[i]:.4f}"]
    for j in range(N):
        if j != i:
            true_terms.append(f"- {K/N:.2f}*sin(th{i}-th{j})")
    print(f"    [TRUE] = {' '.join(true_terms)}", flush=True)
print(flush=True)

# ---------------------------------------------------------------------------
# 5. Rollout
# ---------------------------------------------------------------------------
def build_lift_single(theta):
    """Lift a single (N,) state → (1, 2*N_OD)."""
    sin_f = np.array([np.sin(theta[i] - theta[j]) for (i, j) in OD_PAIRS])
    return sin_f[None, :]


def rollout(theta0, n_steps, dt):
    """RK4 rollout using the learned model."""
    theta = theta0.copy().astype(np.float64)
    traj = [theta.copy()]

    for _ in range(n_steps - 1):
        def f(th):
            phi = build_lift_single(th)
            return model.predict(phi).ravel()

        k1 = f(theta)
        k2 = f(theta + 0.5 * dt * k1)
        k3 = f(theta + 0.5 * dt * k2)
        k4 = f(theta + dt * k3)
        theta = theta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(theta.copy())

    return np.array(traj)


# Use first IC trajectory's test region for rollout
N_ROLLOUT = 500
n_test_start = int(T_INNER * 0.85)
theta0_test = THETA_inner[n_test_start]

print(f"[ROLLOUT] Rolling out {N_ROLLOUT} steps ...", flush=True)
pred = rollout(theta0_test, N_ROLLOUT, DT_SIM)
true = THETA_inner[n_test_start:n_test_start + N_ROLLOUT]

rmse_theta = np.sqrt(np.mean((pred - true) ** 2))
print(f"[EVAL]  Rollout RMSE th: {rmse_theta:.6f}", flush=True)

# Order parameter
r_true = np.abs(np.mean(np.exp(1j * true), axis=1))
r_pred = np.abs(np.mean(np.exp(1j * pred), axis=1))
rmse_r = np.sqrt(np.mean((r_pred - r_true) ** 2))
print(f"[EVAL]  Order parameter RMSE r: {rmse_r:.6f}", flush=True)

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Kuramoto", exist_ok=True)

t_roll = np.arange(N_ROLLOUT) * DT_SIM

# 6a. Phase trajectories
fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
colors = plt.cm.tab10.colors
for i, ax in enumerate(axes):
    ax.plot(t_roll, true[:, i], color=colors[i], lw=1.2, label="True")
    ax.plot(t_roll, pred[:, i], color=colors[i], lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(f"$\\theta_{i}$")
    if i == 0:
        ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("time")
fig.suptitle("Kuramoto: phase trajectories", fontsize=11)
fig.tight_layout()
fig.savefig("results/Kuramoto/phase_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Kuramoto/phase_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Order parameter
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_roll, r_true, color="steelblue", lw=1.4, label="True $r(t)$")
ax.plot(t_roll, r_pred, color="tomato", lw=1.0, ls="--", label="KANDy $r(t)$")
ax.set_xlabel("time")
ax.set_ylabel("$r(t)$")
ax.set_title(f"Kuramoto order parameter  (RMSE={rmse_r:.4f})")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("results/Kuramoto/order_parameter.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Kuramoto/order_parameter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(
        model.train_results_,
        save="results/Kuramoto/loss_curves",
    )
    plt.close(fig)

# 6d. Edge activations — only plot surviving edges after pruning
n_total = len(Phi_inner)
n_sub = min(5000, int(n_total * 0.70))
sub_idx = np.random.choice(int(n_total * 0.70), n_sub, replace=False)
train_phi_t = torch.tensor(Phi_inner[sub_idx], dtype=torch.float32)
try:
    fig, axes = plot_all_edges(
        model.model_,
        X=train_phi_t,
        in_var_names=FEATURE_NAMES,
        out_var_names=[f"dth{i}/dt" for i in range(N)],
        save="results/Kuramoto/edge_activations",
    )
    plt.close(fig)
except Exception as e:
    print(f"[WARN] Edge activation plot failed: {e}")

print("[FIGS]  Saved results/Kuramoto/")
