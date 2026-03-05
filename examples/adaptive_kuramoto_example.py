#!/usr/bin/env python3
"""KANDy example: Adaptive Kuramoto–Sakaguchi oscillators.

N coupled phase oscillators with adaptive coupling weights:
    dθ_i/dt  = ω_i + Σ_j  κ_ij * sin(θ_j - θ_i + α)
    dκ_ij/dt = -ε * [κ_ij + sin(θ_j - θ_i + β)]

Parameters:
    N     = 5 oscillators
    ω_i   ~ Uniform(-0.5, 0.5)  (fixed natural frequencies)
    α, β  = -π/4, -π/2
    ε     = 0.1  (adaptation rate)

State vector: (θ_1,...,θ_N, κ_12,...,κ_N(N-1)) ∈ R^{N + N(N-1)}

Koopman lift: relative phases and off-diagonal couplings
    phi(θ, κ) = [sin(θ_j - θ_i) for i≠j] ∪ [κ_ij for i≠j]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                 N*(N-1) features             N*(N-1) features
              = 2*N*(N-1) features total

The phase equations depend on sin of pairwise differences → the lift
directly encodes this nonlinearity.  The coupling equations are linear in κ
(plus a sin term already included in the first block).

KAN:  width = [2*N*(N-1), N]   (predicts N phase velocities)
      base_fun = torch.sin      (matches the sinusoidal coupling structure)

A second KAN (width = [2*N*(N-1), N*(N-1)]) learns the κ dynamics.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift, angle_mse
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED  = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N      = 5             # number of oscillators
ALPHA  = -np.pi / 4   # phase lag in phase equation
BETA   = -np.pi / 2   # phase lag in coupling equation
EPS    = 0.1          # adaptation rate

# Fixed natural frequencies
rng  = np.random.default_rng(SEED)
OMEGA = rng.uniform(-0.5, 0.5, size=N)

# Off-diagonal index pairs (i, j) where i ≠ j, ordered (0,1),(0,2),...
OD_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
N_OD     = len(OD_PAIRS)   # N*(N-1)
N_FEAT   = 2 * N_OD        # total lift dimension

print(f"[PARAMS] N={N}, N_OD={N_OD}, N_FEAT={N_FEAT}")
print(f"         KAN_theta: [{N_FEAT}, {N}]")
print(f"         KAN_kappa: [{N_FEAT}, {N_OD}]")

# ---------------------------------------------------------------------------
# 1. Simulation
# ---------------------------------------------------------------------------

def adaptive_kuramoto_rhs(t, y):
    """Full RHS of the adaptive Kuramoto-Sakaguchi system."""
    theta = y[:N]           # (N,)
    kappa = y[N:].reshape(N, N)   # (N, N) — κ_ij; diagonal is ignored

    dtheta = OMEGA.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                dtheta[i] += kappa[i, j] * np.sin(theta[j] - theta[i] + ALPHA)

    dkappa = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dkappa[i, j] = -EPS * (kappa[i, j] + np.sin(theta[j] - theta[i] + BETA))

    return np.concatenate([dtheta, dkappa.ravel()])


def simulate(T_end: float = 200.0, dt: float = 0.05, seed: int = SEED):
    """Simulate adaptive Kuramoto; return (t, theta_traj, kappa_traj)."""
    rng0  = np.random.default_rng(seed)
    theta0 = rng0.uniform(0, 2 * np.pi, size=N)
    kappa0 = rng0.uniform(-0.5, 0.5, size=(N, N))
    np.fill_diagonal(kappa0, 0.0)

    y0     = np.concatenate([theta0, kappa0.ravel()])
    t_eval = np.arange(0, T_end, dt)
    sol    = solve_ivp(adaptive_kuramoto_rhs, [0, T_end], y0,
                       t_eval=t_eval, method="RK45",
                       rtol=1e-8, atol=1e-10, dense_output=False)

    theta_traj = sol.y[:N, :].T              # (T, N)
    kappa_traj = sol.y[N:, :].T.reshape(-1, N, N)   # (T, N, N)
    return sol.t, theta_traj, kappa_traj


print("[DATA]  Simulating adaptive Kuramoto ...")
t_sim, THETA, KAPPA = simulate(T_end=300.0, dt=0.05)
T_snap = len(t_sim)
DT_SIM = t_sim[1] - t_sim[0]
print(f"[DATA]  T={T_snap} snapshots, dt={DT_SIM:.3f}")

# ---------------------------------------------------------------------------
# 2. Koopman lift  phi(theta, kappa) -> R^{2*N_OD}
# ---------------------------------------------------------------------------
THETA_FEAT_NAMES = [f"sin(θ{j}-θ{i})" for (i, j) in OD_PAIRS]
KAPPA_FEAT_NAMES = [f"κ{i}{j}"         for (i, j) in OD_PAIRS]
FEATURE_NAMES    = THETA_FEAT_NAMES + KAPPA_FEAT_NAMES


def build_lift(theta: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Lift (T, N) + (T, N, N) → (T, 2*N_OD).

    theta : (T, N) phase angles
    kappa : (T, N, N) coupling matrix (diagonal unused)
    """
    T = theta.shape[0]
    sin_feats  = np.column_stack([
        np.sin(theta[:, j] - theta[:, i]) for (i, j) in OD_PAIRS
    ])                                          # (T, N_OD)
    kappa_feats = np.column_stack([
        kappa[:, i, j] for (i, j) in OD_PAIRS
    ])                                          # (T, N_OD)
    return np.hstack([sin_feats, kappa_feats])  # (T, 2*N_OD)


Theta = build_lift(THETA, KAPPA)       # (T, 2*N_OD)

# Targets: phase velocities and κ velocities via central differences
Theta_inner = Theta[1:-1]
THETA_inner = THETA[1:-1]
KAPPA_inner = KAPPA[1:-1]

# Time derivatives via central differences
THETA_dot = (THETA[2:] - THETA[:-2]) / (2.0 * DT_SIM)   # (T-2, N)
KAPPA_dot = (KAPPA[2:] - KAPPA[:-2]) / (2.0 * DT_SIM)   # (T-2, N, N)

# Extract off-diagonal κ_dot
KAPPA_dot_od = np.column_stack([
    KAPPA_dot[:, i, j] for (i, j) in OD_PAIRS
])   # (T-2, N_OD)

print(f"[DATA]  Training samples: {len(Theta_inner)}")

# ---------------------------------------------------------------------------
# 3. Train KANDy for phase dynamics  dθ/dt = f(phi)
# ---------------------------------------------------------------------------
print("\n--- Model A: phase dynamics dθ/dt (KAN=[{}, {}]) ---".format(N_FEAT, N))

theta_lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kuramoto_lift")

model_theta = KANDy(
    lift=theta_lift,
    grid=5,
    k=3,
    steps=500,
    seed=SEED,
    base_fun=torch.sin,   # sinusoidal base matches coupling structure
)

# Phase dynamics: angle_mse as the rollout loss so that
# a predicted phase of 2π − ε is not penalised as a large error.
# Derivative supervision uses standard MSE (default loss_fn).
model_theta.fit(
    X=Theta_inner,
    X_dot=THETA_dot,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    rollout_loss_fn=angle_mse,
)

# ---------------------------------------------------------------------------
# 4. Train KANDy for coupling dynamics  dκ/dt = g(phi)
# ---------------------------------------------------------------------------
print("\n--- Model B: coupling dynamics dκ/dt (KAN=[{}, {}]) ---".format(N_FEAT, N_OD))

kappa_lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kappa_lift")

model_kappa = KANDy(
    lift=kappa_lift,
    grid=5,
    k=3,
    steps=500,
    seed=SEED,
    base_fun=torch.sin,
)

model_kappa.fit(
    X=Theta_inner,
    X_dot=KAPPA_dot_od,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
for name, mdl, out_names in [
    ("Phase (dθ/dt)", model_theta, [f"dθ{i}/dt" for i in range(N)]),
    ("Coupling (dκ/dt)", model_kappa, [f"dκ{i}{j}/dt" for i,j in OD_PAIRS[:4]] + ["..."]),
]:
    print(f"\n[SYMBOLIC] {name}:")
    try:
        formulas = mdl.get_formula(var_names=FEATURE_NAMES, round_places=2)
        for lab, f in zip(out_names[:min(3, len(formulas))], formulas[:3]):
            print(f"  {lab} = {f}")
        if len(formulas) > 3:
            print(f"  ... ({len(formulas)} total)")
    except Exception as exc:
        print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 6. Coupled rollout: integrate both learned models together
# ---------------------------------------------------------------------------

def rollout_coupled(theta0: np.ndarray, kappa0_od: np.ndarray,
                    T_steps: int, dt: float) -> tuple:
    """RK4 rollout using both learned models.

    theta0     : (N,) initial phases
    kappa0_od  : (N_OD,) initial off-diagonal couplings (flattened)
    Returns: (theta_traj (T, N), kappa_od_traj (T, N_OD))
    """
    theta    = theta0.copy().astype(np.float64)
    kappa_od = kappa0_od.copy().astype(np.float64)
    theta_hist    = [theta.copy()]
    kappa_od_hist = [kappa_od.copy()]

    def get_phi(th, kp):
        """Build the 2*N_OD feature vector from current state."""
        sin_f = np.array([np.sin(th[j] - th[i]) for (i, j) in OD_PAIRS])
        return np.concatenate([sin_f, kp])[None, :]   # (1, 2*N_OD)

    def f(th, kp):
        phi = get_phi(th, kp)
        dth = model_theta.predict(phi).ravel()
        dkp = model_kappa.predict(phi).ravel()
        return dth, dkp

    for _ in range(T_steps - 1):
        # RK4
        dt1, dk1 = f(theta,                   kappa_od)
        dt2, dk2 = f(theta + 0.5*dt*dt1,      kappa_od + 0.5*dt*dk1)
        dt3, dk3 = f(theta + 0.5*dt*dt2,      kappa_od + 0.5*dt*dk2)
        dt4, dk4 = f(theta + dt*dt3,           kappa_od + dt*dk3)

        theta    = theta    + (dt / 6.0) * (dt1 + 2*dt2 + 2*dt3 + dt4)
        kappa_od = kappa_od + (dt / 6.0) * (dk1 + 2*dk2 + 2*dk3 + dk4)
        theta_hist.append(theta.copy())
        kappa_od_hist.append(kappa_od.copy())

    return np.array(theta_hist), np.array(kappa_od_hist)


T_INNER = len(Theta_inner)
n_test  = int(T_INNER * 0.15)
t0_idx  = T_INNER - n_test

theta0_test    = THETA_inner[t0_idx]
kappa_od_test  = np.array([KAPPA_inner[t0_idx, i, j] for (i, j) in OD_PAIRS])

print(f"\n[ROLLOUT] Rolling out {n_test} steps ...")
pred_theta, pred_kappa_od = rollout_coupled(
    theta0_test, kappa_od_test, n_test, DT_SIM
)
true_theta    = THETA_inner[t0_idx:t0_idx + n_test]
true_kappa_od = np.column_stack([
    KAPPA_inner[t0_idx:t0_idx + n_test, i, j] for (i, j) in OD_PAIRS
])

rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
rmse_kappa = np.sqrt(np.mean((pred_kappa_od - true_kappa_od) ** 2))
print(f"[EVAL]  Rollout RMSE θ: {rmse_theta:.6f}")
print(f"[EVAL]  Rollout RMSE κ: {rmse_kappa:.6f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Kuramoto", exist_ok=True)

t_roll = np.arange(n_test) * DT_SIM

# 7a. Phase trajectories
fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
colors = plt.cm.tab10.colors
for i, ax in enumerate(axes):
    ax.plot(t_roll, true_theta[:, i],  color=colors[i], lw=1.2, label="True")
    ax.plot(t_roll, pred_theta[:, i],  color=colors[i], lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(f"$\\theta_{i}$")
    if i == 0:
        ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: phase trajectories", fontsize=11)
fig.tight_layout()
fig.savefig("results/Kuramoto/phase_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Kuramoto/phase_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Coupling weight trajectories (first 4 pairs)
n_show = min(4, N_OD)
fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show), sharex=True)
if n_show == 1:
    axes = [axes]
for k_idx, ax in enumerate(axes):
    i, j = OD_PAIRS[k_idx]
    ax.plot(t_roll, true_kappa_od[:, k_idx],  color=colors[k_idx % 10], lw=1.2)
    ax.plot(t_roll, pred_kappa_od[:, k_idx],  color=colors[k_idx % 10], lw=1.0, ls="--")
    ax.set_ylabel(f"$\\kappa_{{{i}{j}}}$")
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: coupling weights", fontsize=11)
fig.tight_layout()
fig.savefig("results/Kuramoto/kappa_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Kuramoto/kappa_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. Loss curves (theta model)
if hasattr(model_theta, "train_results_") and model_theta.train_results_:
    fig, ax = plot_loss_curves(
        model_theta.train_results_,
        title="Kuramoto phase model loss",
        save="results/Kuramoto/loss_phase",
    )
    plt.close(fig)

print("[FIGS]  Saved results/Kuramoto/")
