#!/usr/bin/env python3
"""KANDy example: Lorenz system.

The Lorenz system is a 3D ODE:
    x_dot = sigma*(y - x)
    y_dot = x*(rho - z) - y
    z_dot = x*y - beta*z

with sigma=10, rho=28, beta=8/3.

Because the RHS contains bilinear cross-terms x*y and x*z, the Koopman lift
must include these explicitly.  The single-layer KAN (width=[6, 3]) then only
needs to learn separable univariate functions of each lifted feature.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import (
    plot_all_edges,
    plot_attractor_overlay,
    plot_trajectory_error,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# 1. Data generation — Lorenz ODE (RK4)
# ---------------------------------------------------------------------------
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0


def lorenz_rhs(state):
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


DT      = 0.005
T_MAX   = 50.0
BURN_IN = 2.0

n_steps = int(np.round(T_MAX / DT))
t_fine  = np.linspace(0.0, T_MAX, n_steps + 1)

traj = np.zeros((n_steps + 1, 3))
traj[0] = [1.0, 1.0, 1.0]
for i in range(n_steps):
    traj[i + 1] = rk4_step(lorenz_rhs, traj[i], DT)

# Discard transient
burn_idx = int(np.floor(BURN_IN / DT))
t_data = t_fine[burn_idx:]
X      = traj[burn_idx:]          # (N, 3)

# Forward-difference derivatives (same as research code)
X_dot = (X[1:] - X[:-1]) / DT    # (N-1, 3)
X     = X[:-1]
t_data = t_data[:-1]

print(f"[DATA]  N={len(X)} snapshots, dt={DT}")

# ---------------------------------------------------------------------------
# 2. Koopman lift  phi: R^3 -> R^6
#    theta = [x, y, z, x*y, x*z, y*z]
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["x", "y", "z", "xy", "xz", "yz"]


def lorenz_lift(X: np.ndarray) -> np.ndarray:
    """Lift R^3 -> R^6 including all pairwise products."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([x, y, z, x * y, x * z, y * z])


def lorenz_lift_torch(X: torch.Tensor) -> torch.Tensor:
    """Torch version of lift (differentiable, for rollout loss)."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return torch.stack([x, y, z, x * y, x * z, y * z], dim=1)


lift = CustomLift(fn=lorenz_lift, output_dim=6, torch_fn=lorenz_lift_torch, name="lorenz_lift")

# ---------------------------------------------------------------------------
# 3. KANDy model  (single-layer KAN: width=[6, 3])
#    Matches research code: grid=2, k=1, RBF base, rollout loss
# ---------------------------------------------------------------------------
rbf = lambda x: torch.exp(-(3 * x**2))

model = KANDy(
    lift=lift,
    grid=2,
    k=1,
    steps=300,
    seed=SEED,
    base_fun=rbf,
)

model.fit(
    X=X,
    X_dot=X_dot,
    dt=DT,
    val_frac=0.0,
    test_frac=0.2,
    lamb=0.0,
    opt="LBFGS",
    lr=1e-3,
    rollout_weight=1.0,
    rollout_horizon=10,
    stop_grid_update_step=300,
    patience=0,
)

# ---------------------------------------------------------------------------
# 4. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas ...")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=2)
    labels = ["x_dot", "y_dot", "z_dot"]
    for lab, f in zip(labels, formulas):
        print(f"  {lab} = {f}")
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 5. Rollout validation
# ---------------------------------------------------------------------------
# Use the final 20 % of data as the test window
N        = len(X)
n_test   = int(N * 0.20)
x0_test  = X[N - n_test]
T_test   = n_test
true_traj = X[N - n_test:]

pred_traj = model.rollout(x0_test, T=T_test, dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={T_test} steps): {rmse:.6f}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Lorenz", exist_ok=True)

# 6a. Attractor overlay (x-z projection)
fig, ax = plot_attractor_overlay(
    true_traj, pred_traj,
    dim_x=0, dim_y=2,
    labels=["True Lorenz", "KANDy"],
    colors=["#1f77b4", "#d62728"],
    save="results/Lorenz/attractor",
)
plt.close(fig)

# 6b. Trajectory error
t_test = np.arange(T_test) * DT
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_test,
    save="results/Lorenz/trajectory_error",
)
plt.close(fig)

# 6c. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(
        model.train_results_,
        save="results/Lorenz/loss_curves",
    )
    plt.close(fig)

# 6d. All edge activations
train_theta = torch.tensor(
    lorenz_lift(X[: int(N * 0.70)]), dtype=torch.float32
)
fig, axes = plot_all_edges(
    model.model_,
    X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["x_dot", "y_dot", "z_dot"],
    save="results/Lorenz/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/Lorenz/")
