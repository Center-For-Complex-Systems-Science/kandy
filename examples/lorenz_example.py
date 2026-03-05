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
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_all_edge_activations,
    plot_all_edges,
    plot_attractor_overlay,
    plot_trajectory_error,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data generation — Lorenz ODE
# ---------------------------------------------------------------------------
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ]


DT    = 0.01
T_MAX = 30.0          # 30 seconds of simulation
BURN  = 5.0           # seconds of transient to discard

t_full = np.arange(0.0, T_MAX + BURN, DT)
sol    = solve_ivp(lorenz, [t_full[0], t_full[-1]], [1.0, 1.0, 1.0],
                   t_eval=t_full, method="RK45", rtol=1e-10, atol=1e-12)

# Discard transient
n_burn = int(BURN / DT)
X      = sol.y[:, n_burn:].T          # (N, 3)
t_data = t_full[n_burn:]

# Analytical derivatives (use the ODE directly — cleaner than finite diff)
X_dot = np.column_stack([
    SIGMA * (X[:, 1] - X[:, 0]),
    X[:, 0] * (RHO - X[:, 2]) - X[:, 1],
    X[:, 0] * X[:, 1] - BETA * X[:, 2],
])

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


lift = CustomLift(fn=lorenz_lift, output_dim=6, name="lorenz_lift")

# ---------------------------------------------------------------------------
# 3. KANDy model  (single-layer KAN: width=[6, 3])
# ---------------------------------------------------------------------------
model = KANDy(
    lift=lift,
    grid=5,
    k=3,
    steps=500,
    seed=SEED,
)

model.fit(
    X=X,
    X_dot=X_dot,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
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

# 6a. Attractor overlay (x–z projection)
fig, ax = plot_attractor_overlay(
    true_traj, pred_traj,
    dim_x=0, dim_y=2,
    labels=["True Lorenz", "KANDy"],
    colors=["#1f77b4", "#d62728"],
    title="Lorenz Attractor",
    save="results/Lorenz/attractor",
)
plt.close(fig)

# 6b. Trajectory error
t_test = np.arange(T_test) * DT
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_test,
    title="Lorenz trajectory error",
    save="results/Lorenz/trajectory_error",
)
plt.close(fig)

# 6c. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(
        model.train_results_,
        title="Lorenz training loss",
        save="results/Lorenz/loss_curves",
    )
    plt.close(fig)

# 6d. All edge activations
train_theta = torch.tensor(
    lorenz_lift(X[: int(N * 0.70)]), dtype=torch.float32
)
activations = get_all_edge_activations(model.model_, X=train_theta)
fig = plot_all_edges(
    model.model_,
    X=train_theta,
    input_names=FEATURE_NAMES,
    output_names=["x_dot", "y_dot", "z_dot"],
    title="Lorenz KAN edge activations",
    save="results/Lorenz/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/Lorenz/")
