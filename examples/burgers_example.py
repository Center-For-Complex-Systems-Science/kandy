#!/usr/bin/env python3
"""KANDy example: Inviscid Burgers equation.

The inviscid Burgers PDE:
    u_t + u * u_x = 0      on x ∈ [0, 2π], periodic

can be written as  u_t = -u * u_x = -∂_x(u²/2).

Koopman lift:  phi(u) = [u,  u_x,  (u²/2)_x]
                         ^^^  ^^^   ^^^^^^^^^^
                         1D   deriv   nonlin deriv
    -> 3 features, KAN = [3, 1] per spatial point

Data generation: method-of-characteristics (exact) with simple periodic ICs,
followed by Rusanov (local-Lax-Friedrichs) flux for the shock-forming regime.

The activation on the (u²/2)_x edge should recover the identity function
(coefficient ≈ -1) since  u_t = -∂_x(u²/2).

We also show the sech² curve fit for the shock edge activation.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift, solve_burgers
from kandy.numerics import muscl_reconstruct, burgers_flux, burgers_speed
from kandy.plotting import (
    get_edge_activation,
    plot_edge,
    plot_all_edges,
    plot_loss_curves,
    fit_sech2,
    fit_linear,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

L    = 2.0 * np.pi
N_X  = 128           # spatial grid points
DT   = 0.005         # time step
N_T  = 3_000         # total time steps
BURN = 300           # transient

x_grid = np.linspace(0, L, N_X, endpoint=False)
dx     = L / N_X

# ---------------------------------------------------------------------------
# 1. Data generation via kandy.numerics
# ---------------------------------------------------------------------------
# Initial condition: smooth sinusoidal
u0 = np.sin(x_grid) + 0.5 * np.sin(2 * x_grid)

print("[DATA]  Simulating inviscid Burgers (Rusanov / TVD-RK2) ...")
U_full = solve_burgers(u0, n_steps=N_T, dt=DT, scheme="rusanov",
                       limiter="minmod", time_stepper="tvdrk2")
U = U_full[BURN:]                     # discard transient
T_snap = U.shape[0]
print(f"[DATA]  T={T_snap} snapshots, N_x={N_X}")

# ---------------------------------------------------------------------------
# 2. Feature library and derivative computation
# ---------------------------------------------------------------------------

def tvd_derivative(u: np.ndarray) -> np.ndarray:
    """First derivative via minmod-limited MUSCL slopes."""
    # muscl_reconstruct returns the slope * dx/2; recover slope / dx
    u_L, _ = muscl_reconstruct(u, dx, limiter="minmod")
    return (u_L - u) * 2.0 / dx   # = slope (units: [u]/[x])


FEATURE_NAMES = ["u", "u_x", "d(u²/2)/dx"]
N_FEATURES    = 3


def build_burgers_library(U_batch: np.ndarray) -> np.ndarray:
    """Build (T*N_x, 3) feature matrix from snapshot array (T, N_x)."""
    rows = []
    for t in range(U_batch.shape[0]):
        u    = U_batch[t]                    # (N_x,)
        u_x  = tvd_derivative(u)
        f_x  = tvd_derivative(0.5 * u ** 2)  # ∂_x(u²/2)
        rows.append(np.column_stack([u, u_x, f_x]))
    return np.vstack(rows)                   # (T*N_x, 3)


# Time derivative via central differences
U_inner = U[1:-1]                                      # (T-2, N_x)
U_dot   = (U[2:] - U[:-2]) / (2.0 * DT)               # (T-2, N_x)

print("[DATA]  Building feature library ...")
Theta   = build_burgers_library(U_inner)               # (T_inner * N_x, 3)
U_t_flat = U_dot.ravel()[:, None]                      # (T_inner * N_x, 1)

print(f"[DATA]  Theta shape: {Theta.shape}")

# ---------------------------------------------------------------------------
# 3. KANDy model  (KAN=[3, 1])
# ---------------------------------------------------------------------------
burgers_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="burgers_lift")

model = KANDy(
    lift=burgers_lift,
    grid=7,
    k=3,
    steps=500,
    seed=SEED,
)

model.fit(
    X=Theta,
    X_dot=U_t_flat,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 4. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=2)
    print(f"  u_t = {formulas[0]}")
    # Expect something close to: -d(u²/2)/dx  (coefficient ≈ -1)
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
N_total = len(Theta)
n_test  = int(N_total * 0.15)
Th_test = Theta[N_total - n_test:]
Ut_test = U_t_flat[N_total - n_test:]

Ut_pred = model.predict(Th_test)
mse     = np.mean((Ut_pred - Ut_test) ** 2)
print(f"\n[EVAL]  Test MSE: {mse:.6e}   RMSE: {mse**0.5:.6e}")

# ---------------------------------------------------------------------------
# 6. Time rollout using learned model
# ---------------------------------------------------------------------------

def rollout_burgers(model_fn, u0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Euler rollout using the KANDy model (Euler for simplicity here)."""
    u    = u0.copy()
    traj = [u.copy()]
    for _ in range(n_steps - 1):
        theta = build_burgers_library(u[None, :])  # (N_x, 3)
        u_t   = model.predict(theta).ravel()        # (N_x,)
        u     = u + dt * u_t
        traj.append(u.copy())
    return np.array(traj)


T_INNER = U_inner.shape[0]
n_roll  = min(400, int(T_INNER * 0.15))
t0_idx  = T_INNER - n_roll
u0_roll = U_inner[t0_idx]

true_roll = U_inner[t0_idx: t0_idx + n_roll]
pred_roll = rollout_burgers(model, u0_roll, n_roll, DT)

rmse_roll = np.sqrt(np.mean((pred_roll - true_roll) ** 2))
print(f"[EVAL]  Rollout RMSE (T={n_roll} steps): {rmse_roll:.6f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Burgers", exist_ok=True)

# 7a. Space-time comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t_arr = np.arange(n_roll) * DT
for ax, data, title in zip(axes,
                             [true_roll, pred_roll],
                             ["True Burgers", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[0, t_arr[-1], 0, L],
                   cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xlabel("time"); ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.suptitle("Inviscid Burgers", fontsize=12)
fig.tight_layout()
fig.savefig("results/Burgers/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Burgers/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Edge activations with sech² fit on u-edge and linear fit on d(u²/2)/dx
n_sub     = min(5000, int(N_total * 0.70))
sub_idx   = np.random.choice(int(N_total * 0.70), n_sub, replace=False)
train_t   = torch.tensor(Theta[sub_idx], dtype=torch.float32)

# Edge (layer 0, input 2 [d(u²/2)/dx], output 0 [u_t]) — should be linear
x_edge, y_edge = get_edge_activation(model.model_, l=0, i=2, j=0, X=train_t)
lin_fit = fit_linear(x_edge, y_edge)
sech_fit = fit_sech2(x_edge, y_edge)
fig, ax = plot_edge(
    x_edge, y_edge,
    fits=[lin_fit, sech_fit],
    title="Burgers: edge (d(u²/2)/dx → u_t)",
    xlabel="d(u²/2)/dx",
    ylabel="u_t contribution",
    save="results/Burgers/edge_flux",
)
plt.close(fig)

# 7c. All edges
fig = plot_all_edges(
    model.model_,
    X=train_t,
    input_names=FEATURE_NAMES,
    output_names=["u_t"],
    title="Burgers KAN edge activations",
    save="results/Burgers/edge_activations",
)
plt.close(fig)

# 7d. Loss curves
if hasattr(model, "train_results_") and model.train_results_:
    fig, ax = plot_loss_curves(
        model.train_results_,
        title="Burgers training loss",
        save="results/Burgers/loss_curves",
    )
    plt.close(fig)

print("[FIGS]  Saved results/Burgers/")
