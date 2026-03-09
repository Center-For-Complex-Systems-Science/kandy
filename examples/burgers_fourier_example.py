#!/usr/bin/env python3
"""KANDy example: Inviscid Burgers — Fourier-mode initial conditions.

Same PDE as burgers_example.py:
    u_t + u * u_x = 0

but with initial conditions drawn from random Fourier modes:
    u_0(x) = Σ_{k=1}^{K_max}  a_k * cos(k*x) + b_k * sin(k*x)

with a_k, b_k ~ N(0, σ_k²) and σ_k ∝ k^{-3/2} (energy spectrum decay).

This creates a diverse training set with shock structures at different
positions and amplitudes, making the identification task harder.  The
KANDy model with sech²-shaped edge activations can recover the shock
solutions symbolically.

Feature library (same as burgers_example):
    phi(u) = [u, u_x, ∂_x(u²/2)]
    KAN = [3, 1]

We train on multiple trajectories simultaneously (stacked dataset).
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift, solve_burgers
from kandy.numerics import muscl_reconstruct
from kandy.plotting import (
    get_edge_activation,
    plot_edge,
    plot_all_edges,
    plot_loss_curves,
    fit_sech2,
    fit_sech2_tanh,
    fit_linear,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

L     = 2.0 * np.pi
N_X   = 128
DT    = 0.004
N_T   = 2_500    # steps per trajectory
BURN  = 200
N_IC  = 5        # number of random initial conditions
K_MAX = 5        # max Fourier mode

x_grid = np.linspace(0, L, N_X, endpoint=False)
dx     = L / N_X

# ---------------------------------------------------------------------------
# 1. Random Fourier-mode initial conditions
# ---------------------------------------------------------------------------

def random_fourier_ic(x: np.ndarray, k_max: int = K_MAX,
                      seed: int = 0) -> np.ndarray:
    """Smooth random IC from superposition of Fourier modes."""
    rng = np.random.default_rng(seed)
    u = np.zeros_like(x)
    for k in range(1, k_max + 1):
        sigma = k ** (-1.5)
        a_k = rng.normal(0, sigma)
        b_k = rng.normal(0, sigma)
        u += a_k * np.cos(k * x) + b_k * np.sin(k * x)
    return u.astype(np.float32)


def simulate_burgers_tvdrk2(u0: np.ndarray, n_t: int, dt: float) -> np.ndarray:
    """Wrapper around kandy.numerics.solve_burgers for backward compatibility."""
    return solve_burgers(u0, n_steps=n_t, dt=dt, scheme="rusanov",
                         limiter="minmod", time_stepper="tvdrk2").astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Generate multi-IC dataset
# ---------------------------------------------------------------------------

def tvd_deriv(u: np.ndarray) -> np.ndarray:
    """First derivative via minmod-limited MUSCL slopes."""
    u_L, _ = muscl_reconstruct(u, dx, limiter="minmod")
    return (u_L - u) * 2.0 / dx


def build_library(U_batch: np.ndarray) -> tuple:
    """Build Theta (T*N_x, 3) and U_t (T*N_x, 1) from (T, N_x) array."""
    T = U_batch.shape[0]
    # Central-diff time derivative (trim boundary)
    U_inner  = U_batch[1:-1]
    U_t      = (U_batch[2:] - U_batch[:-2]) / (2.0 * DT)   # (T-2, N_x)

    rows = []
    for t in range(U_inner.shape[0]):
        u   = U_inner[t]
        u_x = tvd_deriv(u)
        f_x = tvd_deriv(0.5 * u ** 2)
        rows.append(np.column_stack([u, u_x, f_x]))
    Theta   = np.vstack(rows)                   # (T_inner * N_x, 3)
    U_t_flat = U_t.ravel()[:, None]             # (T_inner * N_x, 1)
    return Theta, U_t_flat


print("[DATA]  Generating Burgers trajectories (Fourier ICs) ...")
Theta_list, Ut_list = [], []
for ic_idx in range(N_IC):
    u0  = random_fourier_ic(x_grid, k_max=K_MAX, seed=SEED + ic_idx)
    U   = simulate_burgers_tvdrk2(u0, N_T, DT)[BURN:]
    Th, Ut = build_library(U)
    Theta_list.append(Th)
    Ut_list.append(Ut)

Theta_full   = np.vstack(Theta_list).astype(np.float32)
U_t_full     = np.vstack(Ut_list).astype(np.float32)

# Subsample to avoid memory/numerical issues with PyKAN LBFGS + grid updates
MAX_SAMPLES = 100_000
if len(Theta_full) > MAX_SAMPLES:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Theta_full), MAX_SAMPLES, replace=False)
    Theta   = Theta_full[idx]
    U_t_all = U_t_full[idx]
    print(f"[DATA]  Subsampled {MAX_SAMPLES} from {len(Theta_full)} total (from {N_IC} trajectories)")
else:
    Theta   = Theta_full
    U_t_all = U_t_full
    print(f"[DATA]  Total samples: {len(Theta)} (from {N_IC} trajectories)")

# ---------------------------------------------------------------------------
# 4. KANDy model  (KAN = [3, 1])
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["u", "u_x", "d(u²/2)/dx"]

burgers_lift = CustomLift(fn=lambda X: X, output_dim=3, name="burgers_fourier_lift")

model = KANDy(
    lift=burgers_lift,
    grid=7,
    k=3,
    steps=500,
    seed=SEED,
)

model.fit(
    X=Theta,
    X_dot=U_t_all,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=2)
    print(f"  u_t = {formulas[0]}")
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 6. Evaluation — test on a fresh unseen IC
# ---------------------------------------------------------------------------
u0_test    = random_fourier_ic(x_grid, k_max=K_MAX, seed=SEED + N_IC + 1)
U_test     = simulate_burgers_tvdrk2(u0_test, N_T, DT)[BURN:]
Th_test, Ut_test = build_library(U_test)

Ut_pred = model.predict(Th_test)
mse     = np.mean((Ut_pred - Ut_test) ** 2)
print(f"\n[EVAL]  Held-out IC test MSE: {mse:.6e}   RMSE: {mse**0.5:.6e}")

# Rollout on held-out IC — SSP-RK3 with CFL substeps (same as baselines)
def kandy_rhs(u):
    """Compute u_t = KANDy(phi(u)) for spatial field u (1, N_x)."""
    u_1d = u.ravel()
    th = np.column_stack([u_1d, tvd_deriv(u_1d), tvd_deriv(0.5 * u_1d**2)])
    return model.predict(th).ravel().reshape(u.shape)


def ssp_rk3_step(u, h, rhs_fn):
    k1 = rhs_fn(u)
    u1 = u + h * k1
    k2 = rhs_fn(u1)
    u2 = 0.75 * u + 0.25 * (u1 + h * k2)
    k3 = rhs_fn(u2)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + h * k3)


def rollout_burgers(u0: np.ndarray, n_steps: int, dt: float,
                    cfl: float = 0.35) -> np.ndarray:
    u = u0[np.newaxis, :].copy()
    traj = [u0.copy()]
    for _ in range(n_steps - 1):
        umax = np.max(np.abs(u)) + 1e-12
        h_hyp = cfl * dx / umax
        n_sub = max(1, int(np.ceil(dt / h_hyp)))
        h = dt / n_sub
        for _ in range(n_sub):
            u = ssp_rk3_step(u, h, kandy_rhs)
            if np.any(np.isnan(u)):
                traj.append(np.full_like(u0, np.nan))
                return np.array(traj)
        traj.append(u.ravel().copy())
    return np.array(traj)


U_inner_test = U_test[1:-1]
N_ROLL = min(400, len(U_inner_test))
u0_roll   = U_inner_test[0]
true_roll = U_inner_test[:N_ROLL]
pred_roll = rollout_burgers(u0_roll, N_ROLL, DT)

rmse_roll = np.sqrt(np.mean((pred_roll - true_roll) ** 2))
print(f"[EVAL]  Rollout RMSE (T={N_ROLL}, held-out IC): {rmse_roll:.6f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
RESULTS = "results/Burgers-Fourier"
os.makedirs(RESULTS, exist_ok=True)

# 7a. Space-time heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t_arr = np.arange(N_ROLL) * DT
for ax, data, title in zip(axes,
                             [true_roll, pred_roll],
                             ["True Burgers (Fourier IC)", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[0, t_arr[-1], 0, L],
                   cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xlabel("time"); ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.suptitle("Burgers (Fourier ICs)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS}/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Edge activations — highlight the sech² / sech²·tanh shapes
n_sub   = min(8000, int(len(Theta) * 0.70))
sub_idx = np.random.choice(int(len(Theta) * 0.70), n_sub, replace=False)
train_t = torch.tensor(Theta[sub_idx], dtype=torch.float32)

# Edge: d(u²/2)/dx → u_t  (should be approximately linear, coeff ≈ -1)
x_e, y_e = get_edge_activation(model.model_, l=0, i=2, j=0, X=train_t)
fig, ax = plot_edge(
    x_e, y_e, fits=["linear", "sech2"],
    title="Burgers-Fourier: edge (d(u²/2)/dx → u_t)",
    xlabel="d(u²/2)/dx", ylabel="u_t contribution",
    save=f"{RESULTS}/edge_flux",
)
plt.close(fig)

# Edge: u_x → u_t
x_e2, y_e2 = get_edge_activation(model.model_, l=0, i=1, j=0, X=train_t)
fig, ax = plot_edge(
    x_e2, y_e2, fits=["linear", "sech2_tanh"],
    title="Burgers-Fourier: edge (u_x → u_t)",
    xlabel="u_x", ylabel="u_t contribution",
    save=f"{RESULTS}/edge_ux",
)
plt.close(fig)

# All edges grid
fig, axes = plot_all_edges(
    model.model_,
    X=train_t,
    fits=["linear"],
    in_var_names=FEATURE_NAMES,
    out_var_names=["u_t"],
    save=f"{RESULTS}/edge_activations",
)
plt.close(fig)

# Loss curves
if hasattr(model, "train_results_") and model.train_results_:
    fig, ax = plot_loss_curves(
        model.train_results_,
        save=f"{RESULTS}/loss_curves",
    )
    plt.close(fig)

print(f"[FIGS]  Saved {RESULTS}/")
