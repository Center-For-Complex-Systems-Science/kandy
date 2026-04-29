#!/usr/bin/env python3
"""KANDy example: Inviscid Burgers equation.

The inviscid Burgers PDE:
    u_t + u * u_x = 0      on x ∈ [-π, π], periodic

Koopman lift:  phi(u) = [u, u_x, u*u_x]
    -> 3 features per spatial point, KAN = [3, 1]

The u*u_x feature directly encodes the Burgers nonlinearity so the KAN
learns u_t ≈ -1 * u*u_x.

Data generation: Rusanov flux with RK45 (scipy), sin(x) IC, integrated
past shock time to t=1.4.

This example uses the lower-level ``fit_kan`` API directly because the PDE
rollout dynamics function requires spatial derivative computation that the
high-level ``KANDy`` class does not handle automatically.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from kan import KAN

from kandy.training import fit_kan
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# Spatial grid
X_MIN, X_MAX = -np.pi, np.pi
DX = 0.02
x_grid = np.arange(X_MIN, X_MAX + DX, DX)
NX = len(x_grid)

# Time grid
T0, T1 = 0.0, 1.4
DT = 0.002
t_grid = np.linspace(T0, T1, int(round((T1 - T0) / DT)) + 1)

# ---------------------------------------------------------------------------
# 1. Data generation — Rusanov flux + RK45
# ---------------------------------------------------------------------------
u0 = np.sin(x_grid).astype(np.float64)


def flux(u):
    return 0.5 * u**2


def burgers_rhs(_t, u):
    """MOL RHS with Rusanov (local Lax-Friedrichs) numerical flux."""
    uL = u
    uR = np.roll(u, -1)
    fL = flux(uL)
    fR = flux(uR)
    a = np.maximum(np.abs(uL), np.abs(uR))
    F_iphalf = 0.5 * (fL + fR) - 0.5 * a * (uR - uL)
    F_imhalf = np.roll(F_iphalf, 1)
    return -(F_iphalf - F_imhalf) / DX


print("[DATA]  Generating Burgers data with shock (Rusanov + RK45) ...")
sol = solve_ivp(
    burgers_rhs, (T0, T1), u0,
    t_eval=t_grid, method="RK45",
    rtol=1e-7, atol=1e-9,
)
if not sol.success:
    raise RuntimeError(sol.message)

U_true = sol.y.T.astype(np.float32)  # (Nt, Nx)
NT = U_true.shape[0]
print(f"[DATA]  Nt={NT}, Nx={NX}")

# ---------------------------------------------------------------------------
# 2. Sparse supervision snapshots & spatial derivatives
# ---------------------------------------------------------------------------
# All snapshots (not sparse) — matches research code final version
t_train = t_grid.astype(np.float32)
train_indices = list(range(len(t_train)))

X_snap = torch.tensor(U_true[train_indices], dtype=torch.float32, device=device)
t_snap = torch.tensor(t_train, dtype=torch.float32, device=device)
K = X_snap.shape[0]


def minmod(a, b):
    """Elementwise minmod limiter."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(torch.abs(a), torch.abs(b))


def tvd_ux(u, dx):
    """TVD minmod-limited spatial derivative, periodic."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    up = torch.roll(u, shifts=-1, dims=1)
    um = torch.roll(u, shifts=+1, dims=1)
    du_f = (up - u) / dx
    du_b = (u - um) / dx
    return minmod(du_b, du_f)


def ddx(u, dx=DX):
    """Central difference d/dx, periodic."""
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2 * dx)


# Forward-difference time derivatives
dt_seg = (t_snap[1:] - t_snap[:-1]).unsqueeze(1)  # (K-1, 1)
U_k = X_snap[:-1]                                  # (K-1, Nx)
Ut_k = (X_snap[1:] - X_snap[:-1]) / dt_seg         # (K-1, Nx)

# Spatial derivatives
ux_k = tvd_ux(U_k, DX)  # (K-1, Nx)

# ---------------------------------------------------------------------------
# 3. Feature library: [u, u_x, u*u_x]
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["u", "u_x", "u*u_x"]
N_FEATURES = 3


def build_burgers_library(u, ux=None):
    """Build [u, u_x, u*u_x] from (B, Nx) state tensor."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if ux is None:
        ux = tvd_ux(u, DX)
    Theta = torch.stack([u, ux, u * ux], dim=-1)  # (B, Nx, 3)
    B, N, F = Theta.shape
    return Theta.reshape(B * N, F)


Theta = build_burgers_library(U_k, ux_k)  # ((K-1)*Nx, F)
y = Ut_k.reshape(-1, 1)                   # ((K-1)*Nx, 1)

# No normalization — preserves physical coefficients for symbolic extraction
X_mean = torch.zeros(1, N_FEATURES)
X_std = torch.ones(1, N_FEATURES)
Theta_n = Theta

# Subsample for LBFGS (struggles with >80K samples)
MAX_SAMPLES = 80_000
N_total = Theta_n.shape[0]
g_perm = torch.Generator(device=device)
g_perm.manual_seed(SEED)
perm = torch.randperm(N_total, device=device, generator=g_perm)
if N_total > MAX_SAMPLES:
    perm = perm[:MAX_SAMPLES]
    N_total = MAX_SAMPLES

# Random train/test split
test_frac = 0.2
N_test = max(1, int(test_frac * N_total))
test_idx = perm[:N_test]
train_idx = perm[N_test:]

dataset = {
    "train_input": Theta_n[train_idx],
    "train_label": y[train_idx],
    "test_input":  Theta_n[test_idx],
    "test_label":  y[test_idx],
}

print(f"[DATA]  Features: {N_FEATURES}, samples: {N_total} (subsampled from {Theta_n.shape[0]})")
print(f"[DATA]  Train: {len(train_idx)}, Test: {N_test}")

# ---------------------------------------------------------------------------
# 4. Rollout windows for integration loss
# ---------------------------------------------------------------------------
ROLLOUT_HORIZON = 3

U_series = torch.tensor(U_true, dtype=torch.float32, device=device)
t_series = torch.tensor(t_grid, dtype=torch.float32, device=device)


def sample_windows(U, T, H, n_windows, seed=0):
    g = torch.Generator(device=U.device)
    g.manual_seed(seed)
    max_start = U.shape[0] - (H + 1)
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g, device=U.device)
    trj_u = torch.stack([U[s:s + H + 1] for s in starts], dim=0)
    trj_t = torch.stack([T[s:s + H + 1] for s in starts], dim=0)
    return trj_u, trj_t


train_u_trj, train_t_trj = sample_windows(U_series, t_series, ROLLOUT_HORIZON, 1, seed=1)
dataset["train_traj"] = train_u_trj  # (1, H+1, Nx)
dataset["train_t"] = train_t_trj     # (1, H+1)

# ---------------------------------------------------------------------------
# 5. KAN model (width=[3, 1]) and dynamics function
# ---------------------------------------------------------------------------
rbf = lambda x: torch.exp(-(x**2))
kan_pde = KAN(width=[N_FEATURES, 1], grid=7, k=3, base_fun=rbf, seed=SEED)


def dynamics_training_fn(u_state):
    """PDE dynamics: u (B, Nx) -> u_t (B, Nx) via feature library + KAN."""
    if u_state.dim() == 1:
        u_state = u_state.unsqueeze(0)
    ux = tvd_ux(u_state, DX)
    Theta_local = build_burgers_library(u_state, ux)
    Theta_local_n = (Theta_local - X_mean) / X_std
    ut = kan_pde(Theta_local_n)
    ut = ut[:, 0].reshape(u_state.shape[0], u_state.shape[1])
    return ut


# ---------------------------------------------------------------------------
# 6. Train with fit_kan (rollout loss enabled)
# ---------------------------------------------------------------------------
print("\n[TRAIN] Training KAN with rollout loss ...")
results = fit_kan(
    kan_pde,
    dataset,
    steps=50,
    rollout_weight=10.9,
    rollout_horizon=ROLLOUT_HORIZON,
    dynamics_fn=dynamics_training_fn,
    integrator="rk4",
    patience=0,
)

# ---------------------------------------------------------------------------
# 7. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
kan_pde.save_act = True
with torch.no_grad():
    _ = kan_pde(dataset["train_input"])

# Per-edge symbolic fitting with r2 threshold (matches robust_auto_symbolic)
import sympy as sp

R2_THRESHOLD = 0.80
for i in range(N_FEATURES):
    name, fun, r2, c = kan_pde.suggest_symbolic(
        0, i, 0, lib=["x", "0"], verbose=False, weight_simple=0.0
    )
    r2 = float(r2)
    if r2 >= R2_THRESHOLD and str(name) != "0":
        kan_pde.fix_symbolic(0, i, 0, str(name))
        print(f"  edge ({i}→0): {name}  r2={r2:.4f}")
    else:
        kan_pde.fix_symbolic(0, i, 0, "0")
        print(f"  edge ({i}→0): ZEROED  (best r2={r2:.4f})")

exprs, inputs = kan_pde.symbolic_formula()
sub_map = {
    sp.Symbol(str(inputs[i])): sp.Symbol(FEATURE_NAMES[i])
    for i in range(len(inputs))
}
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym)
    # Round small coefficients
    sym = sym.xreplace({n: round(float(n), 3) for n in sym.atoms(sp.Number)})
    print(f"  u_t = {sym}")

# ---------------------------------------------------------------------------
# 8. Full rollout from IC
# ---------------------------------------------------------------------------
def dynamics_fn_eval(u):
    """Evaluation-time dynamics (same as training but with no_grad)."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    ux = tvd_ux(u, DX)
    Theta_local = build_burgers_library(u, ux)
    Theta_local_n = (Theta_local - X_mean) / X_std
    ut = kan_pde(Theta_local_n)
    if ut.dim() == 2 and ut.shape[1] == 1:
        ut = ut[:, 0]
    return ut.reshape(u.shape[0], u.shape[1])


def rk4_step_pde(u, dt_val):
    k1 = dynamics_fn_eval(u)
    k2 = dynamics_fn_eval(u + 0.5 * dt_val * k1)
    k3 = dynamics_fn_eval(u + 0.5 * dt_val * k2)
    k4 = dynamics_fn_eval(u + dt_val * k3)
    return u + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


print("\n[EVAL]  Running full rollout ...")
with torch.no_grad():
    u_pred = torch.tensor(u0.astype(np.float32), device=device).unsqueeze(0)
    U_pred_list = [u_pred.squeeze(0).numpy()]
    for k_step in range(NT - 1):
        dt_k = float(t_grid[k_step + 1] - t_grid[k_step])
        u_pred = rk4_step_pde(u_pred, torch.tensor([[dt_k]], device=device))
        U_pred_list.append(u_pred.squeeze(0).numpy())

U_pred = np.stack(U_pred_list, axis=0)  # (Nt, Nx)

rmse_final = np.sqrt(np.mean((U_pred[-1] - U_true[-1]) ** 2))
rmse_all = np.sqrt(np.mean((U_pred - U_true) ** 2))
print(f"[EVAL]  Final-time RMSE: {rmse_final:.6f}")
print(f"[EVAL]  Full-trajectory RMSE: {rmse_all:.6f}")

# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
use_pub_style()
RESULTS = "results/Burgers"
os.makedirs(RESULTS, exist_ok=True)

# 9a. Space-time comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes, [U_true, U_pred], ["True Burgers", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[T0, T1, X_MIN, X_MAX],
                   cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.tight_layout()
fig.savefig(f"{RESULTS}/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9b. Final-time line plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_grid, U_true[-1], "k-", lw=1.5, label="True")
ax.plot(x_grid, U_pred[-1], "r--", lw=1.5, label="KANDy")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS}/final_snapshot.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/final_snapshot.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9c. Edge activations
n_sub = min(5000, len(train_idx))
sub_idx = train_idx[:n_sub]
train_t = Theta_n[sub_idx]

fig, axes = plot_all_edges(
    kan_pde,
    X=train_t,
    fits=["linear"],
    in_var_names=FEATURE_NAMES,
    out_var_names=["u_t"],
    save=f"{RESULTS}/edge_activations",
)
plt.close(fig)

# 9d. Loss curves
if results:
    fig, ax = plot_loss_curves(
        results,
        save=f"{RESULTS}/loss_curves",
    )
    plt.close(fig)

print(f"[FIGS]  Saved {RESULTS}/")
