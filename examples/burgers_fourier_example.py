#!/usr/bin/env python3
"""KANDy example: Inviscid Burgers — Fourier-mode initial conditions.

Same PDE as burgers_example.py:
    u_t + u * u_x = 0

but with a random Fourier initial condition:
    u_0(x) = Σ_{k=1}^{K}  a_k * sin(k*x + phi_k),   a_k ∝ k^{-3/2}

Feature library includes u_xx so the KAN can learn a small viscosity
term if needed for stability:
    phi(u) = [u, u_x, u*u_x, u_xx]
    KAN = [4, 1]

Evaluation rollout uses Rusanov flux + KAN correction (blended)
to maintain stability through shocks.

This example uses the lower-level ``fit_kan`` API directly because the PDE
rollout dynamics function requires spatial derivative computation.
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
T0, T1 = 0.0, 3.0
DT = 0.002
t_grid = np.linspace(T0, T1, int(round((T1 - T0) / DT)) + 1)

# ---------------------------------------------------------------------------
# 1. Data generation — Random Fourier IC + Rusanov flux
# ---------------------------------------------------------------------------
K_FOURIER = 20
P_DECAY = 1.5
a_coeff = np.random.randn(K_FOURIER) * (np.arange(1, K_FOURIER + 1) ** (-P_DECAY))
phi_coeff = 2 * np.pi * np.random.rand(K_FOURIER)
u0 = sum(
    a_coeff[k] * np.sin((k + 1) * x_grid + phi_coeff[k]) for k in range(K_FOURIER)
).astype(np.float64)


def flux_np(u):
    return 0.5 * u**2


def burgers_rhs_np(_t, u):
    """Rusanov (LLF) flux, periodic BCs."""
    uL, uR = u, np.roll(u, -1)
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    F_iphalf = 0.5 * (flux_np(uL) + flux_np(uR)) - 0.5 * alpha * (uR - uL)
    F_imhalf = np.roll(F_iphalf, 1)
    return -(F_iphalf - F_imhalf) / DX


print("[DATA]  Generating Burgers ground truth (Rusanov + RK45) ...")
sol = solve_ivp(
    burgers_rhs_np, (T0, T1), u0,
    t_eval=t_grid, method="RK45", rtol=1e-7, atol=1e-9,
)
if not sol.success:
    raise RuntimeError(sol.message)

U_true = sol.y.T.astype(np.float32)  # (Nt, Nx)
NT = U_true.shape[0]
print(f"[DATA]  Nt={NT}, Nx={NX}")

# ---------------------------------------------------------------------------
# 2. Spatial derivatives (TVD, shock-robust)
# ---------------------------------------------------------------------------


def minmod(a, b):
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(torch.abs(a), torch.abs(b))


def tvd_ux(u, dx):
    """TVD minmod-limited first derivative, periodic."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    up = torch.roll(u, shifts=-1, dims=1)
    um = torch.roll(u, shifts=+1, dims=1)
    return minmod((u - um) / dx, (up - u) / dx)


def laplacian(u, dx):
    """Second derivative (for viscosity term), periodic."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    return (torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)) / (dx**2)


# ---------------------------------------------------------------------------
# 3. Training data: all snapshots, forward-difference time derivatives
# ---------------------------------------------------------------------------
t_train = t_grid.astype(np.float32)

X_snap = torch.tensor(U_true, dtype=torch.float32, device=device)  # (Nt, Nx)
t_snap = torch.tensor(t_train, dtype=torch.float32, device=device)

dt_seg = (t_snap[1:] - t_snap[:-1]).unsqueeze(1)  # (Nt-1, 1)
U_k = X_snap[:-1]                                   # (Nt-1, Nx)
Ut_k = (X_snap[1:] - X_snap[:-1]) / dt_seg          # (Nt-1, Nx)

ux_k = tvd_ux(U_k, DX)

# ---------------------------------------------------------------------------
# 4. Feature library: [u, u_x, u*u_x, u_xx]
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["u", "u_x", "u*u_x", "u_xx"]
N_FEATURES = 4


def build_library(u, ux=None):
    """Build feature matrix [u, u_x, u*u_x, u_xx] from (B, Nx) state."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if ux is None:
        ux = tvd_ux(u, DX)
    uxx = laplacian(u, DX)
    Theta = torch.stack([u, ux, u * ux, uxx], dim=-1)  # (B, Nx, 4)
    B, N, F = Theta.shape
    return Theta.reshape(B * N, F)


Theta = build_library(U_k, ux_k)  # ((Nt-1)*Nx, 4)
y = Ut_k.reshape(-1, 1)           # ((Nt-1)*Nx, 1)

# Normalize features
X_mean = Theta.mean(dim=0, keepdim=True)
X_std = Theta.std(dim=0, keepdim=True) + 1e-8
Theta_n = (Theta - X_mean) / X_std

Theta_np = Theta_n.numpy()
y_np = y.numpy()

print(f"[DATA]  Features: {N_FEATURES} {FEATURE_NAMES}")
print(f"[DATA]  Samples: {len(Theta_np)}")

# ---------------------------------------------------------------------------
# 5. Rollout windows for integration loss
# ---------------------------------------------------------------------------
ROLLOUT_HORIZON = 5

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


train_u_trj, train_t_trj = sample_windows(U_series, t_series, ROLLOUT_HORIZON, 12, seed=1)

# ---------------------------------------------------------------------------
# 6. KANDy model and PDE dynamics function
# ---------------------------------------------------------------------------
rbf = lambda x: torch.exp(-(3 * x**2))
burgers_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="burgers_fourier")

model = KANDy(lift=burgers_lift, grid=5, k=3, steps=50, seed=SEED, base_fun=rbf)


def dynamics_fn(u):
    """u: (B, Nx) -> u_t: (B, Nx) via feature library + KAN."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    Theta_local = build_library(u)
    Theta_local_n = (Theta_local - X_mean) / X_std
    ut = model.model_(Theta_local_n)
    if ut.dim() == 2 and ut.shape[1] == 1:
        ut = ut[:, 0]
    return ut.reshape(u.shape[0], u.shape[1])


# ---------------------------------------------------------------------------
# 7. Train via KANDy.fit() with custom dynamics and trajectory windows
# ---------------------------------------------------------------------------
print("\n[TRAIN] Training KAN with rollout loss ...")
model.fit(
    X=Theta_np,
    X_dot=y_np,
    val_frac=0.0,
    test_frac=0.2,
    lr=10,
    rollout_weight=10.1,
    rollout_horizon=ROLLOUT_HORIZON,
    dynamics_fn=dynamics_fn,
    dataset_extras={"train_traj": train_u_trj, "train_t": train_t_trj[0]},
    stop_grid_update_step=250,
    patience=0,
)

# ---------------------------------------------------------------------------
# 8. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
kan_pde = model.model_
kan_pde.save_act = True
with torch.no_grad():
    _ = kan_pde(model._train_input[:5000])

# Per-edge symbolic fitting with r2 threshold (matches robust_auto_symbolic)
import sympy as sp

R2_THRESHOLD = 0.95
for i in range(N_FEATURES):
    name, fun, r2, c = kan_pde.suggest_symbolic(
        0, i, 0, lib=["x", "0"], verbose=False, weight_simple=0.0
    )
    r2 = float(r2)
    if r2 >= R2_THRESHOLD and str(name) != "0":
        kan_pde.fix_symbolic(0, i, 0, str(name))
        print(f"  edge ({i}→0) [{FEATURE_NAMES[i]}]: {name}  r2={r2:.4f}")
    else:
        kan_pde.fix_symbolic(0, i, 0, "0")
        print(f"  edge ({i}→0) [{FEATURE_NAMES[i]}]: ZEROED  (best r2={r2:.4f})")

exprs, inputs = kan_pde.symbolic_formula()
sub_map = {
    sp.Symbol(str(inputs[i])): sp.Symbol(FEATURE_NAMES[i])
    for i in range(len(inputs))
}
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym)
    sym = sym.xreplace({n: round(float(n), 3) for n in sym.atoms(sp.Number)})
    print(f"  u_t = {sym}")

# ---------------------------------------------------------------------------
# 9. Evaluation rollout — Rusanov + KAN blended
# ---------------------------------------------------------------------------


def kan_rusanov_step_eval(u, dt_val, blend=0.3):
    """Blended: Rusanov backbone + KAN correction. blend=0 -> pure Rusanov."""
    nu_eval = 0.5 * DX

    def rhs(u_):
        uL, uR = u_, torch.roll(u_, shifts=-1, dims=1)
        fL, fR = 0.5 * uL**2, 0.5 * uR**2
        alpha = torch.maximum(torch.abs(uL), torch.abs(uR))
        F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        F_imhalf = torch.roll(F_iphalf, shifts=1, dims=1)
        base = -(F_iphalf - F_imhalf) / DX
        u_xx = (torch.roll(u_, -1, dims=1) - 2.0 * u_ + torch.roll(u_, 1, dims=1)) / (DX**2)
        base = base + nu_eval * u_xx
        kan_ut = dynamics_fn(u_)
        return base + blend * (kan_ut - base)

    # SSP-RK3
    s1 = u + dt_val * rhs(u)
    s2 = 0.75 * u + 0.25 * (s1 + dt_val * rhs(s1))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (s2 + dt_val * rhs(s2))


print("\n[EVAL]  Running blended Rusanov+KAN rollout ...")
with torch.no_grad():
    u_pred = torch.tensor(u0.astype(np.float32), device=device).unsqueeze(0)
    U_pred_list = [u_pred.squeeze(0).numpy()]
    for k_step in range(NT - 1):
        dt_k = float(t_grid[k_step + 1] - t_grid[k_step])
        u_pred = kan_rusanov_step_eval(u_pred, dt_k, blend=0.3)
        U_pred_list.append(u_pred.squeeze(0).numpy())
        if np.abs(U_pred_list[-1]).max() > 100:
            print(f"  WARNING: blowup at step {k_step+1}, stopping")
            break

U_pred = np.stack(U_pred_list, axis=0)

rmse_final = np.sqrt(np.mean((U_pred[-1] - U_true[:len(U_pred)][-1]) ** 2))
rmse_all = np.sqrt(np.mean((U_pred - U_true[:len(U_pred)]) ** 2))
print(f"[EVAL]  Final-time RMSE: {rmse_final:.6f}")
print(f"[EVAL]  Full-trajectory RMSE: {rmse_all:.6f}")

# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
use_pub_style()
RESULTS = "results/Burgers-Fourier"
os.makedirs(RESULTS, exist_ok=True)

# 10a. Space-time comparison
N_plot = len(U_pred)
t_plot = t_grid[:N_plot]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes,
                            [U_true[:N_plot], U_pred],
                            ["True Burgers (Fourier IC)", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[t_plot[0], t_plot[-1], X_MIN, X_MAX],
                   cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.tight_layout()
fig.savefig(f"{RESULTS}/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10b. Final-time snapshot
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_grid, U_true[N_plot - 1], "k-", lw=1.5, label="True")
ax.plot(x_grid, U_pred[-1], "r--", lw=1.5, label="KANDy")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS}/final_snapshot.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/final_snapshot.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 10c. Edge activations
n_sub = min(5000, len(Theta_np))
sub_theta = torch.tensor(Theta_np[:n_sub], dtype=torch.float32)

fig, axes = plot_all_edges(
    kan_pde,
    X=sub_theta,
    fits=["linear"],
    in_var_names=FEATURE_NAMES,
    out_var_names=["u_t"],
    save=f"{RESULTS}/edge_activations",
)
plt.close(fig)

# 10d. Loss curves
if model.train_results_:
    fig, ax = plot_loss_curves(
        model.train_results_,
        save=f"{RESULTS}/loss_curves",
    )
    plt.close(fig)

print(f"[FIGS]  Saved {RESULTS}/")
