#!/usr/bin/env python3
"""KANDy example: Kuramoto–Sivashinsky (KS) PDE.

The KS equation:
    u_t + u*u_x + u_xx + u_xxxx = 0

on a periodic domain [0, L) with L=64, Nx=128.

Feature library (12 features per spatial point):
    [u, u_x, u_xx, u_xxxx, u², u³, u_x², u_xx², u*u_x, u*u_xx, u*u_xxxx, u_x*u_xx]

KAN = [12, 1] with grid=10, k=5, RBF base.

Spatial derivatives via periodic finite-difference matrices.
Data generated with scipy BDF solver.

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
from kan import KAN
import sympy as sp

from kandy.training import fit_kan
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Problem parameters
# ---------------------------------------------------------------------------
L = 64.0
NX = 128
DX = L / NX
x_grid = np.arange(0.0, L, DX)

T_SPIN = 100.0
T_TRAIN = 20.0
DT = 0.1
t_all = np.linspace(0.0, T_SPIN + T_TRAIN, int(round((T_SPIN + T_TRAIN) / DT)) + 1)

# ---------------------------------------------------------------------------
# 2. Periodic finite-difference derivative matrices
# ---------------------------------------------------------------------------
def periodic_shift_mat(N, k):
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        M[i, (i + k) % N] = 1.0
    return M

I_mat = np.eye(NX, dtype=np.float64)
Sp = periodic_shift_mat(NX, +1)
Sm = periodic_shift_mat(NX, -1)
Spp = periodic_shift_mat(NX, +2)
Smm = periodic_shift_mat(NX, -2)

Dx_mat = (Sp - Sm) / (2.0 * DX)
Dxx_mat = (Sp - 2.0 * I_mat + Sm) / (DX**2)
Dxxxx_mat = (Smm - 4.0 * Sm + 6.0 * I_mat - 4.0 * Sp + Spp) / (DX**4)


def ks_rhs_np(_t, u):
    ux = Dx_mat @ u
    uxx = Dxx_mat @ u
    uxxxx = Dxxxx_mat @ u
    return -(u * ux) - uxx - uxxxx


# ---------------------------------------------------------------------------
# 3. Generate training data (scipy BDF with spin-up)
# ---------------------------------------------------------------------------
print("[DATA]  Generating KS data with spin-up (SciPy BDF) ...")
u0_init = (np.cos(2 * np.pi * x_grid / L) + 0.1 * np.random.randn(NX)).astype(np.float64)

sol = solve_ivp(
    ks_rhs_np, (0.0, T_SPIN + T_TRAIN), u0_init,
    t_eval=t_all, method="BDF",
    rtol=1e-6, atol=1e-8,
)
if not sol.success:
    raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")

U_all = sol.y.T  # (Nt_all, Nx)
spin_idx = int(round(T_SPIN / DT))
U_true = U_all[spin_idx:]
t_data = t_all[spin_idx:] - T_SPIN
NT = U_true.shape[0]

print(f"[DATA]  Nt={NT}, Nx={NX}")

# ---------------------------------------------------------------------------
# 4. Training data: all snapshots, forward-difference time derivatives
# ---------------------------------------------------------------------------
t_train = np.arange(0.0, T_TRAIN + DT, DT).astype(np.float32)
train_indices = [int(round(tt / DT)) for tt in t_train]

X_snap = torch.tensor(U_true[train_indices], dtype=torch.float32, device=device)
t_snap = torch.tensor(t_train, dtype=torch.float32, device=device)

dt_seg = (t_snap[1:] - t_snap[:-1]).unsqueeze(1)
U_k = X_snap[:-1]
Ut_k = (X_snap[1:] - X_snap[:-1]) / dt_seg

U_k_np = U_k.numpy()
Ut_k_np = Ut_k.numpy()

# Spatial derivatives via FD matrices
ux_np = U_k_np @ Dx_mat.T
uxx_np = U_k_np @ Dxx_mat.T
uxxxx_np = U_k_np @ Dxxxx_mat.T

# ---------------------------------------------------------------------------
# 5. Feature library: 12 features per spatial point
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "u", "u_x", "u_xx", "u_xxxx",
    "u^2", "u^3",
    "u_x^2", "u_xx^2",
    "u*u_x", "u*u_xx", "u*u_xxxx", "u_x*u_xx",
]
N_FEATURES = 12


def build_library_features(u, ux, uxx, uxxxx):
    feats = [u, ux, uxx, uxxxx]
    feats += [u**2, u**3]
    feats += [ux**2, uxx**2]
    feats += [u * ux, u * uxx, u * uxxxx, ux * uxx]
    Theta = np.stack(feats, axis=-1)          # (Nt, Nx, F)
    return Theta.reshape(-1, Theta.shape[-1])  # (Nt*Nx, F)


Theta_np = build_library_features(U_k_np, ux_np, uxx_np, uxxxx_np)
y_np = Ut_k_np.reshape(-1, 1)

X = torch.tensor(Theta_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np, dtype=torch.float32, device=device)

# Normalize features
X_mean = X.mean(dim=0, keepdim=True)
X_std = X.std(dim=0, keepdim=True) + 1e-8
Xn = (X - X_mean) / X_std

# Random train/test split
N_total = Xn.shape[0]
perm = torch.randperm(N_total, device=device)
N_test = max(1, int(0.2 * N_total))
test_idx = perm[:N_test]
train_idx = perm[N_test:]

dataset = {
    "train_input": Xn[train_idx],
    "train_label": y[train_idx],
    "test_input": Xn[test_idx],
    "test_label": y[test_idx],
}

print(f"[DATA]  Features: {N_FEATURES}, train: {len(train_idx)}, test: {N_test}")

# ---------------------------------------------------------------------------
# 6. Rollout windows for integration loss
# ---------------------------------------------------------------------------
ROLLOUT_HORIZON = 1

Dx_t = torch.tensor(Dx_mat, dtype=torch.float32, device=device)
Dxx_t = torch.tensor(Dxx_mat, dtype=torch.float32, device=device)
Dxxxx_t = torch.tensor(Dxxxx_mat, dtype=torch.float32, device=device)

U_series = torch.tensor(U_true, dtype=torch.float32, device=device)
t_series = torch.tensor(t_data, dtype=torch.float32, device=device)


def sample_windows(U, T, H, n_windows, seed=0):
    g = torch.Generator(device=U.device)
    g.manual_seed(seed)
    max_start = U.shape[0] - (H + 1)
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g, device=U.device)
    trj_u = torch.stack([U[s:s + H + 1] for s in starts], dim=0)
    trj_t = torch.stack([T[s:s + H + 1] for s in starts], dim=0)
    return trj_u, trj_t


train_u_trj, train_t_trj = sample_windows(U_series, t_series, ROLLOUT_HORIZON, 128, seed=1)
dataset["train_traj"] = train_u_trj
dataset["train_t"] = train_t_trj

# ---------------------------------------------------------------------------
# 7. KAN model and dynamics function
# ---------------------------------------------------------------------------
rbf = lambda x: torch.exp(-(x**2))
kan_pde = KAN(width=[N_FEATURES, 1], grid=10, k=5, base_fun=rbf, seed=SEED)


def dynamics_fn(state):
    """KS dynamics: state (B, Nx) -> u_t (B, Nx) via library + KAN."""
    if state.dim() == 1:
        state = state.unsqueeze(0)

    u = state
    ux = u @ Dx_t.T
    uxx = u @ Dxx_t.T
    uxxxx = u @ Dxxxx_t.T

    feats = [u, ux, uxx, uxxxx, u**2, u**3,
             ux**2, uxx**2, u * ux, u * uxx, u * uxxxx, ux * uxx]
    Theta = torch.stack(feats, dim=-1)  # (B, Nx, F)
    B, Nx_local, F = Theta.shape
    Theta = Theta.reshape(B * Nx_local, F)
    Theta_n = (Theta - X_mean) / X_std

    ut = kan_pde(Theta_n)
    if ut.dim() == 2 and ut.shape[1] == 1:
        ut = ut[:, 0]
    return ut.reshape(B, Nx_local)


# ---------------------------------------------------------------------------
# 8. Train with rollout loss
# ---------------------------------------------------------------------------
print("\n[TRAIN] Training KAN with rollout loss ...")
results = fit_kan(
    kan_pde,
    dataset,
    steps=100,
    lr=1e-3,
    lamb=0.0,
    rollout_weight=0.9,
    rollout_horizon=ROLLOUT_HORIZON,
    dynamics_fn=dynamics_fn,
    integrator="rk4",
    stop_grid_update_step=200,
    patience=0,
)

# ---------------------------------------------------------------------------
# 9. Symbolic extraction (robust_auto_symbolic approach)
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
kan_pde.save_act = True
with torch.no_grad():
    _ = kan_pde(dataset["train_input"][:2000])

kan_pde.auto_symbolic(lib=["x", "x^2", "x^3", "0"], weight_simple=0.1)
exprs_raw, vars_ = kan_pde.symbolic_formula()

u_sym, u_x_sym, u_xx_sym, u_xxxx_sym = sp.symbols("u u_x u_xx u_xxxx")
feature_syms = [
    u_sym, u_x_sym, u_xx_sym, u_xxxx_sym,
    u_sym**2, u_sym**3,
    u_x_sym**2, u_xx_sym**2,
    u_sym * u_x_sym, u_sym * u_xx_sym, u_sym * u_xxxx_sym, u_x_sym * u_xx_sym,
]

sub_map = {vars_[i]: feature_syms[i] for i in range(min(len(vars_), len(feature_syms)))}


def flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(flatten(it))
        return out
    return [obj]


def round_numbers(expr, places=3):
    return expr.xreplace({a: sp.Float(round(float(a), places))
                          for a in expr.atoms(sp.Number)})


cleaned = []
for expr in flatten(exprs_raw):
    if not hasattr(expr, "free_symbols"):
        continue
    expr_sub = expr.subs(sub_map)
    expr_sub = sp.together(sp.expand(expr_sub))
    expr_sub = round_numbers(expr_sub, 3)
    cleaned.append(expr_sub)

for ex in cleaned:
    print(f"  u_t = {ex}")
print(f"\n  [TRUE] u_t = -u*u_x - u_xx - u_xxxx")

# ---------------------------------------------------------------------------
# 10. Evaluation — pointwise MSE
# ---------------------------------------------------------------------------
with torch.no_grad():
    yhat = kan_pde(Xn).cpu().numpy().reshape(-1)
    ytrue = y.cpu().numpy().reshape(-1)
mse = np.mean((yhat - ytrue)**2)
print(f"\n[EVAL]  Overall MSE: {mse:.6e}")

# ---------------------------------------------------------------------------
# 11. Rollout using RK4 with substeps
# ---------------------------------------------------------------------------
n_sub = 10
h_sub = DT / n_sub

print("[EVAL]  Running rollout ...")
kan_pde.eval()
U_kan = np.zeros_like(U_true, dtype=np.float64)
U_kan[0] = U_true[0]

state = torch.tensor(U_true[0], dtype=torch.float32, device=device).unsqueeze(0)

with torch.no_grad():
    for step in range(1, NT):
        for _ in range(n_sub):
            k1 = dynamics_fn(state)
            k2 = dynamics_fn(state + 0.5 * h_sub * k1)
            k3 = dynamics_fn(state + 0.5 * h_sub * k2)
            k4 = dynamics_fn(state + h_sub * k3)
            state = state + (h_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        U_kan[step] = state.squeeze(0).numpy()
        if np.any(np.isnan(U_kan[step])) or np.abs(U_kan[step]).max() > 1e3:
            print(f"  Blowup at step {step}, stopping rollout")
            U_kan[step:] = np.nan
            break

N_valid = np.sum(~np.isnan(U_kan[:, 0]))
if N_valid > 1:
    rmse = np.sqrt(np.nanmean((U_kan[:N_valid] - U_true[:N_valid])**2))
    print(f"[EVAL]  Rollout RMSE ({N_valid} valid steps): {rmse:.6f}")

# ---------------------------------------------------------------------------
# 12. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/KS", exist_ok=True)

# 12a. Space-time heatmaps
N_plot = min(N_valid, NT)
t_plot = t_data[:N_plot]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes, [U_true[:N_plot], U_kan[:N_plot]], ["True KS", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[t_plot[0], t_plot[-1], 0, L],
                   cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.tight_layout()
fig.savefig("results/KS/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig("results/KS/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 12b. Edge activations
n_sub_plot = min(5000, len(train_idx))
sub_theta = Xn[train_idx[:n_sub_plot]]
fig, axes = plot_all_edges(
    kan_pde,
    X=sub_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["u_t"],
    save="results/KS/edge_activations",
)
plt.close(fig)

# 12c. Loss curves
if results:
    fig, ax = plot_loss_curves(
        results,
        save="results/KS/loss_curves",
    )
    plt.close(fig)

print(f"[FIGS]  Saved results/KS/")
