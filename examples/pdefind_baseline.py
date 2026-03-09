#!/usr/bin/env python3
"""Baselines for KANDy comparison — Inviscid Burgers (sinusoidal IC).

Implements OLS, Ridge, LASSO, PDE-FIND (STLSQ) on the inviscid Burgers equation:
    u_t + (u²/2)_x = 0   →   u_t = -u·u_x

Uses the same data and derivative operators as burgers_example.py for a fair comparison.

Results saved to results/Burgers/baselines/
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RESULTS = "results/Burgers/baselines"
os.makedirs(RESULTS, exist_ok=True)

from kandy import solve_burgers
from kandy.numerics import muscl_reconstruct
from kandy.plotting import use_pub_style

use_pub_style()

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Generate data (same as burgers_example.py)
# ---------------------------------------------------------------------------
L    = 2.0 * np.pi
N_X  = 128
DT   = 0.005
N_T  = 3_000
BURN = 100

x_grid = np.linspace(0, L, N_X, endpoint=False)
dx     = L / N_X

u0 = np.sin(x_grid) + 0.5 * np.sin(2 * x_grid)

print("[DATA]  Simulating inviscid Burgers (Rusanov / TVD-RK2) ...")
U_full = solve_burgers(u0, n_steps=N_T, dt=DT, scheme="rusanov",
                       limiter="minmod", time_stepper="tvdrk2")
U = U_full[BURN:]
T_snap = U.shape[0]
print(f"[DATA]  T={T_snap} snapshots, N_x={N_X}")


# ---------------------------------------------------------------------------
# 2. Feature library (TVD minmod derivatives — same as KANDy)
# ---------------------------------------------------------------------------
def tvd_derivative(u, dx_):
    u_L, _ = muscl_reconstruct(u, dx_, limiter="minmod")
    return (u_L - u) * 2.0 / dx_


def laplacian(u, dx_):
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx_ ** 2)


# Time derivative via central differences
U_inner = U[1:-1]
U_dot   = (U[2:] - U[:-2]) / (2.0 * DT)

# Build feature library: [u, u_x, u*u_x, u_xx]
feat_names = ["u", "u_x", "u*u_x", "u_xx"]
rows = []
for t in range(U_inner.shape[0]):
    u   = U_inner[t]
    u_x = tvd_derivative(u, dx)
    u_xx = laplacian(u, dx)
    rows.append(np.column_stack([u, u_x, u * u_x, u_xx]))

Theta = np.vstack(rows)
y_all = U_dot.ravel()

# Subsample and split
MAX_SAMPLES = 60_000
rng = np.random.default_rng(SEED)
N_total = len(Theta)
perm = rng.permutation(N_total)
N_test = min(12_000, int(0.2 * N_total))
test_idx  = perm[:N_test]
train_idx = perm[N_test:N_test + min(MAX_SAMPLES, N_total - N_test)]

Theta_train = Theta[train_idx]
Theta_test  = Theta[test_idx]
y_train = y_all[train_idx]
y_test  = y_all[test_idx]

print(f"[DATA]  Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"[DATA]  Features: {feat_names}")


# ---------------------------------------------------------------------------
# 3. Rollout infrastructure
# ---------------------------------------------------------------------------
def ssp_rk3_step(u, dt_val, rhs_fn):
    k1 = rhs_fn(u)
    u1 = u + dt_val * k1
    k2 = rhs_fn(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt_val * k2)
    k3 = rhs_fn(u2)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt_val * k3)


def rollout_cfl(u0_, n_steps, dt_out, rhs_fn, dx_, cfl=0.35):
    u = u0_[np.newaxis, :].copy()
    traj = [u0_.copy()]
    for _ in range(n_steps - 1):
        umax = np.max(np.abs(u)) + 1e-12
        h_hyp = cfl * dx_ / umax
        n_sub = max(1, int(np.ceil(dt_out / h_hyp)))
        h = dt_out / n_sub
        for _ in range(n_sub):
            u = ssp_rk3_step(u, h, rhs_fn)
            if np.any(np.isnan(u)):
                traj.append(np.full_like(u0_, np.nan))
                return np.array(traj)
        traj.append(u[0].copy())
    return np.array(traj)


def make_rhs(coefs, intercept=0.0):
    c_u, c_ux, c_uux, c_uxx = coefs
    def rhs_fn(u):
        ux = tvd_derivative(u.ravel(), dx)[np.newaxis, :]
        uxx = laplacian(u.ravel(), dx)[np.newaxis, :]
        return c_u * u + c_ux * ux + c_uux * (u * ux) + c_uxx * uxx + intercept
    return rhs_fn


def compute_nrmse(U_pred, U_ref):
    mask = np.isfinite(U_pred) & np.isfinite(U_ref)
    if mask.sum() == 0:
        return np.inf
    return np.sqrt(np.mean((U_pred[mask] - U_ref[mask]) ** 2) / np.var(U_ref[mask]))


# Reference rollout data
N_ROLL = 300
t0_idx = U_inner.shape[0] - N_ROLL
u0_roll = U_inner[t0_idx]
true_roll = U_inner[t0_idx:t0_idx + N_ROLL]

results = []
equations = {}
rollouts = {}


def eval_method(name, coefs, intercept=0.0):
    rhs = make_rhs(coefs, intercept)
    pred_test = Theta_test @ coefs + intercept
    mse = float(np.mean((pred_test - y_test) ** 2))
    r2 = 1.0 - mse / (np.var(y_test) + 1e-14)

    pred_roll = rollout_cfl(u0_roll, N_ROLL, DT, rhs, dx)
    nrmse = compute_nrmse(pred_roll, true_roll)

    eq_terms = []
    for c, n in zip(coefs, feat_names):
        if abs(c) > 1e-6:
            eq_terms.append(f"{c:+.6f}*{n}")
    if abs(intercept) > 1e-6:
        eq_terms.append(f"{intercept:+.6f}")
    eq = "u_t = " + " ".join(eq_terms) if eq_terms else "u_t = 0"

    results.append({"name": name, "mse": mse, "r2": r2, "nrmse": nrmse})
    equations[name] = eq
    rollouts[name] = pred_roll
    print(f"  {name:40s} | MSE={mse:.3e} R²={r2:.4f} | NRMSE={nrmse:.4f}")
    print(f"    Equation: {eq}")
    return pred_roll


# ---------------------------------------------------------------------------
# 4. Run baselines
# ---------------------------------------------------------------------------

# --- OLS ---
print("\n=== OLS ===")
from sklearn.linear_model import LinearRegression
ols = LinearRegression(fit_intercept=True)
ols.fit(Theta_train, y_train)
eval_method("OLS", ols.coef_, ols.intercept_)

# --- Ridge ---
print("\n=== Ridge ===")
from sklearn.linear_model import Ridge
best_ridge, best_ridge_mse = None, np.inf
for alpha in [1e-6, 1e-4, 1e-2, 0.1, 1.0]:
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(Theta_train, y_train)
    mse = np.mean((reg.predict(Theta_test) - y_test) ** 2)
    if mse < best_ridge_mse:
        best_ridge_mse = mse
        best_ridge = reg
print(f"  Best alpha: {best_ridge.alpha}")
eval_method("Ridge", best_ridge.coef_, best_ridge.intercept_)

# --- LASSO ---
print("\n=== LASSO ===")
from sklearn.linear_model import Lasso
best_lasso, best_lasso_mse = None, np.inf
for alpha in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
    reg = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True)
    reg.fit(Theta_train, y_train)
    mse = np.mean((reg.predict(Theta_test) - y_test) ** 2)
    if mse < best_lasso_mse:
        best_lasso_mse = mse
        best_lasso = reg
print(f"  Best alpha: {best_lasso.alpha}")
eval_method("LASSO", best_lasso.coef_, best_lasso.intercept_)

# --- STLSQ (PDE-FIND) ---
print("\n=== PDE-FIND (STLSQ) ===")
best_stlsq_mse = np.inf
best_stlsq_coefs = None
best_thr = None

for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    coefs = np.linalg.lstsq(Theta_train, y_train, rcond=None)[0]
    for _ in range(30):
        mask = np.abs(coefs) > threshold
        if mask.sum() == 0:
            break
        coefs_new = np.zeros_like(coefs)
        coefs_new[mask] = np.linalg.lstsq(Theta_train[:, mask], y_train, rcond=None)[0]
        if np.allclose(coefs_new, coefs, atol=1e-12):
            break
        coefs = coefs_new

    pred = Theta_test @ coefs
    mse = np.mean((pred - y_test) ** 2)
    n_active = np.sum(np.abs(coefs) > 1e-8)
    if mse < best_stlsq_mse:
        best_stlsq_mse = mse
        best_stlsq_coefs = coefs.copy()
        best_thr = threshold

print(f"  Best threshold: {best_thr}")
eval_method("STLSQ", best_stlsq_coefs)

# --- Ideal sparse (only u*u_x) ---
print("\n=== Ideal sparse (u*u_x only) ===")
c = np.linalg.lstsq(Theta_train[:, 2:3], y_train, rcond=None)[0][0]
ideal_coefs = np.array([0.0, 0.0, c, 0.0])
eval_method(f"Ideal sparse (c={c:.4f})", ideal_coefs)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
print("\n[FIGS]  Generating plots ...")

# Space-time comparisons
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
t_arr = np.arange(N_ROLL) * DT
vmin, vmax = true_roll.min(), true_roll.max()

plot_items = [
    ("Ground truth", true_roll),
    ("OLS", rollouts["OLS"]),
    ("Ridge", rollouts["Ridge"]),
    ("LASSO", rollouts["LASSO"]),
    ("STLSQ", rollouts["STLSQ"]),
    (f"Ideal (c={c:.3f})", rollouts.get(f"Ideal sparse (c={c:.4f})", true_roll)),
]

for ax, (title, data) in zip(axes.ravel(), plot_items):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[0, t_arr[-1], 0, L],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xlabel("time"); ax.set_ylabel("x")
    ax.set_title(title)

fig.colorbar(im, ax=axes, label="u(x,t)", shrink=0.8)
fig.suptitle("Baselines — Inviscid Burgers (sinusoidal IC)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS}/baselines_spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/baselines_spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# Snapshot comparison at final time
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_grid, true_roll[-1], 'k-', lw=2, label="Ground truth")
for name in ["OLS", "Ridge", "LASSO", "STLSQ"]:
    roll = rollouts[name]
    if roll.shape[0] == N_ROLL and np.all(np.isfinite(roll[-1])):
        ax.plot(x_grid, roll[-1], '--', lw=1.2, label=name)
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title(f"Final snapshot (t = {t_arr[-1]:.2f})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS}/baselines_snapshot.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/baselines_snapshot.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# NRMSE over time
fig, ax = plt.subplots(figsize=(10, 4))
for name in ["OLS", "Ridge", "LASSO", "STLSQ"]:
    roll = rollouts[name]
    nrmses = []
    for i in range(min(N_ROLL, roll.shape[0])):
        ref_var = np.var(true_roll[i])
        if ref_var < 1e-14 or not np.all(np.isfinite(roll[i])):
            nrmses.append(np.nan)
        else:
            nrmses.append(np.sqrt(np.mean((roll[i] - true_roll[i]) ** 2) / ref_var))
    ax.plot(t_arr[:len(nrmses)], nrmses, label=name, lw=1.5)
ax.set_xlabel("t"); ax.set_ylabel("NRMSE")
ax.set_title("Rollout NRMSE — Baselines (Inviscid Burgers)")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_yscale("log"); ax.set_ylim(bottom=1e-4)
fig.tight_layout()
fig.savefig(f"{RESULTS}/baselines_nrmse.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/baselines_nrmse.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
report = f"""# Baselines — Inviscid Burgers (sinusoidal IC)

## Setup
- Domain: [0, 2π], N={N_X}, dx={dx:.4f}
- Time: dt={DT}, N_T={N_T}, BURN={BURN}
- IC: sin(x) + 0.5*sin(2x)
- Data: kandy.numerics.solve_burgers (Rusanov, TVD minmod, TVD-RK2)
- Features: {feat_names}
- Train: {len(train_idx)}, Test: {len(test_idx)}
- Rollout: SSP-RK3 with CFL substeps, {N_ROLL} steps

## True equation
u_t = -1.0 * u*u_x

## Results

| Method | 1-step MSE | R² | Rollout NRMSE |
|--------|-----------|------|---------------|
"""

for r in results:
    nrmse_str = f"{r['nrmse']:.4f}" if np.isfinite(r['nrmse']) else "N/A"
    report += f"| {r['name']} | {r['mse']:.3e} | {r['r2']:.4f} | {nrmse_str} |\n"

report += "\n## Discovered Equations\n\n"
for name, eq in equations.items():
    report += f"**{name}:**\n```\n{eq}\n```\n\n"

report += """## Notes
- Feature library [u, u_x, u*u_x, u_xx] matches KANDy's Koopman lift.
- True equation: u_t = -1.0*u*u_x. Only the u*u_x coefficient should be non-zero.
- Spatial derivatives use TVD minmod (consistent with KANDy).
"""

with open(f"{RESULTS}/baselines_report.md", "w") as f:
    f.write(report)
print(f"\n[DONE]  Report: {RESULTS}/baselines_report.md")
print(f"[DONE]  Figures: {RESULTS}/baselines_*.png")
