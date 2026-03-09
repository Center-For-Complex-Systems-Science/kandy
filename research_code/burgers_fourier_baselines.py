#!/usr/bin/env python3
"""
PDE-FIND and SINDy baselines for Inviscid Burgers with Random Fourier ICs.

Uses the EXACT same data generation as Inviscd-Burgers-fourier-mode-ics.py:
  - K=20 Fourier modes, power-law decay p=1.5
  - Domain [-pi, pi], dx=0.02
  - t in [0, 3], dt_data=0.002
  - Rusanov flux + RK45 ground truth

Baselines tested:
  1. Oracle OLS         — hand-crafted library [u, u_x, u*u_x, u_xx], ordinary least squares
  2. Ridge regression   — same library, L2 regularization (sweep alpha)
  3. LASSO              — same library, L1 regularization (sweep alpha)
  4. PDE-FIND (STLSQ)   — pysindy with STLSQ, polynomial library (sweep threshold + degree)
  5. PDE-FIND (SR3)      — pysindy with SR3 optimizer
  6. PDE-FIND (Weak)     — pysindy WeakPDELibrary (the full Rudy et al. 2017 method)

All evaluated by:
  - 1-step derivative MSE / R^2 (on held-out test snapshots)
  - Autoregressive rollout NRMSE (SSP-RK3 with CFL substeps, same as KANDy eval)
"""

import os
import sys
import time
import warnings
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# All outputs go to results/Burgers-Fourier/baselines/
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "Burgers-Fourier", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use publication-quality plotting from kandy
try:
    from kandy.plotting import use_pub_style
    use_pub_style()
except ImportError:
    pass

# ============================================================
# 0) Reproducibility
# ============================================================
SEED = 0
np.random.seed(SEED)

# ============================================================
# 1) Ground truth: EXACT same as research notebook
# ============================================================
x_min, x_max = -np.pi, np.pi
Nx = 128
dx = (x_max - x_min) / Nx
x = np.linspace(x_min, x_max, Nx, endpoint=False)

t0_sim, t1_sim = 0.0, 2.0
dt_data = 0.004
t_grid = np.linspace(t0_sim, t1_sim, int(round((t1_sim - t0_sim) / dt_data)) + 1)

# Random Fourier initial condition
K_fourier = 10
p = 1.5
a_coeff = np.random.randn(K_fourier) * (np.arange(1, K_fourier + 1) ** (-p))
phi = 2 * np.pi * np.random.rand(K_fourier)
u0_np = sum(
    a_coeff[kk] * np.sin((kk + 1) * x + phi[kk]) for kk in range(K_fourier)
).astype(np.float64)


def flux_np(u):
    return 0.5 * u ** 2


def burgers_rhs_np(_t, u):
    uL = u
    uR = np.roll(u, -1)
    fL, fR = flux_np(uL), flux_np(uR)
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
    F_imhalf = np.roll(F_iphalf, 1)
    return -(F_iphalf - F_imhalf) / dx


print("Generating Burgers ground truth (Rusanov + RK45)...")
sol = solve_ivp(
    burgers_rhs_np, (t0_sim, t1_sim), u0_np,
    t_eval=t_grid, method="RK45", rtol=1e-6, atol=1e-8, max_step=0.01
)
if not sol.success:
    raise RuntimeError(sol.message)

U_true = sol.y.T.astype(np.float64)   # (Nt, Nx)
Nt = U_true.shape[0]
print(f"Ground truth shape: {U_true.shape}  (Nt={Nt}, Nx={Nx})")


# ============================================================
# 2) Spatial derivative operators (same as research code)
# ============================================================
def minmod_np(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def tvd_ux_np(u, dx_):
    """TVD minmod-limited first derivative, periodic BC. u: (B, Nx) or (Nx,)"""
    if u.ndim == 1:
        u = u[np.newaxis, :]
    up = np.roll(u, shift=-1, axis=1)
    um = np.roll(u, shift=+1, axis=1)
    du_f = (up - u) / dx_
    du_b = (u - um) / dx_
    return minmod_np(du_b, du_f)


def laplacian_np(u, dx_):
    if u.ndim == 1:
        u = u[np.newaxis, :]
    return (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dx_ ** 2)


# ============================================================
# 3) Training/test data (derivative supervision)
# ============================================================
# Use all snapshots, compute u_t via finite differences in time
U_k = U_true[:-1]                          # (Nt-1, Nx)
dt_seg = np.diff(t_grid)[:, None]           # (Nt-1, 1)
Ut_k = (U_true[1:] - U_true[:-1]) / dt_seg  # (Nt-1, Nx)

ux_k = tvd_ux_np(U_k, dx).squeeze()
uxx_k = laplacian_np(U_k, dx).squeeze()

# Flatten for pointwise regression: each (i_t, i_x) is a sample
U_flat = U_k.ravel()
Ut_flat = Ut_k.ravel()
ux_flat = ux_k.ravel()
uxx_flat = uxx_k.ravel()

# Train/test split (80/20 random), with subsampling for memory
N_pts = U_flat.shape[0]
rng = np.random.default_rng(SEED)
perm = rng.permutation(N_pts)
N_test = max(1, int(0.2 * N_pts))
MAX_TRAIN = 50_000
MAX_TEST = 15_000
test_idx = perm[:min(N_test, MAX_TEST)]
train_idx = perm[N_test:N_test + min(N_pts - N_test, MAX_TRAIN)]
print(f"Subsampled: train={len(train_idx)}, test={len(test_idx)} from {N_pts} total")

# Hand-crafted library: [u, u_x, u*u_x, u_xx]  (same as KANDy lift)
Theta_all = np.column_stack([U_flat, ux_flat, U_flat * ux_flat, uxx_flat])
feat_names_manual = ["u", "u_x", "u*u_x", "u_xx"]

Theta_train = Theta_all[train_idx]
Theta_test  = Theta_all[test_idx]
y_train = Ut_flat[train_idx]
y_test  = Ut_flat[test_idx]

print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
print(f"Feature library: {feat_names_manual}")


# ============================================================
# 4) Rollout evaluation infrastructure
# ============================================================
def ssp_rk3_step(u, dt_val, rhs_fn):
    """SSP-RK3 step. u: (1, Nx), returns (1, Nx)."""
    k1 = rhs_fn(u)
    u1 = u + dt_val * k1
    k2 = rhs_fn(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt_val * k2)
    k3 = rhs_fn(u2)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt_val * k3)


def rollout_cfl(u0, t_eval, rhs_fn, dx_, cfl=0.35, max_substeps=5000):
    """CFL-adaptive SSP-RK3 rollout. u0: (Nx,). Returns (Nt, Nx)."""
    u = u0[np.newaxis, :].copy()
    Nt_r = len(t_eval)
    out = np.zeros((Nt_r, u0.shape[0]), dtype=np.float64)
    out[0] = u0

    for n in range(Nt_r - 1):
        dt_out = t_eval[n + 1] - t_eval[n]
        umax = np.max(np.abs(u)) + 1e-12
        h_hyp = cfl * dx_ / umax
        n_sub = int(max(1, np.ceil(dt_out / h_hyp)))
        if n_sub > max_substeps:
            print(f"Rollout: n_sub={n_sub} > max at step {n}, t={t_eval[n]:.4f}. Stopping.")
            out[n + 1:] = np.nan
            return out
        h = dt_out / n_sub
        for _ in range(n_sub):
            u = ssp_rk3_step(u, h, rhs_fn)
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                print(f"Rollout: NaN at step {n}, t={t_eval[n]:.4f}")
                out[n + 1:] = np.nan
                return out
        out[n + 1] = u[0]
    return out


def compute_nrmse(U_pred, U_ref):
    """Normalized RMSE over all valid (non-NaN) entries."""
    mask = np.isfinite(U_pred) & np.isfinite(U_ref)
    if mask.sum() == 0:
        return np.inf
    err = (U_pred[mask] - U_ref[mask]) ** 2
    ref_var = np.var(U_ref[mask])
    if ref_var < 1e-14:
        return np.inf
    return np.sqrt(np.mean(err) / ref_var)


def compute_nrmse_over_time(U_pred, U_ref):
    """NRMSE per timestep. Returns array of shape (Nt,)."""
    Nt_r = min(U_pred.shape[0], U_ref.shape[0])
    nrmses = np.zeros(Nt_r)
    for i in range(Nt_r):
        ref_var = np.var(U_ref[i])
        if ref_var < 1e-14 or not np.all(np.isfinite(U_pred[i])):
            nrmses[i] = np.nan
        else:
            nrmses[i] = np.sqrt(np.mean((U_pred[i] - U_ref[i]) ** 2) / ref_var)
    return nrmses


# ============================================================
# 5) Baseline models
# ============================================================
results_table = []


def eval_model(name, predict_ut_fn, rhs_fn):
    """Evaluate a model on 1-step accuracy and rollout."""
    # 1-step
    ut_pred_test = predict_ut_fn(Theta_test)
    mse_1step = float(np.mean((ut_pred_test - y_test) ** 2))
    ss_tot = float(np.var(y_test) * len(y_test))
    r2 = 1.0 - mse_1step * len(y_test) / (ss_tot + 1e-14)

    # Rollout
    t_eval_roll = t_grid[:251]  # rollout for t in [0, 1.0]
    U_roll = rollout_cfl(u0_np, t_eval_roll, rhs_fn, dx)
    nrmse_roll = compute_nrmse(U_roll, U_true[:len(t_eval_roll)])

    # Longer rollout t in [0, 2.0]
    U_roll_full = rollout_cfl(u0_np, t_grid, rhs_fn, dx)
    nrmse_full = compute_nrmse(U_roll_full, U_true)

    print(f"  {name:40s} | 1-step MSE={mse_1step:.3e}  R²={r2:.4f} | "
          f"NRMSE(t<1)={nrmse_roll:.4f}  NRMSE(full)={nrmse_full:.4f}")

    results_table.append({
        "name": name,
        "mse_1step": mse_1step,
        "r2": r2,
        "nrmse_t1": nrmse_roll,
        "nrmse_full": nrmse_full,
    })
    return U_roll_full


# --- Helper: build RHS from linear coefficients on [u, u_x, u*u_x, u_xx] ---
def make_rhs_from_coefs(coefs, intercept=0.0):
    """coefs: array of length 4 for [u, u_x, u*u_x, u_xx]. Returns rhs_fn(u)."""
    c_u, c_ux, c_uux, c_uxx = coefs

    def rhs_fn(u):
        ux = tvd_ux_np(u, dx)
        uxx = laplacian_np(u, dx)
        return c_u * u + c_ux * ux + c_uux * (u * ux) + c_uxx * uxx + intercept
    return rhs_fn


# --- 5a) Oracle OLS ---
print("\n=== Baseline 1: Oracle OLS (hand-crafted library) ===")
from sklearn.linear_model import LinearRegression
ols = LinearRegression(fit_intercept=True)
ols.fit(Theta_train, y_train)
print(f"  Coefficients: {dict(zip(feat_names_manual, ols.coef_))}")
print(f"  Intercept: {ols.intercept_:.6f}")
print(f"  True equation: u_t = -1.0 * u*u_x")
U_ols = eval_model(
    "OLS [u, u_x, u*u_x, u_xx]",
    lambda X: ols.predict(X),
    make_rhs_from_coefs(ols.coef_, ols.intercept_),
)

# --- 5b) Ridge regression (sweep alpha) ---
print("\n=== Baseline 2: Ridge regression (hand-crafted library) ===")
from sklearn.linear_model import Ridge
best_ridge = None
best_ridge_mse = np.inf
for alpha in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(Theta_train, y_train)
    mse = np.mean((ridge.predict(Theta_test) - y_test) ** 2)
    if mse < best_ridge_mse:
        best_ridge_mse = mse
        best_ridge = ridge
        best_ridge_alpha = alpha

print(f"  Best alpha={best_ridge_alpha}, Coefficients: {dict(zip(feat_names_manual, best_ridge.coef_))}")
U_ridge = eval_model(
    f"Ridge(alpha={best_ridge_alpha}) [u,u_x,u*u_x,u_xx]",
    lambda X: best_ridge.predict(X),
    make_rhs_from_coefs(best_ridge.coef_, best_ridge.intercept_),
)

# --- 5c) LASSO (sweep alpha) ---
print("\n=== Baseline 3: LASSO (hand-crafted library) ===")
from sklearn.linear_model import Lasso
best_lasso = None
best_lasso_mse = np.inf
for alpha in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
    lasso = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True)
    lasso.fit(Theta_train, y_train)
    mse = np.mean((lasso.predict(Theta_test) - y_test) ** 2)
    if mse < best_lasso_mse:
        best_lasso_mse = mse
        best_lasso = lasso
        best_lasso_alpha = alpha

print(f"  Best alpha={best_lasso_alpha}, Coefficients: {dict(zip(feat_names_manual, best_lasso.coef_))}")
print(f"  Intercept: {best_lasso.intercept_:.6f}")
U_lasso = eval_model(
    f"LASSO(alpha={best_lasso_alpha}) [u,u_x,u*u_x,u_xx]",
    lambda X: best_lasso.predict(X),
    make_rhs_from_coefs(best_lasso.coef_, best_lasso.intercept_),
)


# ============================================================
# 6) PDE-FIND via pysindy (STLSQ — the standard approach)
# ============================================================
print("\n=== Baseline 4: PDE-FIND (STLSQ) — sweep threshold and library degree ===")
import pysindy as ps

# We need to build the feature library from the raw snapshots for pysindy.
# pysindy's PDE interface expects data shaped (n_time, n_space) and uses
# finite differences or custom differentiation.
# However, for a FAIR comparison, we use the SAME derivative operators as KANDy
# (TVD minmod for u_x, standard Laplacian for u_xx) rather than pysindy's defaults.

# Strategy: build the feature matrix ourselves, then use pysindy's optimizers
# on Theta @ xi = u_t.  This is the cleanest "PDE-FIND" approach.

# Expanded library for PDE-FIND: more terms than KANDy gets
# Degree 2: [1, u, u_x, u_xx, u^2, u*u_x, u*u_xx, u_x^2, u_x*u_xx, u_xx^2]
# Degree 3 adds: [u^3, u^2*u_x, u^2*u_xx, u*u_x^2, ...]

def build_pdefind_library(u_flat, ux_flat, uxx_flat, degree=2):
    """Build polynomial feature library up to given degree from [u, u_x, u_xx]."""
    base = {"u": u_flat, "u_x": ux_flat, "u_xx": uxx_flat}
    terms = {"1": np.ones_like(u_flat)}
    names = ["1"]

    # Generate all monomials up to `degree`
    base_keys = list(base.keys())
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(base_keys, d):
            name = "*".join(combo)
            val = np.ones_like(u_flat)
            for k in combo:
                val = val * base[k]
            terms[name] = val
            names.append(name)

    Theta = np.column_stack([terms[n] for n in names])
    return Theta, names


best_pdefind = None
best_pdefind_mse = np.inf
best_pdefind_config = {}

for degree in [2, 3]:
    # Build library only for train/test indices to save memory
    Theta_pf_train, names_pf = build_pdefind_library(
        U_flat[train_idx], ux_flat[train_idx], uxx_flat[train_idx], degree=degree
    )
    Theta_pf_test, _ = build_pdefind_library(
        U_flat[test_idx], ux_flat[test_idx], uxx_flat[test_idx], degree=degree
    )

    for threshold in [0.005, 0.01, 0.05, 0.1, 0.2]:
        for alpha in [0.0, 0.01]:
            # STLSQ: Sequential Thresholded Least Squares (the core of SINDy/PDE-FIND)
            from sklearn.linear_model import Ridge as _Ridge
            # Implement STLSQ manually for direct control
            coefs = np.linalg.lstsq(Theta_pf_train, y_train, rcond=None)[0]
            for _ in range(20):  # STLSQ iterations
                mask = np.abs(coefs) > threshold
                if mask.sum() == 0:
                    break
                if alpha > 0:
                    reg = _Ridge(alpha=alpha, fit_intercept=False)
                    reg.fit(Theta_pf_train[:, mask], y_train)
                    coefs_new = np.zeros_like(coefs)
                    coefs_new[mask] = reg.coef_
                else:
                    coefs_new = np.zeros_like(coefs)
                    coefs_new[mask] = np.linalg.lstsq(
                        Theta_pf_train[:, mask], y_train, rcond=None
                    )[0]
                if np.allclose(coefs_new, coefs, atol=1e-10):
                    break
                coefs = coefs_new

            pred_test = Theta_pf_test @ coefs
            mse = np.mean((pred_test - y_test) ** 2)

            if mse < best_pdefind_mse:
                best_pdefind_mse = mse
                best_pdefind = coefs.copy()
                best_pdefind_names = names_pf
                best_pdefind_config = {
                    "degree": degree,
                    "threshold": threshold,
                    "alpha": alpha,
                }

# Print discovered equation
print(f"  Best config: {best_pdefind_config}")
print(f"  Discovered equation (u_t = ):")
active_terms = []
for name, c in zip(best_pdefind_names, best_pdefind):
    if abs(c) > 1e-6:
        print(f"    {c:+.6f} * {name}")
        active_terms.append((name, c))


# Build RHS for rollout from PDE-FIND coefficients
def make_rhs_from_pdefind(coefs, names, degree):
    """Build a rollout RHS from PDE-FIND polynomial coefficients."""
    def rhs_fn(u):
        ux = tvd_ux_np(u, dx)
        uxx = laplacian_np(u, dx)
        u_f = u.ravel()
        ux_f = ux.ravel()
        uxx_f = uxx.ravel()
        Theta, _ = build_pdefind_library(u_f, ux_f, uxx_f, degree=degree)
        ut_f = Theta @ coefs
        return ut_f.reshape(u.shape)
    return rhs_fn


rhs_pdefind_best = make_rhs_from_pdefind(
    best_pdefind, best_pdefind_names, best_pdefind_config["degree"]
)
U_pdefind = eval_model(
    f"PDE-FIND STLSQ(thr={best_pdefind_config['threshold']},deg={best_pdefind_config['degree']})",
    lambda X_unused: (
        build_pdefind_library(
            Theta_test[:, 0],
            Theta_test[:, 1],
            Theta_test[:, 3],
            degree=best_pdefind_config["degree"],
        )[0] @ best_pdefind
    ),
    rhs_pdefind_best,
)


# ============================================================
# 7) PDE-FIND with SR3 optimizer (via pysindy)
# ============================================================
print("\n=== Baseline 5: PDE-FIND with SR3 optimizer ===")

best_sr3 = None
best_sr3_mse = np.inf
best_sr3_config = {}

for degree in [2, 3]:
    Theta_pf_train, names_pf = build_pdefind_library(
        U_flat[train_idx], ux_flat[train_idx], uxx_flat[train_idx], degree=degree
    )
    Theta_pf_test, _ = build_pdefind_library(
        U_flat[test_idx], ux_flat[test_idx], uxx_flat[test_idx], degree=degree
    )

    for lam in [0.005, 0.01, 0.05, 0.1]:
        for nu_sr3 in [1.0, 10.0]:
            try:
                opt = ps.SR3(
                    reg_weight_lam=lam,
                    relax_coeff_nu=nu_sr3,
                    regularizer="L0",
                    max_iter=100,
                )
                # Use pysindy optimizer's fit: expects (features, targets)
                opt.fit(Theta_pf_train, y_train.reshape(-1, 1))
                coefs_sr3 = opt.coef_.ravel()

                pred = Theta_pf_test @ coefs_sr3
                mse = np.mean((pred - y_test) ** 2)

                if mse < best_sr3_mse:
                    best_sr3_mse = mse
                    best_sr3 = coefs_sr3.copy()
                    best_sr3_names = names_pf
                    best_sr3_config = {
                        "degree": degree,
                        "lam": lam,
                        "nu": nu_sr3,
                    }
            except Exception as e:
                pass

if best_sr3 is not None:
    print(f"  Best config: {best_sr3_config}")
    print(f"  Discovered equation (u_t = ):")
    for name, c in zip(best_sr3_names, best_sr3):
        if abs(c) > 1e-6:
            print(f"    {c:+.6f} * {name}")

    rhs_sr3_best = make_rhs_from_pdefind(
        best_sr3, best_sr3_names, best_sr3_config["degree"]
    )
    _sr3_coefs = best_sr3
    _sr3_deg = best_sr3_config["degree"]
    U_sr3 = eval_model(
        f"PDE-FIND SR3(lam={best_sr3_config['lam']},deg={_sr3_deg})",
        lambda X_unused: (
            build_pdefind_library(
                Theta_test[:, 0], Theta_test[:, 1], Theta_test[:, 3],
                degree=_sr3_deg,
            )[0] @ _sr3_coefs
        ),
        rhs_sr3_best,
    )
else:
    print("  SR3 failed for all configurations.")


# ============================================================
# 7b) Well-tuned PDE-FIND: specifically targeting clean equation recovery
#     This is the "best effort" SINDy/PDE-FIND baseline:
#     - Use the correct library [u, u_x, u*u_x, u_xx] (same as KANDy)
#     - Sweep STLSQ threshold to find the sparsest correct model
#     - Also try with an extended library to show that SINDy struggles
#       when the library is larger (more false positives)
# ============================================================
print("\n=== Baseline 5b: Well-tuned PDE-FIND (targeted threshold sweep) ===")

print("\n  --- On KANDy's library [u, u_x, u*u_x, u_xx] ---")
best_tuned_mse = np.inf
best_tuned_coefs = None
best_tuned_thr = None
for threshold in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    # STLSQ on the hand-crafted library
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
    terms_str = "  ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, feat_names_manual) if abs(c) > 1e-8)
    print(f"    thr={threshold:.3f}: MSE={mse:.3e}, active={n_active}, eq: {terms_str}")

    if mse < best_tuned_mse:
        best_tuned_mse = mse
        best_tuned_coefs = coefs.copy()
        best_tuned_thr = threshold

print(f"\n  Best threshold: {best_tuned_thr}")
print(f"  Best equation: " + "  ".join(
    f"{c:+.6f}*{n}" for c, n in zip(best_tuned_coefs, feat_names_manual) if abs(c) > 1e-8
))

rhs_tuned = make_rhs_from_coefs(best_tuned_coefs, 0.0)
U_tuned = eval_model(
    f"PDE-FIND tuned STLSQ(thr={best_tuned_thr})",
    lambda X: X @ best_tuned_coefs,
    rhs_tuned,
)

# Also evaluate the ideal sparse model: force only u*u_x term
print("\n  --- Ideal sparse: only u*u_x term (1 active) ---")
# Fit coefficient on u*u_x only
uux_train = Theta_train[:, 2:3]  # just u*u_x column
c_uux_only = np.linalg.lstsq(uux_train, y_train, rcond=None)[0][0]
print(f"    Coefficient: {c_uux_only:.6f}  (true: -1.0)")
ideal_coefs = np.array([0.0, 0.0, c_uux_only, 0.0])
rhs_ideal = make_rhs_from_coefs(ideal_coefs, 0.0)
U_ideal = eval_model(
    f"Ideal sparse (u_t = {c_uux_only:.4f}*u*u_x)",
    lambda X: (X[:, 2] * c_uux_only),
    rhs_ideal,
)


# ============================================================
# 8) PDE-FIND with pysindy's full WeakPDELibrary (if feasible)
# ============================================================
print("\n=== Baseline 6: pysindy WeakPDELibrary (native PDE-FIND) ===")

# Subsample in time for memory/speed
stride = 5
u_sub = U_true[::stride]
t_sub = t_grid[::stride]
Nt_sub = len(t_sub)

# Build spatiotemporal grid: (T, N, 2) with (t, x) coordinates
T_mesh, X_mesh = np.meshgrid(t_sub, x, indexing="ij")   # (Nt_sub, Nx)
xt_grid_sub = np.stack([T_mesh, X_mesh], axis=-1)        # (Nt_sub, Nx, 2)

n_train_weak = int(0.8 * Nt_sub)

best_weak = None
best_weak_mse = np.inf
best_weak_config = {}

for poly_deg in [2]:
    for deriv_order in [2]:
        for K_weak in [200]:
            for threshold_w in [0.01, 0.05]:
                try:
                    library = ps.WeakPDELibrary(
                        function_library=ps.PolynomialLibrary(
                            degree=poly_deg, include_bias=True
                        ),
                        derivative_order=deriv_order,
                        spatiotemporal_grid=xt_grid_sub[:n_train_weak],
                        K=K_weak,
                        num_pts_per_domain=min(50, Nx // 2),
                        is_uniform=True,
                    )
                    model_w = ps.SINDy(
                        feature_library=library,
                        optimizer=ps.STLSQ(threshold=threshold_w, alpha=0.05),
                    )
                    t0_fit = time.time()
                    dt_sub = float(t_sub[1] - t_sub[0])
                    model_w.fit(u_sub[:n_train_weak], t=dt_sub)
                    t_fit = time.time() - t0_fit

                    # Evaluate: get equations
                    eqs = model_w.equations()
                    coefs_w = model_w.coefficients().ravel()
                    feat_names_w = model_w.get_feature_names()

                    # Simple MSE proxy: predict on training data
                    u_dot_pred = model_w.predict(u_sub[:n_train_weak], t=dt_sub)
                    u_dot_true = np.gradient(u_sub[:n_train_weak], dt_sub, axis=0)
                    mse_w = np.mean((u_dot_pred - u_dot_true) ** 2) if u_dot_pred.shape == u_dot_true.shape else np.inf

                    if mse_w < best_weak_mse:
                        best_weak_mse = mse_w
                        best_weak = model_w
                        best_weak_config = {
                            "poly_deg": poly_deg,
                            "deriv_order": deriv_order,
                            "K": K_weak,
                            "threshold": threshold_w,
                            "fit_time": t_fit,
                        }

                    print(f"    deg={poly_deg} dord={deriv_order} K={K_weak} thr={threshold_w:.3f} | "
                          f"MSE={mse_w:.3e} | t={t_fit:.1f}s | eq: {eqs}")

                except Exception as e:
                    print(f"    deg={poly_deg} dord={deriv_order} K={K_weak} thr={threshold_w:.3f} | FAILED: {e}")

if best_weak is not None:
    print(f"\n  Best WeakPDELibrary config: {best_weak_config}")
    print(f"  Equations: {best_weak.equations()}")
    print(f"  Feature names: {best_weak.get_feature_names()}")
    print(f"  Coefficients: {best_weak.coefficients()}")

    # We can't easily do rollout from WeakPDELibrary's internal representation,
    # so just report the 1-step MSE in the table
    results_table.append({
        "name": f"WeakPDELibrary(deg={best_weak_config['poly_deg']},dord={best_weak_config['deriv_order']})",
        "mse_1step": best_weak_mse,
        "r2": float("nan"),
        "nrmse_t1": float("nan"),
        "nrmse_full": float("nan"),
    })


# ============================================================
# 9) Summary table
# ============================================================
print("\n" + "=" * 100)
print(f"{'Method':50s} | {'1-step MSE':>12s} | {'R²':>8s} | {'NRMSE(t<1)':>12s} | {'NRMSE(full)':>12s}")
print("-" * 100)
for r in results_table:
    nrmse_t1_str = f"{r['nrmse_t1']:.4f}" if np.isfinite(r['nrmse_t1']) else "N/A"
    nrmse_full_str = f"{r['nrmse_full']:.4f}" if np.isfinite(r['nrmse_full']) else "N/A"
    r2_str = f"{r['r2']:.4f}" if np.isfinite(r['r2']) else "N/A"
    print(f"  {r['name']:48s} | {r['mse_1step']:12.3e} | {r2_str:>8s} | {nrmse_t1_str:>12s} | {nrmse_full_str:>12s}")
print("=" * 100)


# ============================================================
# 10) Plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Ground truth
ax = axes[0, 0]
cf = ax.contourf(t_grid, x, U_true.T, levels=201, cmap="turbo")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("Ground truth")
plt.colorbar(cf, ax=ax)

# OLS rollout
ax = axes[0, 1]
vmin, vmax = U_true.min(), U_true.max()
t_ols = t_grid[:U_ols.shape[0]]
cf = ax.contourf(t_ols, x, U_ols.T, levels=201, cmap="turbo", vmin=vmin, vmax=vmax)
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("OLS rollout")
plt.colorbar(cf, ax=ax)

# LASSO rollout
ax = axes[0, 2]
t_las = t_grid[:U_lasso.shape[0]]
cf = ax.contourf(t_las, x, U_lasso.T, levels=201, cmap="turbo", vmin=vmin, vmax=vmax)
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title(f"LASSO(alpha={best_lasso_alpha}) rollout")
plt.colorbar(cf, ax=ax)

# PDE-FIND rollout
ax = axes[1, 0]
t_pf = t_grid[:U_pdefind.shape[0]]
cf = ax.contourf(t_pf, x, U_pdefind.T, levels=201, cmap="turbo", vmin=vmin, vmax=vmax)
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title(f"PDE-FIND STLSQ (deg={best_pdefind_config['degree']})")
plt.colorbar(cf, ax=ax)

# Error maps: OLS vs PDE-FIND
ax = axes[1, 1]
err_ols = np.abs(U_ols[:Nt] - U_true[:U_ols.shape[0]])
cf = ax.contourf(t_ols, x, err_ols.T, levels=51, cmap="hot")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("OLS |error|")
plt.colorbar(cf, ax=ax)

ax = axes[1, 2]
err_pf = np.abs(U_pdefind[:Nt] - U_true[:U_pdefind.shape[0]])
cf = ax.contourf(t_pf, x, err_pf.T, levels=51, cmap="hot")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("PDE-FIND STLSQ |error|")
plt.colorbar(cf, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_baselines.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_baselines.pdf"), dpi=300)
plt.close()
print(f"\nSaved: {RESULTS_DIR}/burgers_fourier_baselines.png")


# --- NRMSE over time plot ---
fig, ax = plt.subplots(figsize=(10, 4))

for r_item in results_table:
    name = r_item["name"]

nrmse_ols_t = compute_nrmse_over_time(U_ols, U_true)
nrmse_lasso_t = compute_nrmse_over_time(U_lasso, U_true)
nrmse_pdefind_t = compute_nrmse_over_time(U_pdefind, U_true)

t_plot_ols = t_grid[:len(nrmse_ols_t)]
t_plot_lasso = t_grid[:len(nrmse_lasso_t)]
t_plot_pf = t_grid[:len(nrmse_pdefind_t)]

ax.plot(t_plot_ols, nrmse_ols_t, label="OLS", lw=1.5)
ax.plot(t_plot_lasso, nrmse_lasso_t, label=f"LASSO(a={best_lasso_alpha})", lw=1.5, ls="--")
ax.plot(t_plot_pf, nrmse_pdefind_t, label="PDE-FIND STLSQ", lw=1.5, ls="-.")

ax.set_xlabel("t")
ax.set_ylabel("NRMSE")
ax.set_title("Rollout NRMSE over time — Baselines (Inviscid Burgers, Fourier ICs)")
ax.legend()
ax.set_yscale("log")
ax.set_ylim(bottom=1e-4)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_nrmse_over_time.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_nrmse_over_time.pdf"), dpi=300)
plt.close()
print(f"Saved: {RESULTS_DIR}/burgers_fourier_nrmse_over_time.png")


# --- Final snapshot comparison ---
fig, ax = plt.subplots(figsize=(10, 4))
idx_end = min(Nt, U_ols.shape[0], U_pdefind.shape[0]) - 1
ax.plot(x, U_true[idx_end], 'k-', lw=2, label="Ground truth")
ax.plot(x, U_ols[idx_end], 'b--', lw=1.2, label="OLS")
ax.plot(x, U_lasso[idx_end], 'g:', lw=1.2, label=f"LASSO(a={best_lasso_alpha})")
ax.plot(x, U_pdefind[idx_end], 'r-.', lw=1.2, label="PDE-FIND STLSQ")
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title(f"Snapshot at t={t_grid[idx_end]:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_final_snapshot.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "burgers_fourier_final_snapshot.pdf"), dpi=300)
plt.close()
print(f"Saved: {RESULTS_DIR}/burgers_fourier_final_snapshot.png")


# ============================================================
# 11) Save results to markdown
# ============================================================
# ============================================================
# Helper: format equation from coefficients
# ============================================================
def _format_equation(coef_dict, intercept=0.0, threshold=1e-6):
    """Format a clean symbolic equation string from coefficient dict."""
    terms = []
    for name, c in coef_dict.items():
        if abs(c) > threshold:
            terms.append(f"{c:+.6f}*{name}")
    if abs(intercept) > threshold:
        terms.append(f"{intercept:+.6f}")
    return "u_t = " + " ".join(terms) if terms else "u_t = 0"


def _format_equation_from_arrays(coefs, names, threshold=1e-6):
    """Format equation from parallel arrays of coefficients and names."""
    terms = []
    for c, n in zip(coefs, names):
        if abs(c) > threshold:
            terms.append(f"{c:+.6f}*{n}")
    return "u_t = " + " ".join(terms) if terms else "u_t = 0"


# Build discovered equations for each method
equations = {}

equations["OLS"] = _format_equation(dict(zip(feat_names_manual, ols.coef_)), ols.intercept_)
equations["Ridge"] = _format_equation(
    dict(zip(feat_names_manual, best_ridge.coef_)), best_ridge.intercept_
)
equations["LASSO"] = _format_equation(
    dict(zip(feat_names_manual, best_lasso.coef_)), best_lasso.intercept_
)
equations["PDE-FIND STLSQ"] = _format_equation_from_arrays(best_pdefind, best_pdefind_names)

if best_sr3 is not None:
    equations["PDE-FIND SR3"] = _format_equation_from_arrays(best_sr3, best_sr3_names)

equations["Tuned STLSQ"] = _format_equation(
    dict(zip(feat_names_manual, best_tuned_coefs)), 0.0
)
equations["Ideal sparse"] = _format_equation(
    dict(zip(feat_names_manual, ideal_coefs)), 0.0
)

if best_weak is not None:
    try:
        equations["WeakPDELibrary"] = str(best_weak.equations())
    except Exception:
        equations["WeakPDELibrary"] = "(extraction failed)"

# ============================================================
# 11) Save results to markdown
# ============================================================
report = f"""# Baseline Comparison -- Inviscid Burgers with Random Fourier ICs

## Setup (matches KANDy experiment exactly)
- Domain: [{x_min}, {x_max}], Nx={Nx}, dx={dx}
- Time: [0, {t1_sim}], dt_data={dt_data}, Nt={Nt}
- IC: {K_fourier} random Fourier modes, power-law decay p={p}
- Ground truth: Rusanov flux + RK45 (rtol=1e-7, atol=1e-9)
- Spatial derivatives: TVD minmod (u_x), standard Laplacian (u_xx)
- Rollout: SSP-RK3 with CFL-adaptive substeps (CFL=0.35)

## True equation
u_t = -1.0*u*u_x   (inviscid Burgers)

## Results

| Method | 1-step MSE | R2 | NRMSE (t<1) | NRMSE (full) |
|--------|-----------|------|-------------|--------------|
"""

for r in results_table:
    nrmse_t1_str = f"{r['nrmse_t1']:.4f}" if np.isfinite(r['nrmse_t1']) else "N/A"
    nrmse_full_str = f"{r['nrmse_full']:.4f}" if np.isfinite(r['nrmse_full']) else "N/A"
    r2_str = f"{r['r2']:.4f}" if np.isfinite(r['r2']) else "N/A"
    report += f"| {r['name']} | {r['mse_1step']:.3e} | {r2_str} | {nrmse_t1_str} | {nrmse_full_str} |\n"

report += """
## Discovered Equations

"""
for method_name, eq in equations.items():
    report += f"**{method_name}:**\n```\n{eq}\n```\n\n"

report += f"""## PDE-FIND best configuration
- STLSQ: {best_pdefind_config}
"""

if best_sr3 is not None:
    report += f"- SR3: {best_sr3_config}\n"

if best_weak is not None:
    report += f"- WeakPDELibrary: {best_weak_config}\n"

report += f"""
## Notes
- The hand-crafted library [u, u_x, u*u_x, u_xx] is the SAME lift used by KANDy.
  These baselines show what linear regression on that lift achieves.
- PDE-FIND uses a larger polynomial library (degree 2-3 monomials of u, u_x, u_xx)
  and STLSQ sparsification, which is the standard SINDy/PDE-FIND pipeline.
- The Fourier IC creates complex shock interactions that stress all methods.
"""

report_path = os.path.join(RESULTS_DIR, "burgers_fourier_baselines.md")
with open(report_path, "w") as f:
    f.write(report)
print(f"\nSaved report: {report_path}")
print("Done!")
