#!/usr/bin/env python3
"""PDE-FIND + KANDy comparison — Inviscid Burgers (20 Fourier-mode IC).

Uses PySINDy's PDELibrary (the standard PDE-FIND API) and KANDy
on the EXACT same data for a fair comparison.

IC and solver from the research notebook:
  - K=20 Fourier modes, seed=0
  - Domain [-π, π], dx=0.02, Nx≈316
  - t ∈ [0, 3], dt=0.002, Nt=1501
  - Rusanov flux + RK45 (scipy)

Results saved to results/Burgers-Fourier/baselines/
"""

import os
import re
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps

warnings.filterwarnings("ignore")
RESULTS = "results/Burgers-Fourier/baselines"
os.makedirs(RESULTS, exist_ok=True)
COEFF_TOL = 0.01  # shared rounding tolerance: drop terms with |coeff| < this

from kandy import KANDy, CustomLift
from kandy.numerics import muscl_reconstruct
from kandy.plotting import use_pub_style

use_pub_style()

# ============================================================
# 0. Data generation — matches research_code/burgers_fourier_baselines.py
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print(f"Device: {device}")

x_min, x_max = -np.pi, np.pi
Nx = 128
dx = (x_max - x_min) / Nx
x = np.linspace(x_min, x_max, Nx, endpoint=False)  # proper periodic grid

t0_sim, t1_sim = 0.0, 2.0
dt_data = 0.004
t_grid = np.linspace(t0_sim, t1_sim, int(round((t1_sim - t0_sim) / dt_data)) + 1)
Nt = len(t_grid)

K_fourier = 10
p_decay = 1.5
a_coeff = np.random.randn(K_fourier) * (np.arange(1, K_fourier + 1) ** (-p_decay))
phi_phase = 2 * np.pi * np.random.rand(K_fourier)
u0_np = sum(
    a_coeff[kk] * np.sin((kk + 1) * x + phi_phase[kk]) for kk in range(K_fourier)
).astype(np.float64)

print(f"Grid: Nx={Nx}, Nt={Nt}, dx={dx:.5f}, dt={dt_data}")
print(f"IC: {K_fourier} Fourier modes, power-law p={p_decay}")


def flux_fn(u):
    return 0.5 * u ** 2


def burgers_rhs(_t, u):
    uL, uR = u, np.roll(u, -1)
    fL, fR = flux_fn(uL), flux_fn(uR)
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    F_half = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
    return -(F_half - np.roll(F_half, 1)) / dx


print("Solving Burgers (Rusanov + RK45) ...")
sol = solve_ivp(burgers_rhs, (t0_sim, t1_sim), u0_np,
                t_eval=t_grid, method="RK45", rtol=1e-6, atol=1e-8, max_step=0.01)
assert sol.success, sol.message
U_true = sol.y.T  # (Nt, Nx)
print(f"Ground truth shape: {U_true.shape}")


# ============================================================
# 1. PDE-FIND via PySINDy PDELibrary
#    Try multiple differentiation methods for temporal derivative
# ============================================================
print("\n" + "=" * 60)
print("PDE-FIND via PySINDy PDELibrary")
print("=" * 60)

# PySINDy expects (n_space, n_time, n_variables)
u_ps = U_true.T[:, :, np.newaxis].copy()

diff_methods = {
    "FD": ps.FiniteDifference(),
    "SmoothedFD": ps.SmoothedFiniteDifference(),
    "SavGol": ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3),
}

pdefind_models = {}

for diff_name, diff_method in diff_methods.items():
    for degree in [2]:
        best_model, best_score, best_thr = None, -np.inf, None

        for threshold in [0.05, 0.5, 2.0, 5.0]:
            try:
                lib = ps.PDELibrary(
                    function_library=ps.PolynomialLibrary(degree=degree, include_bias=False),
                    derivative_order=3,
                    spatial_grid=x,
                    include_bias=True,
                    is_uniform=True,
                )
                opt = ps.STLSQ(threshold=threshold, alpha=1e-5, normalize_columns=True)
                mdl = ps.SINDy(feature_library=lib, optimizer=opt,
                               differentiation_method=diff_method)
                mdl.fit(u_ps, t=dt_data)

                sc = mdl.score(u_ps, t=dt_data)
                coefs = mdl.coefficients().ravel()
                n_active = np.sum(np.abs(coefs) > 1e-10)

                if n_active > 0 and sc > best_score:
                    best_score = sc
                    best_model = mdl
                    best_thr = threshold
            except Exception as e:
                print(f"  [{diff_name}, deg={degree}, thr={threshold}] FAILED: {e}")

    if best_model is not None:
        coefs = best_model.coefficients().ravel()
        fnames = best_model.get_feature_names()
        n_active = np.sum(np.abs(coefs) > 1e-10)

        label = f"PDE-FIND {diff_name} deg={degree}"
        pdefind_models[label] = {
            "model": best_model,
            "coefs": coefs,
            "feat_names": fnames,
            "n_active": n_active,
            "threshold": best_thr,
            "score": best_score,
        }

        print(f"\n{label} (threshold={best_thr}, R²={best_score:.4f}, {n_active} terms)")
        print(f"  Library ({len(fnames)} terms): {fnames}")
        print(f"\n  {'Term':25s} | {'Coefficient':>14s}")
        print(f"  {'-' * 25}-+-{'-' * 14}")
        for c, n in zip(coefs, fnames):
            marker = " <--" if abs(c) > 1e-10 else ""
            print(f"  {n:25s} | {c:+14.6f}{marker}")


# ============================================================
# 2. KANDy — TVD minmod derivatives (matches research script)
# ============================================================
print("\n" + "=" * 60)
print("KANDy")
print("=" * 60)


def tvd_deriv(u_arr, dx_):
    """TVD minmod-limited first derivative, periodic."""
    u_L, _ = muscl_reconstruct(u_arr, dx_, limiter="minmod")
    return (u_L - u_arr) * 2.0 / dx_


def laplacian_fd(u_arr, dx_):
    """Standard Laplacian (central 2nd-order FD), periodic."""
    return (np.roll(u_arr, -1) - 2 * u_arr + np.roll(u_arr, 1)) / dx_ ** 2


# Forward-diff time derivative (matches research script)
U_k = U_true[:-1]                             # (Nt-1, Nx)
U_dot = (U_true[1:] - U_true[:-1]) / dt_data  # (Nt-1, Nx)

# Build features: [u, u_x, u*u_x, u_xx] using TVD minmod
kandy_rows = []
for t_idx in range(U_k.shape[0]):
    u_snap = U_k[t_idx]
    ux = tvd_deriv(u_snap, dx)
    uxx = laplacian_fd(u_snap, dx)
    kandy_rows.append(np.column_stack([u_snap, ux, u_snap * ux, uxx]))

kandy_Theta = np.vstack(kandy_rows).astype(np.float32)
kandy_Ut = U_dot.ravel()[:, None].astype(np.float32)
print(f"  Features: {len(kandy_Theta)} points, 4 features (TVD minmod)")

MAX_KAN = 100_000
rng_sub = np.random.default_rng(42)
if len(kandy_Theta) > MAX_KAN:
    idx = rng_sub.choice(len(kandy_Theta), MAX_KAN, replace=False)
    kandy_Theta_sub = kandy_Theta[idx]
    kandy_Ut_sub = kandy_Ut[idx]
    print(f"Subsampled {MAX_KAN} from {len(kandy_Theta)}")
else:
    kandy_Theta_sub = kandy_Theta
    kandy_Ut_sub = kandy_Ut

KANDY_FEAT_NAMES = ["u", "u_x", "u*u_x", "u_xx"]
burgers_lift = CustomLift(fn=lambda X: X, output_dim=4, name="identity")

torch.manual_seed(0)
np.random.seed(42)
model_kandy = KANDy(lift=burgers_lift, grid=7, k=3, steps=300, seed=0,
                    device=str(device))

print("Training KANDy (300 steps, suppressing PyKAN output) ...")
import io, sys
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
model_kandy.fit(
    X=kandy_Theta_sub, X_dot=kandy_Ut_sub,
    val_frac=0.15, test_frac=0.15, lamb=0.0, patience=0, verbose=False,
)
sys.stdout = _old_stdout
print("  Training complete.")

kandy_eq = "(symbolic extraction failed)"
try:
    import copy
    import sympy as sp
    from kandy.symbolic import robust_auto_symbolic

    # Work on a COPY so the original model stays intact for rollout
    kan_copy = copy.deepcopy(model_kandy.model_).cpu()
    n_sym = min(5000, len(kandy_Theta_sub))
    sym_input = torch.tensor(kandy_Theta_sub[:n_sym], dtype=torch.float32)
    kan_copy.save_act = True
    kan_copy(sym_input)

    robust_auto_symbolic(
        kan_copy,
        lib=['x', 'x^2', 'x^3', '0'],
        r2_threshold=0.80,
        weight_simple=0.80,
        topk_edges=8,
        set_others_to_zero=True,
    )
    formulas, var_list = kan_copy.symbolic_formula()

    subs = {sp.Symbol(f'x_{i+1}'): sp.Symbol(n) for i, n in enumerate(KANDY_FEAT_NAMES)}
    raw_expr = formulas[0].subs(subs)
    expanded = sp.expand(raw_expr)
    cleaned_terms = []
    for term in sp.Add.make_args(expanded):
        coeff, rest = term.as_coeff_Mul()
        if abs(float(coeff)) > COEFF_TOL:
            cleaned_terms.append(sp.Float(round(float(coeff), 3)) * rest)
    cleaned = sum(cleaned_terms) if cleaned_terms else sp.Integer(0)
    kandy_eq = f"u_t = {cleaned}"
    print(f"  Raw:     u_t = {raw_expr}")
    print(f"  Cleaned: {kandy_eq}")
except Exception as e:
    print(f"\nKANDy symbolic extraction failed: {e}")
    import traceback; traceback.print_exc()


# ============================================================
# 3. Manual baselines: OLS, LASSO on hand-crafted library
# ============================================================
print("\n" + "=" * 60)
print("OLS / LASSO baselines")
print("=" * 60)

feat_names_hc = ["u", "u_x", "u*u_x", "u_xx"]
# Reuse the same features and targets as KANDy (TVD minmod, forward diff)

Theta_hc = kandy_Theta.astype(np.float64)  # same features as KANDy
y_all = U_dot.ravel()

N_total = len(Theta_hc)
perm = rng_sub.permutation(N_total)
N_test = min(20_000, int(0.2 * N_total))
train_idx = perm[N_test:N_test + min(80_000, N_total - N_test)]
test_idx = perm[:N_test]

Theta_train, Theta_test_ = Theta_hc[train_idx], Theta_hc[test_idx]
y_train, y_test_ = y_all[train_idx], y_all[test_idx]


def format_eq_hc(coefs, names, intercept=0.0):
    terms = [f"{round(c, 3):+g}*{n}" for c, n in zip(coefs, names) if abs(c) > COEFF_TOL]
    if abs(intercept) > COEFF_TOL:
        terms.append(f"{round(intercept, 3):+g}")
    return "u_t = " + " ".join(terms) if terms else "u_t = 0"


def format_eq_pdefind(coefs, names):
    terms = [f"{round(c, 3):+g}*{n}" for c, n in zip(coefs, names) if abs(c) > COEFF_TOL]
    return "u_t = " + " ".join(terms) if terms else "u_t = 0"


from sklearn.linear_model import LinearRegression, Lasso

ols = LinearRegression(fit_intercept=True).fit(Theta_train, y_train)
ols_eq = format_eq_hc(ols.coef_, feat_names_hc, ols.intercept_)
print(f"\nOLS: {ols_eq}")

best_lasso, best_lasso_mse = None, np.inf
for alpha in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
    la = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True).fit(Theta_train, y_train)
    mse = np.mean((la.predict(Theta_test_) - y_test_) ** 2)
    if mse < best_lasso_mse:
        best_lasso_mse = mse
        best_lasso = la
lasso_eq = format_eq_hc(best_lasso.coef_, feat_names_hc, best_lasso.intercept_)
print(f"LASSO (alpha={best_lasso.alpha}): {lasso_eq}")


# ============================================================
# 4. Rollout infrastructure
# ============================================================
def central_fd(u_arr, dx_, order):
    """Central FD with periodic BC."""
    if order == 1:
        return (np.roll(u_arr, -1) - np.roll(u_arr, 1)) / (2 * dx_)
    elif order == 2:
        return (np.roll(u_arr, -1) - 2 * u_arr + np.roll(u_arr, 1)) / dx_ ** 2
    elif order == 3:
        return (-np.roll(u_arr, 2) + 2 * np.roll(u_arr, 1)
                - 2 * np.roll(u_arr, -1) + np.roll(u_arr, -2)) / (2 * dx_ ** 3)
    raise ValueError(f"order {order}")


def parse_pysindy_feature(name):
    """Parse PySINDy feature name into (base, power) pairs."""
    name = name.strip()
    if name == '1':
        return [('1', 1)]
    # Match patterns like x0, x0_1, x0_11, x0_111, with optional ^N
    matches = re.findall(r'(x0(?:_1{1,4})?)(\^\d+)?', name)
    if not matches:
        raise ValueError(f"Cannot parse feature: {name}")
    return [(base, int(pw[1:]) if pw else 1) for base, pw in matches]


def make_pdefind_rhs(coefs, feat_names, deriv_order=3):
    """Build rollout RHS from PySINDy PDE-FIND coefficients."""
    # Pre-parse all feature names
    parsed = [parse_pysindy_feature(n) for n in feat_names]

    def rhs_fn(u_2d):
        u_1d = u_2d.ravel()
        derivs = {'x0': u_1d}
        for d in range(1, deriv_order + 1):
            derivs['x0_' + '1' * d] = central_fd(u_1d, dx, d)

        ut = np.zeros_like(u_1d)
        for c, parts in zip(coefs, parsed):
            if abs(c) < 1e-10:
                continue
            val = np.ones_like(u_1d)
            for base, power in parts:
                if base == '1':
                    continue
                val = val * derivs[base] ** power
            ut += c * val
        return ut.reshape(u_2d.shape)

    return rhs_fn


def make_hc_rhs(coefs, intercept=0.0):
    """RHS from hand-crafted [u, u_x, u*u_x, u_xx] coefficients (TVD minmod)."""
    c_u, c_ux, c_uux, c_uxx = coefs

    def rhs_fn(u_2d):
        u_1d = u_2d.ravel()
        ux = tvd_deriv(u_1d, dx)
        uxx = laplacian_fd(u_1d, dx)
        return (c_u * u_1d + c_ux * ux + c_uux * (u_1d * ux) + c_uxx * uxx
                + intercept).reshape(u_2d.shape)

    return rhs_fn


def kandy_rhs(u_2d):
    u_1d = u_2d.ravel()
    ux = tvd_deriv(u_1d, dx)
    uxx = laplacian_fd(u_1d, dx)
    th = np.column_stack([u_1d, ux, u_1d * ux, uxx])
    return model_kandy.predict(th).ravel().reshape(u_2d.shape)


def ssp_rk3_step(u, h, rhs_fn):
    k1 = rhs_fn(u)
    u1 = u + h * k1
    k2 = rhs_fn(u1)
    u2 = 0.75 * u + 0.25 * (u1 + h * k2)
    k3 = rhs_fn(u2)
    return u / 3 + 2 * (u2 + h * k3) / 3


def rollout_cfl(u0_, n_steps, dt_out, rhs_fn, cfl=0.35):
    u = u0_[np.newaxis, :].copy()
    traj = [u0_.copy()]
    for _ in range(n_steps - 1):
        umax = np.max(np.abs(u)) + 1e-12
        h_cfl = cfl * dx / umax
        n_sub = max(1, int(np.ceil(dt_out / h_cfl)))
        h = dt_out / n_sub
        for _ in range(n_sub):
            u = ssp_rk3_step(u, h, rhs_fn)
            if np.any(np.isnan(u)):
                traj.append(np.full_like(u0_, np.nan))
                return np.array(traj)
        traj.append(u[0].copy())
    return np.array(traj)


def compute_nrmse(pred, ref):
    mask = np.isfinite(pred) & np.isfinite(ref)
    if mask.sum() == 0:
        return np.inf
    return np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2) / np.var(ref[mask]))


# ============================================================
# 5. Rollout all methods
# ============================================================
print("\n" + "=" * 60)
print("Rollout evaluation (t=0 to t=3)")
print("=" * 60)

N_ROLL = min(501, Nt)  # t=0 to t=1 (500 steps)
rollout_results = {"Ground truth": U_true[:N_ROLL]}
nrmse_results = {}

# PDE-FIND
for label, res in pdefind_models.items():
    print(f"Rolling out {label} ...")
    try:
        rhs = make_pdefind_rhs(res["coefs"], res["feat_names"], deriv_order=3)
        roll = rollout_cfl(u0_np, N_ROLL, dt_data, rhs)
        nrmse = compute_nrmse(roll, U_true[:N_ROLL])
        rollout_results[label] = roll
        nrmse_results[label] = nrmse
        res["nrmse"] = nrmse
        print(f"  NRMSE: {nrmse:.4f}")
    except Exception as e:
        print(f"  Rollout failed: {e}")
        nrmse_results[label] = np.inf

# OLS
print("Rolling out OLS ...")
ols_roll = rollout_cfl(u0_np, N_ROLL, dt_data, make_hc_rhs(ols.coef_, ols.intercept_))
ols_nrmse = compute_nrmse(ols_roll, U_true[:N_ROLL])
rollout_results["OLS"] = ols_roll
nrmse_results["OLS"] = ols_nrmse
print(f"  NRMSE: {ols_nrmse:.4f}")

# LASSO
print("Rolling out LASSO ...")
lasso_roll = rollout_cfl(u0_np, N_ROLL, dt_data,
                         make_hc_rhs(best_lasso.coef_, best_lasso.intercept_))
lasso_nrmse = compute_nrmse(lasso_roll, U_true[:N_ROLL])
rollout_results["LASSO"] = lasso_roll
nrmse_results["LASSO"] = lasso_nrmse
print(f"  NRMSE: {lasso_nrmse:.4f}")

# KANDy
print("Rolling out KANDy ...")
kandy_roll = rollout_cfl(u0_np.astype(np.float32), N_ROLL, dt_data, kandy_rhs)
# Pad with NaN if rollout terminated early
if kandy_roll.shape[0] < N_ROLL:
    pad = np.full((N_ROLL - kandy_roll.shape[0], kandy_roll.shape[1]), np.nan)
    kandy_roll = np.vstack([kandy_roll, pad])
kandy_nrmse = compute_nrmse(kandy_roll, U_true[:N_ROLL])
rollout_results["KANDy"] = kandy_roll
nrmse_results["KANDy"] = kandy_nrmse
print(f"  NRMSE: {kandy_nrmse:.4f}")


# ============================================================
# 6. Spacetime rollout plots
# ============================================================
print("\n[FIGS] Generating spacetime plots ...")

methods = list(rollout_results.keys())
n_methods = len(methods)
cols = min(n_methods, 3)
rows = (n_methods + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = np.atleast_2d(axes)
vmin, vmax = U_true.min(), U_true.max()
t_plot = t_grid[:N_ROLL]

for idx, name in enumerate(methods):
    r, c = divmod(idx, cols)
    ax = axes[r, c]
    data = rollout_results[name]
    n_t = min(data.shape[0], len(t_plot))
    im = ax.imshow(data[:n_t].T, origin="lower", aspect="auto",
                   extent=[t_plot[0], t_plot[n_t - 1], x[0], x[-1]],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    if name == "Ground truth":
        ax.set_title(name, fontsize=10)
    else:
        nrmse = nrmse_results.get(name, np.nan)
        ax.set_title(f"{name}\nNRMSE={nrmse:.4f}", fontsize=9)

for idx in range(n_methods, rows * cols):
    r, c = divmod(idx, cols)
    axes[r, c].set_visible(False)

fig.colorbar(im, ax=axes.ravel().tolist(), label="u(x,t)", shrink=0.8)
fig.suptitle("Inviscid Burgers — Rollout Comparison (K=20 Fourier IC)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS}/pdefind_spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/pdefind_spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# NRMSE over time
fig, ax = plt.subplots(figsize=(10, 4))
for name in methods:
    if name == "Ground truth":
        continue
    data = rollout_results[name]
    nrmses = []
    for i in range(min(N_ROLL, data.shape[0])):
        ref_var = np.var(U_true[i])
        if ref_var < 1e-14 or not np.all(np.isfinite(data[i])):
            nrmses.append(np.nan)
        else:
            nrmses.append(np.sqrt(np.mean((data[i] - U_true[i]) ** 2) / ref_var))
    ax.plot(t_plot[:len(nrmses)], nrmses, label=name, lw=1.2)

ax.set_xlabel("t")
ax.set_ylabel("NRMSE")
ax.set_title("Rollout NRMSE over time — All Methods")
ax.legend(fontsize=7)
ax.set_yscale("log")
ax.set_ylim(bottom=1e-4)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{RESULTS}/pdefind_nrmse.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS}/pdefind_nrmse.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# Snapshot at t=1, t=2, t=3
for t_snap in [1.0, 2.0, 3.0]:
    snap_idx = int(round(t_snap / dt_data))
    if snap_idx >= Nt:
        continue
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, U_true[snap_idx], 'k-', lw=2, label="Ground truth")
    for name in methods:
        if name == "Ground truth":
            continue
        data = rollout_results[name]
        if snap_idx < data.shape[0] and np.all(np.isfinite(data[snap_idx])):
            ax.plot(x, data[snap_idx], '--', lw=1, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Snapshot at t={t_snap:.1f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/pdefind_snapshot_t{t_snap:.0f}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Summary table
# ============================================================
print("\n" + "=" * 100)
print(f"{'Method':35s} | {'Terms':>5s} | {'NRMSE':>8s} | Equation")
print("-" * 100)

all_equations = {}

for label, res in pdefind_models.items():
    eq = format_eq_pdefind(res["coefs"], res["feat_names"])
    n_terms = sum(1 for c in res["coefs"] if abs(c) > COEFF_TOL)
    all_equations[label] = eq
    nrmse = nrmse_results.get(label, np.nan)
    nrmse_str = f"{nrmse:.4f}" if np.isfinite(nrmse) else "N/A"
    print(f"  {label:33s} | {n_terms:5d} | {nrmse_str:>8s} | {eq}")

n_ols = sum(1 for c in ols.coef_ if abs(c) > COEFF_TOL) + int(abs(ols.intercept_) > COEFF_TOL)
all_equations["OLS"] = ols_eq
nrmse_str = f"{ols_nrmse:.4f}" if np.isfinite(ols_nrmse) else "N/A"
print(f"  {'OLS':33s} | {n_ols:5d} | {nrmse_str:>8s} | {ols_eq}")

n_lasso = sum(1 for c in best_lasso.coef_ if abs(c) > COEFF_TOL) + int(abs(best_lasso.intercept_) > COEFF_TOL)
all_equations["LASSO"] = lasso_eq
nrmse_str = f"{lasso_nrmse:.4f}" if np.isfinite(lasso_nrmse) else "N/A"
print(f"  {'LASSO':33s} | {n_lasso:5d} | {nrmse_str:>8s} | {lasso_eq}")

# Count KANDy terms from cleaned equation
import re as _re
n_kandy = len(_re.findall(r'[+-]', kandy_eq.replace('u_t = ', ''))) or 1
all_equations["KANDy"] = kandy_eq
nrmse_str = f"{kandy_nrmse:.4f}" if np.isfinite(kandy_nrmse) else "N/A"
print(f"  {'KANDy':33s} | {n_kandy:5d} | {nrmse_str:>8s} | {kandy_eq}")

print("=" * 100)


# ============================================================
# 8. Report
# ============================================================
report = f"""# PDE-FIND + KANDy Comparison — Inviscid Burgers (Fourier IC)

## Setup
- Domain: [{x_min:.4f}, {x_max:.4f}], Nx={Nx}, dx={dx}
- Time: [0, {t1_sim}], dt={dt_data}, Nt={Nt}
- IC: {K_fourier} Fourier modes, power-law decay p={p_decay}, seed=0
- Solver: Rusanov flux + RK45 (rtol=1e-6, atol=1e-8)
- Rollout: SSP-RK3 with CFL substeps (CFL=0.35), {N_ROLL} steps

## True equation
u_t = -u * u_x   (inviscid Burgers)

## Results

| Method | Terms | Rollout NRMSE | Equation |
|--------|-------|---------------|----------|
"""

for label, res in pdefind_models.items():
    nrmse = nrmse_results.get(label, np.nan)
    nrmse_s = f"{nrmse:.4f}" if np.isfinite(nrmse) else "N/A"
    report += f"| {label} (thr={res['threshold']}) | {res['n_active']} | {nrmse_s} | `{all_equations[label]}` |\n"

nrmse_s = f"{ols_nrmse:.4f}" if np.isfinite(ols_nrmse) else "N/A"
report += f"| OLS | {n_ols} | {nrmse_s} | `{ols_eq}` |\n"
nrmse_s = f"{lasso_nrmse:.4f}" if np.isfinite(lasso_nrmse) else "N/A"
report += f"| LASSO (alpha={best_lasso.alpha}) | {n_lasso} | {nrmse_s} | `{lasso_eq}` |\n"
nrmse_s = f"{kandy_nrmse:.4f}" if np.isfinite(kandy_nrmse) else "N/A"
report += f"| KANDy [3,1] | — | {nrmse_s} | `{kandy_eq}` |\n"

report += "\n## PDE-FIND Library Terms\n\n"
for label, res in pdefind_models.items():
    report += f"### {label} (threshold={res['threshold']})\n\n"
    report += f"| Term | Coefficient | Active |\n|------|------------|--------|\n"
    for c, n in zip(res["coefs"], res["feat_names"]):
        active = "yes" if abs(c) > 1e-10 else ""
        report += f"| {n} | {c:+.6f} | {active} |\n"
    report += "\n"

report += """## Notes
- PDE-FIND uses PySINDy's PDELibrary (finite difference derivatives) + STLSQ optimizer
- KANDy uses TVD minmod derivatives with conservation-form feature d(u²/2)/dx
- OLS/LASSO use hand-crafted library [u, u_x, u*u_x, u_xx] with TVD minmod derivatives
- True equation: u_t = -1.0*u*u_x (single active term)
"""

with open(f"{RESULTS}/pdefind_report.md", "w") as f:
    f.write(report)

print(f"\n[DONE] Report: {RESULTS}/pdefind_report.md")
print(f"[DONE] Figures: {RESULTS}/pdefind_*.png")
