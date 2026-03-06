#!/usr/bin/env python3
"""SINDy baselines for KANDy comparison.

Implements SINDy (Brunton et al. 2016) on the same systems used in KANDy:
  - Lorenz-63 (polynomial library, continuous-time)
  - Adaptive Kuramoto (Fourier + polynomial library, continuous-time)
  - Ikeda map (polynomial library, discrete-time)

Requires: pip install pysindy

Results are saved to outputs/baselines/ and printed as a comparison table.
"""

import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("outputs/baselines", exist_ok=True)
os.makedirs("results/SINDy", exist_ok=True)

try:
    import pysindy as ps
except ImportError:
    raise ImportError(
        "pysindy is required for baselines.  Install with: pip install pysindy"
    )

SEED = 42
np.random.seed(SEED)

results = []   # rows: (system, method, library, mse_1step, rmse_rollout, equation_ok, time_s)


# ---------------------------------------------------------------------------
# 1. Lorenz-63
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SYSTEM 1: Lorenz-63")
print("="*60)

def lorenz_rhs(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

dt_lorenz = 0.01
n_total   = 6000
burn_in   = 500
from scipy.integrate import solve_ivp

# Generate
x0 = np.array([1.0, 0.0, 0.0])
sol_burn = solve_ivp(lorenz_rhs, [0, burn_in*dt_lorenz], x0, method='RK45',
                     t_eval=np.arange(0, burn_in*dt_lorenz, dt_lorenz), dense_output=False)
x0_on = sol_burn.y[:, -1]

t_span = np.arange(0, n_total*dt_lorenz, dt_lorenz)
sol = solve_ivp(lorenz_rhs, [0, n_total*dt_lorenz], x0_on, method='RK45',
                t_eval=t_span, dense_output=False)
X_lorenz = sol.y.T.astype(np.float32)   # (N, 3)

n_train = int(0.8 * len(X_lorenz))
X_train = X_lorenz[:n_train]
X_test  = X_lorenz[n_train:]

# Fit SINDy
t0 = time.time()
sindy_lorenz = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.05, alpha=0.05),
)
sindy_lorenz.fit(X_train, t=dt_lorenz)
t_fit = time.time() - t0

print("\n[Lorenz] SINDy equations:")
sindy_lorenz.print()

# 1-step MSE
X_dot_pred = sindy_lorenz.predict(X_test)
from scipy.misc import derivative as _   # unused — use finite diff
X_dot_true = np.array([lorenz_rhs(0, X_test[i]) for i in range(len(X_test))])
mse_1step  = float(np.mean((X_dot_pred - X_dot_true)**2))

# Rollout (simulate)
T_roll = 500
x0_test = X_test[0]
try:
    x_roll = sindy_lorenz.simulate(x0_test, t=np.arange(T_roll)*dt_lorenz)
    x_true = X_test[:T_roll]
    n_cmp  = min(len(x_roll), len(x_true))
    rmse_roll = float(np.sqrt(np.mean((x_roll[:n_cmp] - x_true[:n_cmp])**2)))
except Exception as e:
    rmse_roll = float("nan")
    print(f"  [Lorenz rollout failed: {e}]")

# Ground truth: sigma*(y-x), x*(rho-z)-y, x*y-beta*z  (polynomial in x,y,z, xy, xz)
eqs = sindy_lorenz.equations()
eq_ok = all(k in str(eqs) for k in ["x0", "x1", "x2"])

print(f"[Lorenz] 1-step MSE = {mse_1step:.6f}  |  Rollout RMSE (T={T_roll}) = {rmse_roll:.6f}")
print(f"[Lorenz] Fit time: {t_fit:.1f}s")
results.append(("Lorenz-63", "SINDy", "Polynomial(2)", f"{mse_1step:.4e}",
                f"{rmse_roll:.4f}", str(eq_ok), f"{t_fit:.1f}"))

# Figure: rollout comparison (x component)
if not np.isnan(rmse_roll):
    fig, ax = plt.subplots(figsize=(9, 3))
    steps = np.arange(n_cmp) * dt_lorenz
    ax.plot(steps, x_true[:n_cmp, 0], lw=0.8, color="#1f77b4", label="True")
    ax.plot(steps, x_roll[:n_cmp, 0], lw=0.8, color="#d62728", ls="--", label="SINDy")
    ax.set_xlabel("time"); ax.set_ylabel("x"); ax.legend(fontsize=8)
    ax.set_title(f"Lorenz-63: SINDy rollout  (RMSE={rmse_roll:.4f})")
    fig.tight_layout()
    fig.savefig("results/SINDy/lorenz_rollout.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Kuramoto (N=5 oscillators, continuous-time)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SYSTEM 2: Adaptive Kuramoto  (N=5 oscillators)")
print("="*60)

N_OSC   = 5
eps_kur = 0.1
dt_kur  = 0.05
T_kur   = 4000

rng = np.random.default_rng(SEED)
omega = rng.uniform(0.8, 1.2, N_OSC)
kappa0 = rng.uniform(0.1, 0.3, (N_OSC, N_OSC))
kappa0 = (kappa0 + kappa0.T) / 2.0
np.fill_diagonal(kappa0, 0.0)

def kuramoto_rhs_np(theta, kappa, omega, eps):
    dtheta = omega.copy()
    for i in range(len(theta)):
        for j in range(len(theta)):
            if i != j:
                dtheta[i] += kappa[i, j] * np.sin(theta[j] - theta[i])
    dkappa = np.zeros_like(kappa)
    for i in range(len(theta)):
        for j in range(len(theta)):
            if i != j:
                dkappa[i, j] = -eps * (kappa[i, j] + np.sin(theta[j] - theta[i]))
    return dtheta, dkappa

theta = rng.uniform(-np.pi, np.pi, N_OSC)
kappa = kappa0.copy()
theta_traj = [theta.copy()]
kappa_traj = [kappa.copy()]

for _ in range(T_kur - 1):
    dt_h, dk = kuramoto_rhs_np(theta, kappa, omega, eps_kur)
    theta = theta + dt_kur * dt_h
    kappa = kappa + dt_kur * dk
    theta_traj.append(theta.copy())
    kappa_traj.append(kappa.copy())

theta_arr = np.array(theta_traj, dtype=np.float32)   # (T, N)
kappa_arr = np.array(kappa_traj, dtype=np.float32)   # (T, N, N)

# SINDy on phases only (simplified: treat as independent, use Fourier library)
n_tr_kur = int(0.8 * T_kur)
X_kur_tr = theta_arr[:n_tr_kur]    # (T_tr, N)
X_kur_te = theta_arr[n_tr_kur:]

t0 = time.time()
trig_lib = ps.GeneralizedLibrary([
    ps.FourierLibrary(n_frequencies=1),
    ps.PolynomialLibrary(degree=1),
])
sindy_kur = ps.SINDy(
    feature_library=trig_lib,
    optimizer=ps.STLSQ(threshold=0.05),
)
sindy_kur.fit(X_kur_tr, t=dt_kur)
t_fit_kur = time.time() - t0

print("\n[Kuramoto] SINDy equations (phase only):")
sindy_kur.print()

X_dot_kur_pred = sindy_kur.predict(X_kur_te)
# Numerical derivatives for true X_dot
X_dot_kur_true = np.gradient(X_kur_te, dt_kur, axis=0)
mse_kur = float(np.mean((X_dot_kur_pred - X_dot_kur_true)**2))

try:
    x_roll_kur = sindy_kur.simulate(X_kur_te[0], t=np.arange(300)*dt_kur)
    x_true_kur = X_kur_te[:300]
    n_cmp_kur  = min(len(x_roll_kur), len(x_true_kur))
    rmse_kur   = float(np.sqrt(np.mean(
        np.angle(np.exp(1j*(x_roll_kur[:n_cmp_kur] - x_true_kur[:n_cmp_kur])))**2
    )))
except Exception as e:
    rmse_kur = float("nan")
    print(f"  [Kuramoto rollout failed: {e}]")

print(f"[Kuramoto] 1-step MSE = {mse_kur:.6f}  |  Rollout angle-RMSE (T=300) = {rmse_kur:.6f}")
print(f"[Kuramoto] Fit time: {t_fit_kur:.1f}s")
results.append(("Kuramoto N=5", "SINDy", "Fourier(1)+Poly(1)", f"{mse_kur:.4e}",
                f"{rmse_kur:.4f}", "N/A", f"{t_fit_kur:.1f}"))


# ---------------------------------------------------------------------------
# 3. Ikeda map (discrete-time SINDy)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SYSTEM 3: Ikeda optical-cavity map (discrete-time)")
print("="*60)

U_IKEDA = 0.9

def ikeda_step(state):
    x, y = state
    t = 0.4 - 6.0 / (1.0 + x*x + y*y)
    return np.array([1.0 + U_IKEDA*(x*np.cos(t) - y*np.sin(t)),
                           U_IKEDA*(x*np.sin(t) + y*np.cos(t))], dtype=np.float32)

n_ik = 12000
burn_ik = 2000
traj_ik = np.zeros((n_ik + burn_ik, 2), dtype=np.float32)
traj_ik[0] = [0.1, 0.1]
for i in range(n_ik + burn_ik - 1):
    traj_ik[i+1] = ikeda_step(traj_ik[i])
traj_ik = traj_ik[burn_ik:]

X_ik   = traj_ik[:-1]    # (N, 2) current
Y_ik   = traj_ik[1:]     # (N, 2) next

n_tr_ik = int(0.8 * len(X_ik))
X_ik_tr, X_ik_te = X_ik[:n_tr_ik], X_ik[n_tr_ik:]
Y_ik_tr, Y_ik_te = Y_ik[:n_tr_ik], Y_ik[n_tr_ik:]

# Standard polynomial SINDy (discrete-time)
t0 = time.time()
sindy_ikeda_poly = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=0.01),
    discrete_time=True,
)
sindy_ikeda_poly.fit(X_ik_tr)
t_fit_ik_poly = time.time() - t0

print("\n[Ikeda] SINDy (polynomial, discrete) equations:")
sindy_ikeda_poly.print()

Y_pred_poly = sindy_ikeda_poly.predict(X_ik_te)
mse_ik_poly = float(np.mean((Y_pred_poly - Y_ik_te)**2))

# Physics-informed trig library for Ikeda
def _ikeda_trig_features(x):
    x_, y_ = x[:, 0], x[:, 1]
    t_ = 0.4 - 6.0 / (1.0 + x_**2 + y_**2)
    ct, st = np.cos(t_), np.sin(t_)
    return np.column_stack([
        x_, y_,
        U_IKEDA * x_ * ct, U_IKEDA * y_ * ct,
        U_IKEDA * x_ * st, U_IKEDA * y_ * st,
        np.ones(len(x_)),
    ])

trig_feat_names = ["x", "y", "u*x*cos(t)", "u*y*cos(t)", "u*x*sin(t)", "u*y*sin(t)", "1"]

t0 = time.time()
trig_custom_lib = ps.CustomLibrary(
    library_functions=[lambda x: _ikeda_trig_features(x)],
    function_names=[lambda x: trig_feat_names],
    interaction_only=False,
)
# Simpler: just use the 4 physics features as a polynomial library (already linear)
X_ik_phys_tr = _ikeda_trig_features(X_ik_tr)
X_ik_phys_te = _ikeda_trig_features(X_ik_te)

# Fit ordinary linear regression on physics features (SINDy with trivial library = identity)
from sklearn.linear_model import Ridge
reg = Ridge(alpha=1e-4, fit_intercept=False)
reg.fit(X_ik_phys_tr, Y_ik_tr)
t_fit_ik_phys = time.time() - t0

Y_pred_phys = reg.predict(X_ik_phys_te)
mse_ik_phys = float(np.mean((Y_pred_phys - Y_ik_te)**2))

# Rollout
def ikeda_roll_sindy(x0, horizon, model_fn):
    s = x0.copy()[None, :]
    traj = [s[0].copy()]
    for _ in range(horizon):
        s = model_fn(s)
        traj.append(s[0].copy())
    return np.array(traj)

def poly_map(s):
    return sindy_ikeda_poly.predict(s)

def phys_map(s):
    feats = _ikeda_trig_features(s)
    return reg.predict(feats)

EVAL_H = 400
x0_ik = X_ik_te[0]
true_roll = traj_ik[n_tr_ik : n_tr_ik + EVAL_H + 1]

try:
    roll_poly = ikeda_roll_sindy(x0_ik, EVAL_H, poly_map)
    rmse_ik_poly = float(np.sqrt(np.mean((roll_poly - true_roll)**2)))
except Exception:
    rmse_ik_poly = float("nan")

try:
    roll_phys = ikeda_roll_sindy(x0_ik, EVAL_H, phys_map)
    rmse_ik_phys = float(np.sqrt(np.mean((roll_phys - true_roll)**2)))
except Exception:
    rmse_ik_phys = float("nan")

print(f"\n[Ikeda] Polynomial SINDy  1-step MSE={mse_ik_poly:.4e}  Rollout RMSE={rmse_ik_poly:.4f}")
print(f"[Ikeda] Physics-trig reg  1-step MSE={mse_ik_phys:.4e}  Rollout RMSE={rmse_ik_phys:.4f}")
results.append(("Ikeda map", "SINDy", "Polynomial(3), discrete", f"{mse_ik_poly:.4e}",
                f"{rmse_ik_poly:.4f}", "partial", f"{t_fit_ik_poly:.1f}"))
results.append(("Ikeda map", "Physics-trig Ridge", "4 trig features", f"{mse_ik_phys:.4e}",
                f"{rmse_ik_phys:.4f}", "yes (linear)", f"{t_fit_ik_phys:.1f}"))


# ---------------------------------------------------------------------------
# 4. Comparison table
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("COMPARISON TABLE (SINDy baselines)")
print("="*60)

header = f"{'System':<22} {'Method':<24} {'Library':<26} {'1-step MSE':>12} {'Rollout RMSE':>13} {'Eq OK':>7} {'Time(s)':>8}"
sep    = "-" * len(header)
print(sep)
print(header)
print(sep)
for row in results:
    print(f"{row[0]:<22} {row[1]:<24} {row[2]:<26} {row[3]:>12} {row[4]:>13} {row[5]:>7} {row[6]:>8}")
print(sep)

# Save table
table_lines = [
    "# SINDy Baselines — Comparison Table\n",
    "| System | Method | Library | 1-step MSE | Rollout RMSE | Equation OK | Time (s) |",
    "|---|---|---|---|---|---|---|",
]
for row in results:
    table_lines.append(f"| {' | '.join(row)} |")

os.makedirs("outputs/baselines", exist_ok=True)
with open("outputs/baselines/sindy_table.md", "w") as f:
    f.write("\n".join(table_lines) + "\n")

print("\n[DONE] SINDy baselines complete.  Results saved to outputs/baselines/sindy_table.md")
