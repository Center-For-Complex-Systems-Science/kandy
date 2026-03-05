#!/usr/bin/env python3
"""KANDy example: Holling Type II predator-prey system.

The Rosenzweig–MacArthur predator-prey ODE with a Type II functional response:

    dN/dt = r N (1 - N/K)  -  (a N P) / (1 + a h N)
    dP/dt = e (a N P) / (1 + a h N)  -  m P

where N is prey density, P is predator density, and the term
    (a N P) / (1 + a h N)
is the Holling Type II functional response (saturating foraging rate).

The system is trained in **discrete-time map mode**:
    X = (N_n, P_n) → Y = (N_{n+1}, P_{n+1})
using one-step-ahead prediction.  The true continuous-time flow is discretised
by RK4 integration with dt = 0.02.

Physics-informed feature library
---------------------------------
The Holling functional response involves the term 1/(1 + ahN) which is
non-polynomial.  Rather than relying on the KAN to discover it from scratch,
we pre-compute 21 features that span the RHS structure:

    [1, N, P, N², P², NP, denom, invden, fN, logistic, pred, gainP, deathP,
     rN, N/K, rN(N/K), aN, aP, aNP, ahN, ahNP]

where  denom = 1 + ahN,  invden = 1/denom,  fN = N/denom,  pred = aNP/denom.

KAN:  width = [21, 2],  base_fun = 'zero' (no SiLU bias; pure spline)
Optimizer: Adam (better than LBFGS for the large batch/rollout training here)
Symbolic: auto_symbolic_with_costs preferring the physics-aligned features.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.symbolic import auto_symbolic_with_costs, POLY_LIB_CHEAP, POLY_LIB
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    plot_attractor_overlay,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Parameters and reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# True ODE parameters (Rosenzweig–MacArthur)
R_TRUE = 1.0
K_TRUE = 50.0
A_TRUE = 1.2
H_TRUE = 0.1
E_TRUE = 0.6
M_TRUE = 0.4

DT       = 0.02
N_SAMPLES = 20_000
BURN_IN   = 2_000

# ---------------------------------------------------------------------------
# 1. Data generation — RK4 integration of the Holling Type II ODE
# ---------------------------------------------------------------------------

def holling_rhs(state: np.ndarray) -> np.ndarray:
    N, P = state
    denom = 1.0 + A_TRUE * H_TRUE * N
    pred  = (A_TRUE * N * P) / denom
    dN = R_TRUE * N * (1.0 - N / K_TRUE) - pred
    dP = E_TRUE * pred - M_TRUE * P
    return np.array([dN, dP], dtype=np.float64)


def rk4_step_np(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = holling_rhs(state)
    k2 = holling_rhs(state + 0.5 * dt * k1)
    k3 = holling_rhs(state + 0.5 * dt * k2)
    k4 = holling_rhs(state + dt * k3)
    return (state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)


def generate_holling(n_total: int, burn_in: int, x0=(10.0, 5.0)) -> np.ndarray:
    T = n_total + burn_in
    traj = np.zeros((T, 2), dtype=np.float32)
    traj[0] = np.array(x0, dtype=np.float32)
    for i in range(T - 1):
        traj[i + 1] = rk4_step_np(traj[i], DT)
        traj[i + 1] = np.maximum(traj[i + 1], 0.0)   # enforce N, P ≥ 0
    return traj[burn_in:]


print("[DATA]  Generating Holling Type II trajectory ...")
series = generate_holling(N_SAMPLES, BURN_IN)
X_state = series[:-1]   # (N, 2) current state
Y_state = series[1:]    # (N, 2) next state  ← training target

print(f"[DATA]  N={len(X_state)} one-step pairs,  dt={DT}")

# ---------------------------------------------------------------------------
# 2. Physics-informed feature library
#    phi: R² → R^21
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "1", "N", "P",
    "N^2", "P^2", "N*P",
    "denom", "invden", "fN",
    "logistic", "pred", "gainP", "deathP",
    "r*N", "N/K", "r*N*(N/K)",
    "a*N", "a*P", "a*N*P",
    "a*h*N", "(a*h*N)*P",
]
N_FEATURES = len(FEATURE_NAMES)   # 21

# Indices of the Holling-physics-aligned features — give these lower cost
PHYSICS_IDX = {
    FEATURE_NAMES.index("denom"),
    FEATURE_NAMES.index("invden"),
    FEATURE_NAMES.index("fN"),
    FEATURE_NAMES.index("logistic"),
    FEATURE_NAMES.index("pred"),
    FEATURE_NAMES.index("gainP"),
    FEATURE_NAMES.index("deathP"),
}


def build_holling_features(N_arr: np.ndarray, P_arr: np.ndarray) -> np.ndarray:
    """Build (T, 21) feature matrix from prey N and predator P arrays."""
    one      = np.ones_like(N_arr)
    denom    = 1.0 + A_TRUE * H_TRUE * N_arr
    invden   = 1.0 / denom
    fN       = N_arr * invden
    pred     = (A_TRUE * N_arr * P_arr) * invden
    logistic = R_TRUE * N_arr * (1.0 - N_arr / K_TRUE)
    deathP   = M_TRUE * P_arr
    gainP    = E_TRUE * pred

    return np.column_stack([
        one, N_arr, P_arr,
        N_arr ** 2, P_arr ** 2, N_arr * P_arr,
        denom, invden, fN,
        logistic, pred, gainP, deathP,
        R_TRUE * N_arr, (N_arr / K_TRUE), R_TRUE * N_arr * (N_arr / K_TRUE),
        A_TRUE * N_arr, A_TRUE * P_arr, A_TRUE * N_arr * P_arr,
        A_TRUE * H_TRUE * N_arr, (A_TRUE * H_TRUE * N_arr) * P_arr,
    ]).astype(np.float32)


Theta_np = build_holling_features(X_state[:, 0], X_state[:, 1])   # (N, 21)
print(f"[DATA]  Feature matrix shape: {Theta_np.shape}")

# ---------------------------------------------------------------------------
# 3. Train / val / test split and feature normalisation
# ---------------------------------------------------------------------------
N      = len(Theta_np)
n_test = int(N * 0.20)
n_val  = int((N - n_test) * 0.20)
n_train = N - n_test - n_val

Theta_train = Theta_np[:n_train]
Theta_val   = Theta_np[n_train : n_train + n_val]
Theta_test  = Theta_np[n_train + n_val:]

Y_train = Y_state[:n_train]
Y_val   = Y_state[n_train : n_train + n_val]
Y_test  = Y_state[n_train + n_val:]

# Fit normalisation on training set only
feat_mean = Theta_train.mean(axis=0, keepdims=True)
feat_std  = Theta_train.std(axis=0, keepdims=True) + 1e-8


def normalise(Theta: np.ndarray) -> np.ndarray:
    return (Theta - feat_mean) / feat_std


Theta_train_n = normalise(Theta_train)
Theta_val_n   = normalise(Theta_val)
Theta_test_n  = normalise(Theta_test)

# ---------------------------------------------------------------------------
# 4. KANDy model — KAN = [21, 2], base_fun='zero'
#    We use a CustomLift (identity) since features are pre-computed.
# ---------------------------------------------------------------------------
holling_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="holling_lift")

model = KANDy(
    lift=holling_lift,
    grid=5,
    k=3,
    steps=2500,
    seed=SEED,
    base_fun="zero",   # pure spline, no SiLU bias
)

# Use Adam for this system (large dataset, rollout loss, batch training)
# Pass pre-normalised features directly; the lift is identity.
model.fit(
    X=Theta_train_n,
    X_dot=Y_train,
    val_frac=float(n_val) / (n_train + n_val),
    test_frac=float(n_test) / N,
    lamb=0.0,
    opt="Adam",
    lr=2e-3,
    batch=4096,
)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction with physics-preferred costs
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Running physics-preferred auto_symbolic ...")
model.model_.save_act = True
with torch.no_grad():
    t_train = torch.tensor(Theta_train_n, dtype=torch.float32, device=DEVICE)
    model.model_(t_train[:2048])

auto_symbolic_with_costs(
    model.model_,
    preferred_idx=PHYSICS_IDX,
    preferred_lib=POLY_LIB_CHEAP,
    other_lib=POLY_LIB,
    weight_simple=0.7,
    r2_threshold=0.90,
    verbose=1,
)

import sympy as sp
exprs_raw, vars_ = model.model_.symbolic_formula()
N_sym, P_sym = sp.symbols("N P", real=True, nonnegative=True)
r, K, a, h, e, m = sp.symbols("r K a h e m", real=True)

denom_sym  = 1 + a * h * N_sym
invden_sym = 1 / denom_sym
fN_sym     = N_sym * invden_sym
pred_sym   = a * N_sym * P_sym * invden_sym
logistic_sym = r * N_sym * (1 - N_sym / K)

feature_syms = [
    sp.Integer(1), N_sym, P_sym,
    N_sym**2, P_sym**2, N_sym * P_sym,
    denom_sym, invden_sym, fN_sym,
    logistic_sym, pred_sym, e * pred_sym, m * P_sym,
    r * N_sym, (N_sym / K), r * N_sym * (N_sym / K),
    a * N_sym, a * P_sym, a * N_sym * P_sym,
    a * h * N_sym, a * h * N_sym * P_sym,
]
sub_map = {vars_[i]: feature_syms[i] for i in range(len(vars_))}


def _round_expr(expr, places=4):
    return expr.xreplace({n: round(float(n), places) for n in expr.atoms(sp.Number)})


def _flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(_flatten(it))
        return out
    return [obj]


cleaned = []
for expr in _flatten(exprs_raw):
    if not hasattr(expr, "free_symbols"):
        continue
    ex = sp.together(sp.expand(expr.subs(sub_map)))
    cleaned.append(_round_expr(ex, 4))

print("\n[SYMBOLIC] Discovered discrete-time map:")
for label, ex in zip(["N_{n+1}", "P_{n+1}"], cleaned[:2]):
    print(f"  {label} = {ex}")

# ---------------------------------------------------------------------------
# 6. Rollout evaluation
# ---------------------------------------------------------------------------

def map_fn(state_np: np.ndarray) -> np.ndarray:
    """Apply the learned map to a batch of states (numpy interface)."""
    theta = build_holling_features(state_np[:, 0], state_np[:, 1])
    theta_n = (theta - feat_mean) / feat_std
    t = torch.tensor(theta_n, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model.model_(t).cpu().numpy()


def rollout_map(s0: np.ndarray, horizon: int) -> np.ndarray:
    s = s0[None, :]   # (1, 2)
    traj = [s[0].copy()]
    for _ in range(horizon):
        s = map_fn(s)
        traj.append(s[0].copy())
    return np.array(traj)


test_start = series[n_train + n_val]
HORIZON    = min(800, len(Y_test))
pred_roll  = rollout_map(test_start, HORIZON)
true_roll  = series[n_train + n_val : n_train + n_val + HORIZON + 1]

rmse = np.sqrt(np.mean((pred_roll - true_roll) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={HORIZON} steps): {rmse:.6f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/HollingTypeII", exist_ok=True)

t_steps = np.arange(HORIZON + 1) * DT

# 7a. Time series (prey and predator)
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, ci, label in zip(axes, [0, 1], ["Prey N", "Predator P"]):
    ax.plot(t_steps, true_roll[:, ci], color="#1f77b4", lw=1.2, label="True")
    ax.plot(t_steps, pred_roll[:, ci], color="#d62728", lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time")
fig.suptitle("Holling Type II: rollout", fontsize=12)
fig.tight_layout()
fig.savefig("results/HollingTypeII/timeseries.png", dpi=300, bbox_inches="tight")
fig.savefig("results/HollingTypeII/timeseries.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Phase portrait (N–P attractor)
fig, ax = plot_attractor_overlay(
    true_roll, pred_roll,
    dim_x=0, dim_y=1,
    labels=["True", "KANDy"],
    colors=["#1f77b4", "#d62728"],
    title="Holling Type II: phase portrait",
    save="results/HollingTypeII/phase_portrait",
)
plt.close(fig)

# 7c. Loss curves
if hasattr(model, "train_results_") and model.train_results_:
    fig, ax = plot_loss_curves(
        model.train_results_,
        title="Holling Type II training loss",
        save="results/HollingTypeII/loss_curves",
    )
    plt.close(fig)

# 7d. Edge activations
n_sub   = min(4096, n_train)
sub_idx = np.random.choice(n_train, n_sub, replace=False)
train_t = torch.tensor(Theta_train_n[sub_idx], dtype=torch.float32, device=DEVICE)
fig = plot_all_edges(
    model.model_,
    X=train_t,
    input_names=FEATURE_NAMES,
    output_names=["N_{n+1}", "P_{n+1}"],
    title="Holling Type II KAN edge activations",
    save="results/HollingTypeII/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/HollingTypeII/")
