#!/usr/bin/env python3
"""KANDy example: Ikeda optical-cavity map.

The Ikeda map models light circulation in a nonlinear optical ring cavity:

    x_{n+1} = 1 + u * (x_n cos t_n  -  y_n sin t_n)
    y_{n+1} =     u * (x_n sin t_n  +  y_n cos t_n)

where the phase t_n is state-dependent:

    t_n = 0.4 - 6 / (1 + x_n² + y_n²)

and u = 0.9 is the dissipation parameter (u ≥ 0.83 gives chaos).

Physics-informed feature library
---------------------------------
The map is a rotation in the (x, y) plane by angle t_n, scaled by u.
Pre-composing the lift with the exact trig rotation gives 4 features:

    phi(x, y) = [u·x·cos(t),  u·y·cos(t),  u·x·sin(t),  u·y·sin(t)]

where t = t(x, y) above.  With this lift the KAN only needs to learn:

    x_{n+1} ≈ 1 + phi_0 - phi_3      (= 1 + u·x·cos(t) - u·y·sin(t))
    y_{n+1} ≈     phi_2 + phi_1      (= u·x·sin(t)  + u·y·cos(t))

i.e. near-linear combinations of the four features.

KAN:  width = [4, 2],  base_fun = RBF (exp(-x²))
Discrete-map rollout via the "increment trick":
    dynamics_fn(s) = map(s) - s   so that  Euler(s, dt=1) = map(s)

Symbolic: auto_symbolic_with_costs with TRIG_LIB_CHEAP (all 4 features
are physics-informed, so all edges get low cost).
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift, make_windows
from kandy.symbolic import auto_symbolic_with_costs, TRIG_LIB_CHEAP, TRIG_LIB
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

U        = 0.9     # dissipation parameter (chaos at u ≥ 0.83)
N_TOTAL  = 12_000
BURN_IN  = 2_000
HORIZON  = 15      # rollout horizon for training loss (discrete steps)

# ---------------------------------------------------------------------------
# 1. Data generation — Ikeda map iteration
# ---------------------------------------------------------------------------

def ikeda_step(state: np.ndarray, u: float = U) -> np.ndarray:
    x, y = state
    t = 0.4 - 6.0 / (1.0 + x * x + y * y)
    x_next = 1.0 + u * (x * np.cos(t) - y * np.sin(t))
    y_next =        u * (x * np.sin(t) + y * np.cos(t))
    return np.array([x_next, y_next], dtype=np.float32)


def generate_ikeda(n_total: int, burn_in: int, x0=0.1, y0=0.1) -> np.ndarray:
    traj = np.zeros((n_total + burn_in, 2), dtype=np.float32)
    traj[0] = [x0, y0]
    for i in range(n_total + burn_in - 1):
        traj[i + 1] = ikeda_step(traj[i])
    return traj[burn_in:]


print("[DATA]  Generating Ikeda map trajectory ...")
series = generate_ikeda(N_TOTAL, BURN_IN)
X_state = series[:-1]   # (N, 2) current
Y_state = series[1:]    # (N, 2) next

print(f"[DATA]  N={len(X_state)} one-step pairs")

# ---------------------------------------------------------------------------
# 2. Physics-informed feature library
#    phi(x, y) = [u·x·cos(t), u·y·cos(t), u·x·sin(t), u·y·sin(t)]
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["u·x·cos(t)", "u·y·cos(t)", "u·x·sin(t)", "u·y·sin(t)"]
N_FEATURES = 4

# All 4 features are physics-informed — assign cheap symbolic costs to all
PHYSICS_IDX = set(range(N_FEATURES))


def _ikeda_features_np(X: np.ndarray) -> np.ndarray:
    """NumPy: (N, 2) → (N, 4) physics-informed features."""
    x, y = X[:, 0], X[:, 1]
    r2 = x * x + y * y
    q  = 1.0 / (1.0 + r2)
    t  = 0.4 - 6.0 * q
    ct, st = np.cos(t), np.sin(t)
    return np.column_stack([
        U * x * ct,
        U * y * ct,
        U * x * st,
        U * y * st,
    ]).astype(np.float32)


def _ikeda_features_torch(X: torch.Tensor) -> torch.Tensor:
    """Torch: (B, 2) → (B, 4) physics-informed features (gradient-compatible)."""
    x, y = X[:, 0], X[:, 1]
    r2 = x * x + y * y
    q  = 1.0 / (1.0 + r2)
    t  = 0.4 - 6.0 * q
    ct, st = torch.cos(t), torch.sin(t)
    return torch.stack([
        U * x * ct,
        U * y * ct,
        U * x * st,
        U * y * st,
    ], dim=1)


ikeda_lift = CustomLift(
    fn=_ikeda_features_np,
    torch_fn=_ikeda_features_torch,
    output_dim=N_FEATURES,
    name="ikeda_lift",
)

Theta_np = ikeda_lift(X_state)

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

feat_mean = Theta_train.mean(axis=0, keepdims=True)
feat_std  = Theta_train.std(axis=0, keepdims=True) + 1e-8


def normalise(Theta: np.ndarray) -> np.ndarray:
    return (Theta - feat_mean) / feat_std


Theta_train_n = normalise(Theta_train)
Theta_test_n  = normalise(Theta_test)

# ---------------------------------------------------------------------------
# 4. Build windowed trajectory dataset for rollout loss
#    Treat the discrete map as an ODE with:
#      dynamics_fn(s) = map(s) - s
#    Then Euler with dt=1 recovers the exact map iteration.
# ---------------------------------------------------------------------------
feat_mean_t = torch.tensor(feat_mean, dtype=torch.float32, device=DEVICE)
feat_std_t  = torch.tensor(feat_std,  dtype=torch.float32, device=DEVICE)


def map_fn_torch(state_xy: torch.Tensor) -> torch.Tensor:
    """Apply learned map: Theta(state) → normalise → KAN → next state."""
    Theta = ikeda_lift.torch_fn(state_xy)
    Theta_n = (Theta - feat_mean_t) / feat_std_t
    return model.model_(Theta_n)   # (B, 2)


def discrete_rhs(state_xy: torch.Tensor) -> torch.Tensor:
    """Euler-compatible 'derivative': map(s) - s so that s + 1*(map(s)-s) = map(s)."""
    return map_fn_torch(state_xy) - state_xy


# Windowed trajectory data for the rollout loss in fit_kan
window = HORIZON + 1
train_seq = torch.tensor(series[: n_train + 1], dtype=torch.float32, device=DEVICE)
test_seq  = torch.tensor(series[n_train + n_val : n_train + n_val + 2000],
                         dtype=torch.float32, device=DEVICE)

train_windows = make_windows(train_seq, window)   # (Nw, window, 2)
test_windows  = make_windows(test_seq,  window)

t_window = torch.arange(window, dtype=torch.float32, device=DEVICE)   # dt=1 each step

# ---------------------------------------------------------------------------
# 5. KANDy model — KAN = [4, 2], base_fun = RBF
# ---------------------------------------------------------------------------
model = KANDy(
    lift=ikeda_lift,
    grid=5,
    k=3,
    steps=100,          # initial fit; rollout training follows
    seed=SEED,
    base_fun=lambda x: torch.exp(-(x ** 2)),   # RBF base
)

# Phase 1: one-step derivative supervision only (warm start)
print("\n--- Phase 1: warm start (one-step supervision) ---")
model.fit(
    X=Theta_train_n,
    X_dot=Y_train,
    val_frac=float(n_val) / (n_train + n_val),
    test_frac=float(n_test) / N,
    lamb=0.0,
    opt="LBFGS",
    fit_steps=50,
)

# Phase 2: fine-tune with rollout loss using the discrete_rhs trick
# We call fit_kan directly to pass the trajectory dataset and dynamics_fn.
print("\n--- Phase 2: rollout fine-tuning ---")
from kandy.training import fit_kan

dataset_roll = {
    "train_input": torch.tensor(Theta_train_n, dtype=torch.float32, device=DEVICE),
    "train_label": torch.tensor(Y_train,       dtype=torch.float32, device=DEVICE),
    "test_input":  torch.tensor(Theta_test_n,  dtype=torch.float32, device=DEVICE),
    "test_label":  torch.tensor(Y_test,        dtype=torch.float32, device=DEVICE),
    "train_traj":  train_windows,
    "train_t":     t_window,
    "test_traj":   test_windows,
    "test_t":      t_window,
}

rollout_results = fit_kan(
    model.model_,
    dataset_roll,
    opt="LBFGS",
    steps=100,
    lr=1e-4,
    batch=2048,
    rollout_weight=0.6,
    rollout_horizon=HORIZON,
    traj_batch=512,
    dynamics_fn=discrete_rhs,
    integrator="euler",          # Euler + discrete_rhs = exact map iteration
    update_grid=True,
    stop_grid_update_step=2000,
)

# ---------------------------------------------------------------------------
# 6. Symbolic extraction with trig-aware costs
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Running trig-aware auto_symbolic ...")
model.model_.save_act = True
train_t = torch.tensor(Theta_train_n[:1024], dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    model.model_(train_t)

auto_symbolic_with_costs(
    model.model_,
    preferred_idx=PHYSICS_IDX,       # all 4 features are physics-informed
    preferred_lib=TRIG_LIB_CHEAP,    # cheap trig costs for physics edges
    other_lib=TRIG_LIB,
    weight_simple=0.1,
    r2_threshold=0.80,
    verbose=1,
)

import sympy as sp
exprs_raw, vars_ = model.model_.symbolic_formula()
x_sym, y_sym = sp.symbols("x y", real=True)
u_sym = sp.Rational(9, 10)
r2_sym = x_sym**2 + y_sym**2
q_sym  = 1 / (1 + r2_sym)
t_sym  = sp.Rational(2, 5) - 6 * q_sym
ct_sym, st_sym = sp.cos(t_sym), sp.sin(t_sym)

feature_syms_ikeda = [
    u_sym * x_sym * ct_sym,
    u_sym * y_sym * ct_sym,
    u_sym * x_sym * st_sym,
    u_sym * y_sym * st_sym,
]
sub_map_ikeda = {vars_[i]: feature_syms_ikeda[i] for i in range(len(vars_))}


def _flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(_flatten(it))
        return out
    return [obj]


def _round_expr(expr, places=4):
    return expr.xreplace({n: round(float(n), places) for n in expr.atoms(sp.Number)})


cleaned_ikeda = []
for expr in _flatten(exprs_raw):
    if not hasattr(expr, "free_symbols"):
        continue
    ex = sp.together(sp.expand(expr.subs(sub_map_ikeda)))
    cleaned_ikeda.append(_round_expr(ex, 4))

print("\n[SYMBOLIC] Discovered Ikeda map:")
for label, ex in zip(["x_{n+1}", "y_{n+1}"], cleaned_ikeda[:2]):
    print(f"  {label} = {ex}")
print(f"\n[TRUE] x_{{n+1}} = 1 + u*(x*cos(t) - y*sin(t))")
print(f"[TRUE] y_{{n+1}} =     u*(x*sin(t) + y*cos(t))")

# ---------------------------------------------------------------------------
# 7. Rollout evaluation
# ---------------------------------------------------------------------------

def map_fn_np(state_np: np.ndarray) -> np.ndarray:
    theta = ikeda_lift(state_np)
    theta_n = (theta - feat_mean) / feat_std
    t = torch.tensor(theta_n, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model.model_(t).cpu().numpy()


def rollout_discrete(s0: np.ndarray, horizon: int) -> np.ndarray:
    s = s0[None, :]
    traj = [s[0].copy()]
    for _ in range(horizon):
        s = map_fn_np(s)
        traj.append(s[0].copy())
    return np.array(traj)


test_start  = series[n_train + n_val]
EVAL_HORIZON = 400
pred_roll = rollout_discrete(test_start, EVAL_HORIZON)
true_roll = series[n_train + n_val : n_train + n_val + EVAL_HORIZON + 1]

rmse = np.sqrt(np.mean((pred_roll - true_roll) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={EVAL_HORIZON} steps): {rmse:.6f}")

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Ikeda", exist_ok=True)

# 8a. Attractor overlay
fig, ax = plot_attractor_overlay(
    true_roll, pred_roll,
    dim_x=0, dim_y=1,
    labels=["True Ikeda", "KANDy"],
    colors=["#1f77b4", "#d62728"],
    title="Ikeda optical-cavity map",
    save="results/Ikeda/attractor",
)
plt.close(fig)

# 8b. x and y time series
fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
steps = np.arange(EVAL_HORIZON + 1)
for ax, ci, lab in zip(axes, [0, 1], ["x", "y"]):
    ax.plot(steps, true_roll[:, ci], lw=0.8, color="#1f77b4", label="True")
    ax.plot(steps, pred_roll[:, ci], lw=0.8, color="#d62728", ls="--", label="KANDy")
    ax.set_ylabel(f"${lab}_n$")
    ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("step n")
fig.suptitle("Ikeda map rollout", fontsize=11)
fig.tight_layout()
fig.savefig("results/Ikeda/timeseries.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Ikeda/timeseries.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8c. Edge activations
fig = plot_all_edges(
    model.model_,
    X=train_t,
    input_names=FEATURE_NAMES,
    output_names=["x_{n+1}", "y_{n+1}"],
    title="Ikeda KAN edge activations",
    save="results/Ikeda/edge_activations",
)
plt.close(fig)

# 8d. Loss curves (rollout phase)
if rollout_results:
    fig, ax = plot_loss_curves(
        rollout_results,
        title="Ikeda rollout training loss",
        save="results/Ikeda/loss_curves",
    )
    plt.close(fig)

print("[FIGS]  Saved results/Ikeda/")
