#!/usr/bin/env python3
"""KANDy example: Adaptive Kuramoto–Sakaguchi oscillators.

N coupled phase oscillators with adaptive coupling weights:
    dθ_i/dt  = ω_i + Σ_j  κ_ij * sin(θ_j - θ_i + α)
    dκ_ij/dt = -ε * [κ_ij + sin(θ_j - θ_i + β)]

Parameters:
    N     = 5 oscillators
    ω_i   ~ Uniform(-0.5, 0.5)  (fixed natural frequencies)
    α, β  = -π/4, -π/2
    ε     = 0.1  (adaptation rate)

State vector: (θ_1,...,θ_N, κ_12,...,κ_N(N-1)) ∈ R^{N + N(N-1)}

Koopman lift: relative phases (sin AND cos) using θ_i - θ_j convention
    phi(θ, κ) = [sin(θ_i - θ_j) for i≠j]          N*(N-1) features
              ∪ [cos(θ_i - θ_j) for i≠j]          N*(N-1) features
              ∪ [κ_ij for i≠j]                     N*(N-1) features
              = 3*N*(N-1) features total

Why both sin and cos? sin(Δθ + α) = sin(Δθ)cos(α) + cos(Δθ)sin(α) requires
both features; with θ_i - θ_j inputs the model learns sin(θ_i-θ_j-|α|) directly.

KAN:  width = [3*N*(N-1), N]   (predicts N phase velocities)
      base_fun = torch.sin
A second KAN (width = [3*N*(N-1), N*(N-1)]) learns the κ dynamics.

Three improvements over the baseline:
  1. cos(θ_j - θ_i) added to the lift (fixes bilinear representation gap)
  2. Rollout training for the phase model (rollout_weight=0.3, horizon=8 steps, Adam+mini-batch)
  3. Rollout evaluation at T=500 steps (matches the RMSE target horizon)
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from kan import KAN

from kandy import KANDy, CustomLift, angle_mse
from kandy.training import fit_kan, make_windows
from kandy.plotting import (
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED  = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N      = 5             # number of oscillators
ALPHA  = -np.pi / 4   # phase lag in phase equation
BETA   = -np.pi / 2   # phase lag in coupling equation
EPS    = 0.1          # adaptation rate

ROLLOUT_HORIZON = 8    # steps per trajectory window during training
T_EVAL          = 500  # evaluation rollout steps (matches RMSE target)
VAL_FRAC        = 0.15
TEST_FRAC       = 0.15

# Fixed natural frequencies
rng   = np.random.default_rng(SEED)
OMEGA = rng.uniform(-0.5, 0.5, size=N)

# Off-diagonal index pairs (i, j) where i ≠ j, ordered (0,1),(0,2),...
OD_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
N_OD     = len(OD_PAIRS)   # N*(N-1) = 20
N_FEAT   = 3 * N_OD        # sin + cos + κ = 60
N_STATE  = N + N_OD        # 5 + 20 = 25  (raw state for rollout)

DEVICE = torch.device("cpu")  # PyKAN has CUDA tensor issues

print(f"[PARAMS] N={N}, N_OD={N_OD}, N_FEAT={N_FEAT}, N_STATE={N_STATE}")
print(f"         KAN_theta: [{N_FEAT}, {N}]")
print(f"         KAN_kappa: [{N_FEAT}, {N_OD}]")

# ---------------------------------------------------------------------------
# 1. Simulation
# ---------------------------------------------------------------------------

def adaptive_kuramoto_rhs(t, y):
    """Full RHS of the adaptive Kuramoto-Sakaguchi system."""
    theta = y[:N]
    kappa = y[N:].reshape(N, N)

    dtheta = OMEGA.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                dtheta[i] += kappa[i, j] * np.sin(theta[j] - theta[i] + ALPHA)

    dkappa = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dkappa[i, j] = -EPS * (kappa[i, j] + np.sin(theta[j] - theta[i] + BETA))

    return np.concatenate([dtheta, dkappa.ravel()])


def simulate(T_end: float = 200.0, dt: float = 0.05, seed: int = SEED):
    """Simulate adaptive Kuramoto; return (t, theta_traj, kappa_traj)."""
    rng0  = np.random.default_rng(seed)
    theta0 = rng0.uniform(0, 2 * np.pi, size=N)
    kappa0 = rng0.uniform(-0.5, 0.5, size=(N, N))
    np.fill_diagonal(kappa0, 0.0)

    y0     = np.concatenate([theta0, kappa0.ravel()])
    t_eval = np.arange(0, T_end, dt)
    sol    = solve_ivp(adaptive_kuramoto_rhs, [0, T_end], y0,
                       t_eval=t_eval, method="RK45",
                       rtol=1e-8, atol=1e-10, dense_output=False)

    theta_traj = sol.y[:N, :].T
    kappa_traj = sol.y[N:, :].T.reshape(-1, N, N)
    return sol.t, theta_traj, kappa_traj


print("[DATA]  Simulating adaptive Kuramoto ...")
t_sim, THETA, KAPPA = simulate(T_end=300.0, dt=0.05)
T_snap = len(t_sim)
DT_SIM = t_sim[1] - t_sim[0]
print(f"[DATA]  T={T_snap} snapshots, dt={DT_SIM:.3f}")

# ---------------------------------------------------------------------------
# 2. Koopman lift  phi(theta, kappa) -> R^{3*N_OD}
#    FIX 1: include cos(θ_j - θ_i) — needed to represent sin(Δθ + α) exactly
# ---------------------------------------------------------------------------
SIN_NAMES   = [f"sin(θ{i}-θ{j})" for (i, j) in OD_PAIRS]
COS_NAMES   = [f"cos(θ{i}-θ{j})" for (i, j) in OD_PAIRS]
KAPPA_NAMES = [f"κ{i}{j}"         for (i, j) in OD_PAIRS]
FEATURE_NAMES = SIN_NAMES + COS_NAMES + KAPPA_NAMES


def build_lift(theta: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Lift (T, N) + (T, N, N) → (T, 3*N_OD).

    theta : (T, N) phase angles
    kappa : (T, N, N) coupling matrix (diagonal unused)
    Uses θ_i - θ_j convention (negative of θ_j - θ_i).
    """
    sin_feats   = np.column_stack(
        [np.sin(theta[:, i] - theta[:, j]) for (i, j) in OD_PAIRS]
    )                                           # (T, N_OD)
    cos_feats   = np.column_stack(
        [np.cos(theta[:, i] - theta[:, j]) for (i, j) in OD_PAIRS]
    )                                           # (T, N_OD)
    kappa_feats = np.column_stack(
        [kappa[:, i, j] for (i, j) in OD_PAIRS]
    )                                           # (T, N_OD)
    return np.hstack([sin_feats, cos_feats, kappa_feats])  # (T, 3*N_OD)


Theta = build_lift(THETA, KAPPA)       # (T, N_FEAT)

# Targets: phase velocities and κ velocities via central differences
Theta_inner = Theta[1:-1]
THETA_inner = THETA[1:-1]
KAPPA_inner = KAPPA[1:-1]

THETA_dot = (THETA[2:] - THETA[:-2]) / (2.0 * DT_SIM)   # (T-2, N)
KAPPA_dot = (KAPPA[2:] - KAPPA[:-2]) / (2.0 * DT_SIM)   # (T-2, N, N)

KAPPA_dot_od = np.column_stack(
    [KAPPA_dot[:, i, j] for (i, j) in OD_PAIRS]
)   # (T-2, N_OD)

T_INNER = len(Theta_inner)
print(f"[DATA]  Training samples: {T_INNER}")

# Chronological split indices (must match KANDy defaults)
n1 = int(T_INNER * (1 - VAL_FRAC - TEST_FRAC))   # end of train
n2 = int(T_INNER * (1 - TEST_FRAC))               # end of val


def _t(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# 3. Raw state trajectory for rollout training
#    State = (θ_1,...,θ_N, κ_01,...,κ_N(N-1)) — N_STATE = 25 dims
# ---------------------------------------------------------------------------
KAPPA_OD_inner = np.column_stack(
    [KAPPA_inner[:, i, j] for (i, j) in OD_PAIRS]
)                                               # (T-2, N_OD)
state_inner = np.hstack([THETA_inner, KAPPA_OD_inner])   # (T-2, N_STATE=25)

# Windowed segments for rollout: (B, HORIZON+1, N_STATE)
train_traj_windows = make_windows(
    _t(state_inner[:n1]),
    window=ROLLOUT_HORIZON + 1,
)
train_t_arr = torch.arange(ROLLOUT_HORIZON + 1, dtype=torch.float32, device=DEVICE) * float(DT_SIM)
print(f"[DATA]  train_traj windows: {train_traj_windows.shape}")

# ---------------------------------------------------------------------------
# 4. Train Model B: coupling dynamics dκ/dt  (derivative supervision only)
#    Train this FIRST so it can be used (frozen) in the phase rollout.
# ---------------------------------------------------------------------------
print("\n--- Model B: coupling dynamics dκ/dt (KAN=[{}, {}]) ---".format(N_FEAT, N_OD))

kappa_lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kappa_lift")

model_kappa = KANDy(
    lift=kappa_lift,
    grid=5,
    k=3,
    steps=200,
    seed=SEED,
    base_fun=torch.sin,
)

model_kappa.fit(
    X=Theta_inner,
    X_dot=KAPPA_dot_od,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
    lamb=0.0,
)

# ---------------------------------------------------------------------------
# 5. Train Model A: phase dynamics dθ/dt  with rollout regularisation
#    FIX 2 & 3: rollout_weight=0.3, coupled dynamics_fn, horizon=8 steps, Adam
# ---------------------------------------------------------------------------
print("\n--- Model A: phase dynamics dθ/dt with rollout (KAN=[{}, {}]) ---".format(N_FEAT, N))

# Build the KAN directly so we can pass it to fit_kan with train_traj
theta_kan = KAN(
    width=[N_FEAT, N],
    grid=5,
    k=3,
    seed=SEED,
    base_fun=torch.sin,
).to(DEVICE)

# Freeze kappa model during theta rollout training
for p in model_kappa.model_.parameters():
    p.requires_grad_(False)


def _build_phi_torch(state: torch.Tensor) -> torch.Tensor:
    """Build lifted features from raw state.  state: (B, N_STATE) -> phi: (B, N_FEAT)"""
    th = state[:, :N]    # (B, N)
    kp = state[:, N:]    # (B, N_OD)
    sin_f = torch.stack([torch.sin(th[:, i] - th[:, j]) for (i, j) in OD_PAIRS], dim=1)
    cos_f = torch.stack([torch.cos(th[:, i] - th[:, j]) for (i, j) in OD_PAIRS], dim=1)
    return torch.cat([sin_f, cos_f, kp], dim=1)   # (B, N_FEAT)


def dynamics_fn_coupled(state: torch.Tensor) -> torch.Tensor:
    """Coupled dynamics for rollout training.
    state: (B, N_STATE)  ->  d_state: (B, N_STATE)
    Gradient flows through theta_kan only; model_kappa is frozen.
    """
    phi = _build_phi_torch(state)
    dtheta = theta_kan(phi)                  # (B, N)   — gradient here
    with torch.no_grad():
        dkappa = model_kappa.model_(phi)     # (B, N_OD) — frozen
    return torch.cat([dtheta, dkappa], dim=1)   # (B, N_STATE)


def theta_rollout_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """angle_mse on the θ component only.  pred, true: (B, T, N_STATE)"""
    return angle_mse(pred[:, :, :N], true[:, :, :N])


# Full dataset: derivative supervision + rollout trajectories
dataset_theta = {
    "train_input":  _t(Theta_inner[:n1]),
    "train_label":  _t(THETA_dot[:n1]),
    "val_input":    _t(Theta_inner[n1:n2]),
    "val_label":    _t(THETA_dot[n1:n2]),
    "test_input":   _t(Theta_inner[n2:]),
    "test_label":   _t(THETA_dot[n2:]),
    "train_traj":   train_traj_windows,
    "train_t":      train_t_arr,
}

# Phase 1: LBFGS warm-start (derivative supervision only)
print("  Phase 1: LBFGS warm-start (200 steps, deriv only) ...")
fit_kan(
    theta_kan,
    dataset_theta,
    opt="LBFGS",
    steps=200,
    lamb=0.0,
    rollout_weight=0.0,
)

# Phase 2: Adam fine-tune with rollout regularisation
print("  Phase 2: Adam rollout training (200 steps, rollout_weight=0.3) ...")
train_results_theta = fit_kan(
    theta_kan,
    dataset_theta,
    opt="Adam",
    lr=5e-5,
    steps=200,
    lamb=0.0,
    rollout_weight=0.3,
    rollout_loss_fn=theta_rollout_loss,
    dynamics_fn=dynamics_fn_coupled,
    rollout_horizon=ROLLOUT_HORIZON,
    integrator="rk4",
    traj_batch=256,
    patience=50,
)

# Re-enable gradients on kappa model (needed for symbolic extraction later)
for p in model_kappa.model_.parameters():
    p.requires_grad_(True)

# Wrap theta_kan in a KANDy shell for consistent predict/get_formula API
theta_lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="kuramoto_lift")
model_theta = KANDy(
    lift=theta_lift,
    grid=5,
    k=3,
    steps=200,
    seed=SEED,
    base_fun=torch.sin,
    device="cpu",
)
model_theta.model_          = theta_kan
model_theta.lift_dim_       = N_FEAT
model_theta.state_dim_      = N
model_theta._is_fitted      = True
model_theta._train_input    = dataset_theta["train_input"]
model_theta.train_results_  = train_results_theta

# ---------------------------------------------------------------------------
# 6. Coupled rollout — FIX 3: evaluate at T=500 steps
#    NOTE: rollout MUST happen before symbolic extraction.
#    get_formula() calls auto_symbolic() which replaces splines with fitted
#    symbolic functions (log, sqrt, 1/x) that can blow up outside the
#    training distribution.
# ---------------------------------------------------------------------------

def rollout_coupled(theta0: np.ndarray, kappa0_od: np.ndarray,
                    T_steps: int, dt: float) -> tuple:
    """RK4 rollout using both learned models.

    theta0    : (N,) initial phases
    kappa0_od : (N_OD,) initial off-diagonal couplings (flattened)
    Returns: (theta_traj (T, N), kappa_od_traj (T, N_OD))
    """
    theta    = theta0.copy().astype(np.float64)
    kappa_od = kappa0_od.copy().astype(np.float64)
    theta_hist    = [theta.copy()]
    kappa_od_hist = [kappa_od.copy()]

    def get_phi(th, kp):
        """Build the 3*N_OD feature vector from current state."""
        sin_f = np.array([np.sin(th[i] - th[j]) for (i, j) in OD_PAIRS])
        cos_f = np.array([np.cos(th[i] - th[j]) for (i, j) in OD_PAIRS])
        return np.concatenate([sin_f, cos_f, kp])[None, :]   # (1, 3*N_OD)

    def f(th, kp):
        phi = get_phi(th, kp)
        dth = model_theta.predict(phi).ravel()
        dkp = model_kappa.predict(phi).ravel()
        return dth, dkp

    for _ in range(T_steps - 1):
        # RK4
        dt1, dk1 = f(theta,                   kappa_od)
        dt2, dk2 = f(theta + 0.5*dt*dt1,      kappa_od + 0.5*dt*dk1)
        dt3, dk3 = f(theta + 0.5*dt*dt2,      kappa_od + 0.5*dt*dk2)
        dt4, dk4 = f(theta + dt*dt3,           kappa_od + dt*dk3)

        theta    = theta    + (dt / 6.0) * (dt1 + 2*dt2 + 2*dt3 + dt4)
        kappa_od = kappa_od + (dt / 6.0) * (dk1 + 2*dk2 + 2*dk3 + dk4)
        theta_hist.append(theta.copy())
        kappa_od_hist.append(kappa_od.copy())

    return np.array(theta_hist), np.array(kappa_od_hist)


t0_idx = T_INNER - int(T_INNER * TEST_FRAC)
T_EVAL = min(T_EVAL, T_INNER - t0_idx)   # don't exceed available test data

theta0_test   = THETA_inner[t0_idx]
kappa_od_test = np.array([KAPPA_inner[t0_idx, i, j] for (i, j) in OD_PAIRS])

print(f"\n[ROLLOUT] Rolling out {T_EVAL} steps ...")
pred_theta, pred_kappa_od = rollout_coupled(
    theta0_test, kappa_od_test, T_EVAL, DT_SIM
)
true_theta    = THETA_inner[t0_idx:t0_idx + T_EVAL]
true_kappa_od = np.column_stack([
    KAPPA_inner[t0_idx:t0_idx + T_EVAL, i, j] for (i, j) in OD_PAIRS
])

rmse_theta = np.sqrt(np.mean((pred_theta - true_theta) ** 2))
rmse_kappa = np.sqrt(np.mean((pred_kappa_od - true_kappa_od) ** 2))
print(f"[EVAL]  Rollout RMSE θ: {rmse_theta:.6f}")
print(f"[EVAL]  Rollout RMSE κ: {rmse_kappa:.6f}")

# Order parameter r(t) = |1/N * Σ_j exp(i θ_j)|
r_true = np.abs(np.mean(np.exp(1j * true_theta), axis=1))   # (T_EVAL,)
r_pred = np.abs(np.mean(np.exp(1j * pred_theta), axis=1))   # (T_EVAL,)
rmse_r = np.sqrt(np.mean((r_pred - r_true) ** 2))
print(f"[EVAL]  Order parameter RMSE r: {rmse_r:.6f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Adaptive-Kuramoto", exist_ok=True)

t_roll = np.arange(T_EVAL) * DT_SIM

# 7a. Phase trajectories
fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
colors = plt.cm.tab10.colors
for i, ax in enumerate(axes):
    ax.plot(t_roll, true_theta[:, i],  color=colors[i], lw=1.2, label="True")
    ax.plot(t_roll, pred_theta[:, i],  color=colors[i], lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(f"$\\theta_{i}$")
    if i == 0:
        ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: phase trajectories", fontsize=11)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/phase_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/phase_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Coupling weight trajectories (first 4 pairs)
n_show = min(4, N_OD)
fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show), sharex=True)
if n_show == 1:
    axes = [axes]
for k_idx, ax in enumerate(axes):
    i, j = OD_PAIRS[k_idx]
    ax.plot(t_roll, true_kappa_od[:, k_idx],  color=colors[k_idx % 10], lw=1.2)
    ax.plot(t_roll, pred_kappa_od[:, k_idx],  color=colors[k_idx % 10], lw=1.0, ls="--")
    ax.set_ylabel(f"$\\kappa_{{{i}{j}}}$")
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: coupling weights", fontsize=11)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/kappa_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/kappa_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. Order parameter r(t)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_roll, r_true, color="steelblue",  lw=1.4, label="True $r(t)$")
ax.plot(t_roll, r_pred, color="tomato", lw=1.0, ls="--", label="KANDy $r(t)$")
ax.set_xlabel("time")
ax.set_ylabel("$r(t)$")
ax.set_title(f"Kuramoto order parameter  (RMSE={rmse_r:.4f})")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/order_parameter.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/order_parameter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7d. Loss curves (theta model)
if hasattr(model_theta, "train_results_") and model_theta.train_results_:
    fig, ax = plot_loss_curves(
        model_theta.train_results_,
        save="results/Adaptive-Kuramoto/loss_phase",
    )
    plt.close(fig)

print("[FIGS]  Saved results/Adaptive-Kuramoto/")

# ---------------------------------------------------------------------------
# 8. Symbolic extraction (after rollout — splines replaced with sym fns here)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SYMBOLIC EQUATION DISCOVERY")
print("=" * 60)
print(f"\nTrue dθ_i/dt = ω_i + Σ_j κ_ij · sin(θ_i - θ_j - {ALPHA:.3f})")
print(f"     (α={ALPHA:.4f} = -π/4,  inputs: θ_i - θ_j, κ_ij)\n")
print(f"True dκ_ij/dt = -ε·[κ_ij + sin(θ_i - θ_j - {BETA:.3f})]")
print(f"     (β={BETA:.4f} = -π/2,  ε={EPS})\n")

for sym_name, mdl, out_labels in [
    ("Phase dynamics  dθ/dt",
     model_theta,
     [f"dθ{i}/dt" for i in range(N)]),
    ("Coupling dynamics  dκ/dt",
     model_kappa,
     [f"dκ{i}{j}/dt" for (i, j) in OD_PAIRS]),
]:
    print(f"--- {sym_name} ---")
    try:
        formulas = mdl.get_formula(var_names=FEATURE_NAMES, round_places=2)
        for lab, f in zip(out_labels, formulas):
            print(f"  {lab} = {f}")
    except Exception as exc:
        print(f"  Symbolic extraction failed: {exc}")
    print()
