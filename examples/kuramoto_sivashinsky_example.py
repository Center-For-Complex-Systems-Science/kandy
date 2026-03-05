#!/usr/bin/env python3
"""KANDy example: Kuramoto–Sivashinsky (KS) PDE.

The KS equation:
    u_t + u*u_x + u_xx + u_xxxx = 0

is a paradigmatic chaotic PDE.  We discretise on a periodic domain [0, L]
with N_x spatial modes and treat the full spatial state u ∈ R^{N_x} as a
single snapshot.  The Koopman lift builds a 12-feature library at each
spatial point, and a KAN([12, 1]) learns the universal point-wise operator:

    u_t(x) = f( u, u_x, u_xx, u_xxxx, u², u³, u_x², u_xx²,
                 u·u_x, u·u_xx, u·u_xxxx, u_x·u_xx )

Each row of the training data is one (space, time) sample, so the KAN
effectively discovers the structure of the KS equation in closed form.

Data generation: pseudo-spectral method with implicit–explicit (IMEX) time
stepping.  The stiff linear terms (u_xx, u_xxxx) are handled exactly in
Fourier space; the nonlinear term u*u_x is treated explicitly.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_all_edge_activations,
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Problem parameters
# ---------------------------------------------------------------------------
L     = 22.0          # domain length (chaotic regime for L≈22)
N_X   = 64            # spatial grid points
DT    = 0.25          # time step
N_STEPS = 8_000       # total steps
BURN    = 1_000       # transient to discard

x_grid = np.linspace(0, L, N_X, endpoint=False)
dx     = L / N_X

# Wavenumbers for spectral derivatives
k = 2.0 * np.pi / L * np.fft.rfftfreq(N_X, d=1.0 / N_X)

# ---------------------------------------------------------------------------
# 2. Pseudo-spectral IMEX time stepper
# ---------------------------------------------------------------------------

def ks_imex_step(u_hat: np.ndarray, dt: float) -> np.ndarray:
    """One IMEX step for KS in Fourier space.

    Linear part  (u_xx + u_xxxx) handled exactly (implicit).
    Nonlinear part  (-u*u_x = -1/2 * d/dx(u²)) handled explicitly.
    """
    k2 = k ** 2
    k4 = k ** 4
    # Exact factor for stiff linear terms
    L_factor = np.exp((-k2 + k4) * dt)      # e^{(k²-k⁴)dt} wrong sign fix below
    # KS: u_t = -u*u_x - u_xx - u_xxxx
    # Linear: -u_xx - u_xxxx → eigenvalue = k² - k⁴  (grows for k<1, decays for k>1)
    L_factor = np.exp((k2 - k4) * dt)       # exact integration factor

    # Nonlinear: -u * u_x = -1/2 d/dx(u²) in physical space
    u      = np.fft.irfft(u_hat, n=N_X)
    duudx  = np.fft.rfft(u ** 2)
    nl_hat = -0.5 * (1j * k) * duudx        # -1/2 * ik * FFT(u²)

    u_hat_new = L_factor * (u_hat + dt * nl_hat)
    return u_hat_new


def generate_ks_data(n_steps: int = N_STEPS, burn: int = BURN,
                     seed: int = SEED) -> np.ndarray:
    """Generate KS trajectory.  Returns array of shape (n_steps - burn, N_X)."""
    rng = np.random.default_rng(seed)
    # Random low-mode initial condition
    u0  = rng.standard_normal(N_X) * 0.1
    u0 -= u0.mean()
    u_hat = np.fft.rfft(u0)

    snapshots = []
    for step in range(n_steps):
        u_hat = ks_imex_step(u_hat, DT)
        if step >= burn:
            snapshots.append(np.fft.irfft(u_hat, n=N_X).real)

    return np.array(snapshots, dtype=np.float32)   # (T, N_X)


print("[DATA]  Generating KS trajectory ...")
U = generate_ks_data()     # (T, N_X)
T_snap = U.shape[0]
print(f"[DATA]  T={T_snap} snapshots, N_x={N_X} modes")

# ---------------------------------------------------------------------------
# 3. Finite-difference derivative matrices (periodic)
# ---------------------------------------------------------------------------

def make_fd_matrices(n: int, dx: float):
    """Return periodic FD matrices D1, D2, D4 (all shape (n, n))."""
    e = np.ones(n)
    # First derivative — central difference O(h²)
    D1 = (np.diag(e[:-1], 1) - np.diag(e[:-1], -1)) / (2 * dx)
    D1[0, -1] = -1.0 / (2 * dx)
    D1[-1, 0] =  1.0 / (2 * dx)

    # Second derivative
    D2 = (np.diag(e[:-1], 1) - 2 * np.diag(e) + np.diag(e[:-1], -1)) / dx**2
    D2[0, -1] = D2[-1, 0] = 1.0 / dx**2

    # Fourth derivative — compose D2 @ D2
    D4 = D2 @ D2

    return D1, D2, D4


D1, D2, D4 = make_fd_matrices(N_X, dx)

# ---------------------------------------------------------------------------
# 4. Feature library  phi: u ∈ R^{N_x} → theta ∈ R^{N_x × 12}
#
#    At every spatial point the 12-dimensional feature vector is:
#    [u, u_x, u_xx, u_xxxx, u², u³, u_x², u_xx², u·u_x, u·u_xx, u·u_xxxx, u_x·u_xx]
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "u", "u_x", "u_xx", "u_xxxx",
    "u^2", "u^3", "u_x^2", "u_xx^2",
    "u*u_x", "u*u_xx", "u*u_xxxx", "u_x*u_xx",
]
N_FEATURES = 12


def build_ks_library(U_row: np.ndarray) -> np.ndarray:
    """Build feature matrix for one or many snapshots.

    Parameters
    ----------
    U_row : np.ndarray, shape (N_x,) or (T, N_x)

    Returns
    -------
    Theta : np.ndarray, shape (N_x, 12) or (T*N_x, 12)
    """
    batch = U_row.ndim == 2
    if not batch:
        U_row = U_row[None, :]          # (1, N_x)

    T, Nx = U_row.shape
    all_theta = []
    for t in range(T):
        u    = U_row[t]                  # (N_x,)
        u_x  = D1 @ u
        u_xx = D2 @ u
        u_xxxx = D4 @ u
        theta = np.column_stack([
            u, u_x, u_xx, u_xxxx,
            u**2, u**3, u_x**2, u_xx**2,
            u * u_x, u * u_xx, u * u_xxxx, u_x * u_xx,
        ])                              # (N_x, 12)
        all_theta.append(theta)

    result = np.vstack(all_theta)       # (T*N_x, 12)
    return result if batch else result  # always (T*N_x, 12)


# Build dataset: each (space, time) cell is one training sample
# Compute u_t via central differences in time (trim boundary)
U_inner    = U[1:-1]        # (T-2, N_x)
U_dot_time = (U[2:] - U[:-2]) / (2 * DT)   # (T-2, N_x) — time derivative

T_inner = U_inner.shape[0]
print(f"[DATA]  Building feature library for {T_inner}×{N_X} = "
      f"{T_inner * N_X} samples ...")

Theta   = build_ks_library(U_inner)     # (T_inner * N_x, 12)
U_t_flat = U_dot_time.ravel()           # (T_inner * N_x,)   — scalar targets
U_t_flat = U_t_flat[:, None]           # (T_inner * N_x, 1)

print(f"[DATA]  Theta shape: {Theta.shape}, U_t shape: {U_t_flat.shape}")

# ---------------------------------------------------------------------------
# 5. KANDy model  (single-layer KAN: width=[12, 1])
# ---------------------------------------------------------------------------
ks_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="ks_identity")

model = KANDy(
    lift=ks_lift,
    grid=7,
    k=3,
    steps=500,
    seed=SEED,
)

# Theta is already the lifted state — pass X=Theta, X_dot=U_t
# The lift inside KANDy will be identity (we pre-built the features)
model.fit(
    X=Theta,
    X_dot=U_t_flat,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 6. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=2)
    print(f"  u_t = {formulas[0]}")
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 7. Evaluate point-wise MSE on test samples
# ---------------------------------------------------------------------------
N_total    = len(Theta)
n_test     = int(N_total * 0.15)
Theta_test = Theta[N_total - n_test:]
U_t_test   = U_t_flat[N_total - n_test:]

U_t_pred   = model.predict(Theta_test)
mse = np.mean((U_t_pred - U_t_test) ** 2)
print(f"\n[EVAL]  Test MSE (point-wise): {mse:.6e}")
print(f"[EVAL]  Test RMSE:             {mse**0.5:.6e}")

# ---------------------------------------------------------------------------
# 8. Rollout: integrate the learned model in time on held-out snapshots
# ---------------------------------------------------------------------------
def rollout_ks(model_fn, u0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Euler rollout using the KANDy point-wise model.

    model_fn takes a feature matrix (N_x, 12) and returns u_t (N_x, 1).
    """
    u = u0.copy()
    traj = [u.copy()]
    for _ in range(n_steps - 1):
        theta = build_ks_library(u[None, :])    # (N_x, 12)
        u_t   = model.predict(theta).ravel()    # (N_x,)
        u     = u + dt * u_t                    # Euler step
        traj.append(u.copy())
    return np.array(traj)


N_ROLLOUT = 200
u0_rollout  = U_inner[int(T_inner * 0.80)]     # start from test region
true_rollout = U_inner[int(T_inner * 0.80): int(T_inner * 0.80) + N_ROLLOUT]
pred_rollout = rollout_ks(model, u0_rollout, N_ROLLOUT, DT)

rmse_rollout = np.sqrt(np.mean((pred_rollout - true_rollout) ** 2))
print(f"[EVAL]  Rollout RMSE (T={N_ROLLOUT} steps, dt={DT}): {rmse_rollout:.6f}")

# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/KS", exist_ok=True)

# 9a. Space-time heatmap comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t_arr = np.arange(N_ROLLOUT) * DT
for ax, data, title in zip(axes,
                             [true_rollout, pred_rollout],
                             ["True KS", "KANDy"]):
    im = ax.imshow(data.T, origin="lower", aspect="auto",
                   extent=[0, t_arr[-1], 0, L],
                   cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xlabel("time t"); ax.set_ylabel("x")
    ax.set_title(title)
fig.colorbar(im, ax=axes, label="u(x,t)")
fig.suptitle("Kuramoto–Sivashinsky", fontsize=12)
fig.tight_layout()
fig.savefig("results/KS/spacetime.png", dpi=300, bbox_inches="tight")
fig.savefig("results/KS/spacetime.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9b. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(
        model.train_results_,
        title="KS training loss",
        save="results/KS/loss_curves",
    )
    plt.close(fig)

# 9c. Edge activations (subsample training data for speed)
n_sub = min(5000, int(N_total * 0.70))
sub_idx = np.random.choice(int(N_total * 0.70), n_sub, replace=False)
train_theta_t = torch.tensor(Theta[sub_idx], dtype=torch.float32)
fig = plot_all_edges(
    model.model_,
    X=train_theta_t,
    input_names=FEATURE_NAMES,
    output_names=["u_t"],
    title="KS KAN edge activations",
    save="results/KS/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/KS/")
