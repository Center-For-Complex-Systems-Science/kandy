#!/usr/bin/env python3
"""KANDy example: Kuramoto–Sivashinsky (KS) PDE.

The KS equation:
    u_t + u*u_x + u_xx + u_xxxx = 0

is a paradigmatic chaotic PDE.  We discretise on a periodic domain [0, L]
with N_x spatial modes and treat the full spatial state u ∈ R^{N_x} as a
single snapshot.  The Koopman lift builds a 6-feature library at each
spatial point using *spectral* derivatives, and a KAN([6, 1]) learns the
universal point-wise operator:

    u_t(x) = f( u, u_x, u_xx, u_xxxx, u·u_x, u·u_xx )

Key design choices:
- **Spectral derivatives** (exact on periodic domains) instead of FD.
- **Minimal library** — no squared features (u_xx², u_x², etc.) which
  create degenerate solutions where the KAN splits a linear activation
  across correlated edges.
- **ETDRK4** (Kassam & Trefethen, 2005) for data generation and rollout.

Each row of the training data is one (space, time) sample, so the KAN
effectively discovers the structure of the KS equation in closed form.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import (
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
# 2. ETDRK4 pseudo-spectral solver  (Kassam & Trefethen, 2005)
# ---------------------------------------------------------------------------
# KS: u_t = -u*u_x - u_xx - u_xxxx
# Linear: L_hat = k² - k⁴  (grows for |k|<1, decays otherwise)
# Nonlinear: N(u) = -1/2 d/dx(u²)

L_hat = k ** 2 - k ** 4   # linear eigenvalues in Fourier space

# Precompute ETDRK4 coefficients via contour integrals (avoids cancellation)
_M = 32
_r  = np.exp(2j * np.pi * (np.arange(1, _M + 1) - 0.5) / _M)
_LR = DT * L_hat[:, None] + _r[None, :]

E    = np.exp(L_hat * DT)
E2   = np.exp(L_hat * DT / 2)
Q    = DT * np.mean((np.exp(_LR / 2) - 1) / _LR, axis=1).real
f1   = DT * np.mean((-4 - _LR + np.exp(_LR) * (4 - 3 * _LR + _LR**2)) / _LR**3, axis=1).real
f2   = DT * np.mean((2 + _LR + np.exp(_LR) * (-2 + _LR)) / _LR**3, axis=1).real
f3   = DT * np.mean((-4 - 3 * _LR - _LR**2 + np.exp(_LR) * (4 - _LR)) / _LR**3, axis=1).real


def ks_nl(u_hat: np.ndarray) -> np.ndarray:
    """KS nonlinear term in Fourier space: N(u) = -1/2 ik FFT(u²)."""
    u = np.fft.irfft(u_hat, n=N_X)
    return -0.5 * (1j * k) * np.fft.rfft(u ** 2)


def ks_etdrk4_step(u_hat: np.ndarray) -> np.ndarray:
    """One ETDRK4 step (dt is baked into the precomputed coefficients)."""
    Nu  = ks_nl(u_hat)
    a   = E2 * u_hat + Q * Nu
    Na  = ks_nl(a)
    b   = E2 * u_hat + Q * Na
    Nb  = ks_nl(b)
    c   = E2 * a + Q * (2 * Nb - Nu)
    Nc  = ks_nl(c)
    return E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3


def generate_ks_data(n_steps: int = N_STEPS, burn: int = BURN,
                     seed: int = SEED):
    """Generate KS trajectory.

    Returns
    -------
    U : (n_steps - burn, N_X) real snapshots
    UH : (n_steps - burn, N_X//2+1) complex Fourier coefficients
    """
    rng = np.random.default_rng(seed)
    u0  = rng.standard_normal(N_X) * 0.1
    u0 -= u0.mean()
    u_hat = np.fft.rfft(u0)

    snapshots = []
    fourier_snapshots = []
    for step in range(n_steps):
        u_hat = ks_etdrk4_step(u_hat)
        if step >= burn:
            snapshots.append(np.fft.irfft(u_hat, n=N_X).real)
            fourier_snapshots.append(u_hat.copy())

    return (np.array(snapshots, dtype=np.float64),
            np.array(fourier_snapshots))


print("[DATA]  Generating KS trajectory ...")
U, UH = generate_ks_data()     # (T, N_X), (T, N_X//2+1)
T_snap = U.shape[0]
print(f"[DATA]  T={T_snap} snapshots, N_x={N_X} modes")

# ---------------------------------------------------------------------------
# 3. Feature library  phi: u ∈ R^{N_x} → theta ∈ R^{N_x × 6}
#
#    At every spatial point the 6-dimensional feature vector is:
#    [u, u_x, u_xx, u_xxxx, u·u_x, u·u_xx]
#
#    IMPORTANT: No squared features (u_xx², u_x², etc.).  Including both
#    u_xx and u_xx² creates a degeneracy where the KAN splits the u_xx
#    activation across both edges (one quadratic, one linear) that cancel.
#    This leads to incorrect symbolic extraction (x² instead of x).
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["u", "u_x", "u_xx", "u_xxxx", "u*u_x", "u*u_xx"]
N_FEATURES = 6


def build_ks_library(U_snap: np.ndarray, UH_snap: np.ndarray) -> np.ndarray:
    """Build feature matrix using *spectral* derivatives.

    Parameters
    ----------
    U_snap : (T, N_x) real snapshots
    UH_snap : (T, N_x//2+1) complex Fourier coefficients

    Returns
    -------
    Theta : (T*N_x, 6)
    """
    T, Nx = U_snap.shape
    all_theta = []
    for t in range(T):
        u = U_snap[t]
        uh = UH_snap[t]
        u_x    = np.fft.irfft(1j * k * uh, n=N_X)
        u_xx   = np.fft.irfft(-k**2 * uh, n=N_X)
        u_xxxx = np.fft.irfft(k**4 * uh, n=N_X)
        theta = np.column_stack([
            u, u_x, u_xx, u_xxxx,
            u * u_x, u * u_xx,
        ])
        all_theta.append(theta)
    return np.vstack(all_theta)


def build_ks_library_single(u: np.ndarray) -> np.ndarray:
    """Build features for a single snapshot (for rollout).

    Uses spectral derivatives computed from FFT of the real-space field.
    """
    uh = np.fft.rfft(u)
    u_x    = np.fft.irfft(1j * k * uh, n=N_X)
    u_xx   = np.fft.irfft(-k**2 * uh, n=N_X)
    u_xxxx = np.fft.irfft(k**4 * uh, n=N_X)
    return np.column_stack([
        u, u_x, u_xx, u_xxxx,
        u * u_x, u * u_xx,
    ])


# Build dataset: each (space, time) cell is one training sample
# Compute u_t via central differences in time (trim boundary)
U_inner  = U[1:-1]        # (T-2, N_x)
UH_inner = UH[1:-1]       # (T-2, N_x//2+1)
U_dot_time = (U[2:] - U[:-2]) / (2 * DT)   # (T-2, N_x) — time derivative

T_inner = U_inner.shape[0]
print(f"[DATA]  Building spectral feature library for {T_inner}×{N_X} = "
      f"{T_inner * N_X} samples ...")

Theta   = build_ks_library(U_inner, UH_inner)  # (T_inner * N_x, 6)
U_t_flat = U_dot_time.ravel()[:, None]          # (T_inner * N_x, 1)

print(f"[DATA]  Theta shape: {Theta.shape}, U_t shape: {U_t_flat.shape}")

# Subsample for PyKAN (LBFGS + grid updates struggle with >100K samples)
MAX_SAMPLES = 100_000
if len(Theta) > MAX_SAMPLES:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Theta), MAX_SAMPLES, replace=False)
    Theta    = Theta[idx]
    U_t_flat = U_t_flat[idx]
    print(f"[DATA]  Subsampled to {MAX_SAMPLES}")

# ---------------------------------------------------------------------------
# 4. KANDy model  (single-layer KAN: width=[6, 1])
# ---------------------------------------------------------------------------
ks_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="ks_identity")

model = KANDy(
    lift=ks_lift,
    grid=7,
    k=3,
    steps=100,
    seed=SEED,
)

model.fit(
    X=Theta,
    X_dot=U_t_flat,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formula for u_t ...")
import sympy as sp
sym_subset = torch.tensor(Theta[:2048], dtype=torch.float32)
model.model_.save_act = True
with torch.no_grad():
    model.model_(sym_subset)
model.model_.auto_symbolic()
exprs, vars_ = model.model_.symbolic_formula()
sub_map = {sp.Symbol(str(v)): sp.Symbol(n) for v, n in zip(vars_, FEATURE_NAMES)}
formulas = []
for expr_str in exprs:
    sym = sp.sympify(expr_str).xreplace(sub_map)
    sym = sp.expand(sym).xreplace(
        {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
    )
    formulas.append(sym)
print(f"  u_t = {formulas[0]}")
print(f"  [TRUE] u_t = -u*u_x - u_xx - u_xxxx")

# ---------------------------------------------------------------------------
# 6. Evaluate point-wise MSE on test samples
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
# 7. Rollout: integrate the learned model in time on held-out snapshots
# ---------------------------------------------------------------------------
def rollout_ks(model, u0: np.ndarray, n_steps: int) -> np.ndarray:
    """ETDRK4 rollout using the learned KANDy model.

    The stiff linear part (u_xx + u_xxxx) is handled exactly via the same
    ETDRK4 coefficients used for data generation.  The "nonlinear" part is
    whatever the learned model predicts minus the known linear contribution:
        N_learned(u) = KANDy(phi(u)) - linear(u)
    """
    u_hat = np.fft.rfft(u0)
    traj  = [u0.copy()]

    for _ in range(n_steps - 1):
        def learned_nl(v_hat):
            v = np.fft.irfft(v_hat, n=N_X)
            theta = build_ks_library_single(v)   # (N_x, 6)
            u_t_full = model.predict(theta).ravel()
            # Subtract the linear part so ETDRK4 handles it implicitly
            v_xx   = np.fft.irfft(-(k ** 2) * v_hat, n=N_X)
            v_xxxx = np.fft.irfft((k ** 4) * v_hat, n=N_X)
            nl_phys = u_t_full - (-v_xx - v_xxxx)   # total - linear = nonlinear
            return np.fft.rfft(nl_phys)

        Nu = learned_nl(u_hat)
        a  = E2 * u_hat + Q * Nu
        Na = learned_nl(a)
        b  = E2 * u_hat + Q * Na
        Nb = learned_nl(b)
        c  = E2 * a + Q * (2 * Nb - Nu)
        Nc = learned_nl(c)
        u_hat = E * u_hat + Nu * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        traj.append(np.fft.irfft(u_hat, n=N_X).real.copy())

    return np.array(traj)


N_ROLLOUT = 200
u0_rollout  = U_inner[int(T_inner * 0.80)]     # start from test region
true_rollout = U_inner[int(T_inner * 0.80): int(T_inner * 0.80) + N_ROLLOUT]
pred_rollout = rollout_ks(model, u0_rollout, N_ROLLOUT)

rmse_rollout = np.sqrt(np.mean((pred_rollout - true_rollout) ** 2))
print(f"[EVAL]  Rollout RMSE (T={N_ROLLOUT} steps, dt={DT}): {rmse_rollout:.6f}")

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/KS", exist_ok=True)

# 8a. Space-time heatmap comparison
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

# 8b. Loss curves
if hasattr(model, "train_results_") and model.train_results_ is not None:
    fig, ax = plot_loss_curves(
        model.train_results_,
        save="results/KS/loss_curves",
    )
    plt.close(fig)

# 8c. Edge activations (subsample training data for speed)
n_sub = min(5000, int(N_total * 0.70))
sub_idx = np.random.choice(int(N_total * 0.70), n_sub, replace=False)
train_theta_t = torch.tensor(Theta[sub_idx], dtype=torch.float32)
fig, axes = plot_all_edges(
    model.model_,
    X=train_theta_t,
    in_var_names=FEATURE_NAMES,
    out_var_names=["u_t"],
    save="results/KS/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/KS/")
