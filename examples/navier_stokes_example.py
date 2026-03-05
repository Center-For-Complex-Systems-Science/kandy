#!/usr/bin/env python3
"""KANDy example: 3D incompressible Navier–Stokes (ABC Beltrami flow).

The Arnold–Beltrami–Childress (ABC) flow is an exact steady solution of the
3D incompressible Euler equations and an approximate solution of N-S at low Re.
The velocity field on the torus [0, 2π]³ is:

    u = A*sin(z) + C*cos(y)
    v = B*sin(x) + A*cos(z)
    w = C*sin(y) + B*cos(x)

with A=B=C=1.

We perturb this base state and simulate the relaxation / evolution dynamics
using a pseudo-spectral method.  The Koopman lift for the velocity field
u ∈ R³ at each grid point uses:

    phi(u, ω) = [u₁, u₂, u₃, ω₁, ω₂, ω₃, |u|², |ω|², u·ω, Δu₁, Δu₂, Δu₃]
                  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^  ^^^^  ^^^  ^^^^^^^^^^^^^
                  velocity (3)      vorticity (3)    quad  quad  dot  Laplacian (3)
               = 12 features

KAN:  width = [12, 3]   (learns u_t for all 3 velocity components simultaneously)

Note: The N-S data generation uses spectral methods on a coarse grid; this
example focuses on demonstrating the KANDy API for 3D vector-field dynamics.
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
    plot_trajectory_error,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Grid parameters
N_GRID = 16          # N³ spatial grid (coarse for speed)
L      = 2.0 * np.pi
A_ABC  = 1.0         # ABC parameters
B_ABC  = 1.0
C_ABC  = 1.0
NU     = 0.05        # kinematic viscosity (low Re)
DT     = 0.01
N_T    = 3_000
BURN   = 500

# ---------------------------------------------------------------------------
# 1. ABC base flow and spectral tools
# ---------------------------------------------------------------------------

def abc_flow(N: int, A: float = 1.0, B: float = 1.0, C: float = 1.0):
    """Return (u, v, w) arrays of shape (N, N, N) for the ABC flow."""
    xi  = np.linspace(0, L, N, endpoint=False)
    x, y, z = np.meshgrid(xi, xi, xi, indexing="ij")
    u = A * np.sin(z) + C * np.cos(y)
    v = B * np.sin(x) + A * np.cos(z)
    w = C * np.sin(y) + B * np.cos(x)
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)


def spectral_deriv(f_hat: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Compute ∂/∂x_axis of field in spectral space (in-place friendly)."""
    return np.real(np.fft.ifftn(1j * k[axis] * f_hat))


def make_wavenumbers(N: int, L: float):
    """Return 3-tuple of wavenumber arrays broadcastable to (N, N, N)."""
    k1d = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
    kx  = k1d[:, None, None]
    ky  = k1d[None, :, None]
    kz  = k1d[None, None, :]
    return [kx, ky, kz]


def project_incompressible(u_hat, v_hat, w_hat, kv):
    """Helmholtz projection to enforce ∇·u = 0."""
    kx, ky, kz = kv
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0   # avoid div by zero (zero mode stays zero)
    div_hat = kx * u_hat + ky * v_hat + kz * w_hat
    u_hat -= kx * div_hat / k2
    v_hat -= ky * div_hat / k2
    w_hat -= kz * div_hat / k2
    u_hat[0, 0, 0] = v_hat[0, 0, 0] = w_hat[0, 0, 0] = 0.0
    return u_hat, v_hat, w_hat


# ---------------------------------------------------------------------------
# 2. Pseudo-spectral RK4 simulation of 3D N-S
# ---------------------------------------------------------------------------

def ns_rhs(u_hat, v_hat, w_hat, kv, nu: float):
    """Compute RHS of 3D incompressible N-S in spectral space."""
    kx, ky, kz = kv
    u = np.real(np.fft.ifftn(u_hat))
    v = np.real(np.fft.ifftn(v_hat))
    w = np.real(np.fft.ifftn(w_hat))

    # Compute vorticity ω = ∇ × u
    dw_dy = spectral_deriv(w_hat, kv, 1)
    dv_dz = spectral_deriv(v_hat, kv, 2)
    du_dz = spectral_deriv(u_hat, kv, 2)
    dw_dx = spectral_deriv(w_hat, kv, 0)
    dv_dx = spectral_deriv(v_hat, kv, 0)
    du_dy = spectral_deriv(u_hat, kv, 1)

    # Nonlinear term: -(u·∇)u  via physical-space product + spectral div
    du_dx = spectral_deriv(u_hat, kv, 0)
    dv_dy = spectral_deriv(v_hat, kv, 1)
    dw_dz = spectral_deriv(w_hat, kv, 2)

    nl_u = -(u * du_dx + v * du_dy + w * du_dz)
    nl_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    nl_w = -(u * dw_dx + v * dw_dy + w * dw_dz)

    nl_u_hat = np.fft.fftn(nl_u)
    nl_v_hat = np.fft.fftn(nl_v)
    nl_w_hat = np.fft.fftn(nl_w)

    k2 = kx**2 + ky**2 + kz**2
    rhs_u = nl_u_hat - nu * k2 * u_hat
    rhs_v = nl_v_hat - nu * k2 * v_hat
    rhs_w = nl_w_hat - nu * k2 * w_hat

    return rhs_u, rhs_v, rhs_w


def simulate_ns(N: int, n_t: int, dt: float, burn: int,
                nu: float = NU, seed: int = SEED):
    """Simulate 3D N-S starting from perturbed ABC flow."""
    rng = np.random.default_rng(seed)
    u, v, w = abc_flow(N, A_ABC, B_ABC, C_ABC)
    # Add small random perturbation
    u += 0.05 * rng.standard_normal((N, N, N)).astype(np.float32)
    v += 0.05 * rng.standard_normal((N, N, N)).astype(np.float32)
    w += 0.05 * rng.standard_normal((N, N, N)).astype(np.float32)

    kv = make_wavenumbers(N, L)

    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    u_hat, v_hat, w_hat = project_incompressible(u_hat, v_hat, w_hat, kv)

    snapshots = []

    for step in range(n_t):
        # RK4
        r1u, r1v, r1w = ns_rhs(u_hat, v_hat, w_hat, kv, nu)
        r2u, r2v, r2w = ns_rhs(u_hat + 0.5*dt*r1u, v_hat + 0.5*dt*r1v,
                                w_hat + 0.5*dt*r1w, kv, nu)
        r3u, r3v, r3w = ns_rhs(u_hat + 0.5*dt*r2u, v_hat + 0.5*dt*r2v,
                                w_hat + 0.5*dt*r2w, kv, nu)
        r4u, r4v, r4w = ns_rhs(u_hat + dt*r3u, v_hat + dt*r3v,
                                w_hat + dt*r3w, kv, nu)
        u_hat += (dt / 6.0) * (r1u + 2*r2u + 2*r3u + r4u)
        v_hat += (dt / 6.0) * (r1v + 2*r2v + 2*r3v + r4v)
        w_hat += (dt / 6.0) * (r1w + 2*r2w + 2*r3w + r4w)
        u_hat, v_hat, w_hat = project_incompressible(u_hat, v_hat, w_hat, kv)

        if step >= burn:
            snap_u = np.real(np.fft.ifftn(u_hat))
            snap_v = np.real(np.fft.ifftn(v_hat))
            snap_w = np.real(np.fft.ifftn(w_hat))
            snapshots.append(np.stack([snap_u, snap_v, snap_w], axis=-1))   # (N,N,N,3)

    return np.array(snapshots, dtype=np.float32)   # (T, N, N, N, 3)


print(f"[DATA]  Simulating 3D N-S (grid={N_GRID}³) ...")
U_tns = simulate_ns(N_GRID, N_T, DT, BURN)   # (T, N, N, N, 3)
T_snap = U_tns.shape[0]
print(f"[DATA]  T={T_snap} snapshots")

# Flatten to (T * N³, 3) for pointwise processing
U_flat = U_tns.reshape(T_snap, -1, 3)   # (T, N³, 3)

# ---------------------------------------------------------------------------
# 3. Koopman feature library
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "u1", "u2", "u3",
    "ω1", "ω2", "ω3",
    "|u|²", "|ω|²", "u·ω",
    "Δu1", "Δu2", "Δu3",
]
N_FEATURES = 12


def spectral_curl(u_hat, v_hat, w_hat, kv):
    """Return vorticity (ω_x, ω_y, ω_z) in physical space."""
    kx, ky, kz = kv
    ox = np.real(np.fft.ifftn(1j * (ky * w_hat - kz * v_hat)))
    oy = np.real(np.fft.ifftn(1j * (kz * u_hat - kx * w_hat)))
    oz = np.real(np.fft.ifftn(1j * (kx * v_hat - ky * u_hat)))
    return ox, oy, oz


def spectral_laplacian_vec(u_hat, v_hat, w_hat, kv):
    """Return Laplacian of (u, v, w) in physical space."""
    kx, ky, kz = kv
    k2  = kx**2 + ky**2 + kz**2
    lu  = np.real(np.fft.ifftn(-k2 * u_hat))
    lv  = np.real(np.fft.ifftn(-k2 * v_hat))
    lw  = np.real(np.fft.ifftn(-k2 * w_hat))
    return lu, lv, lw


def build_ns_library(U_snap_batch: np.ndarray) -> np.ndarray:
    """Build feature matrix for a batch of snapshots.

    Parameters
    ----------
    U_snap_batch : np.ndarray, shape (T, N, N, N, 3)

    Returns
    -------
    Theta : np.ndarray, shape (T * N³, 12)
    """
    kv = make_wavenumbers(N_GRID, L)
    rows = []
    for t in range(U_snap_batch.shape[0]):
        u = U_snap_batch[t, :, :, :, 0]
        v = U_snap_batch[t, :, :, :, 1]
        w = U_snap_batch[t, :, :, :, 2]

        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)

        ox, oy, oz = spectral_curl(u_hat, v_hat, w_hat, kv)
        lu, lv, lw = spectral_laplacian_vec(u_hat, v_hat, w_hat, kv)

        # Flatten to (N³,)
        N3 = N_GRID ** 3
        u_f  = u.ravel();  v_f = v.ravel();  w_f = w.ravel()
        ox_f = ox.ravel(); oy_f = oy.ravel(); oz_f = oz.ravel()
        lu_f = lu.ravel(); lv_f = lv.ravel(); lw_f = lw.ravel()

        u_sq = u_f**2 + v_f**2 + w_f**2
        o_sq = ox_f**2 + oy_f**2 + oz_f**2
        u_dot_o = u_f*ox_f + v_f*oy_f + w_f*oz_f

        rows.append(np.column_stack([
            u_f, v_f, w_f,
            ox_f, oy_f, oz_f,
            u_sq, o_sq, u_dot_o,
            lu_f, lv_f, lw_f,
        ]))

    return np.vstack(rows).astype(np.float32)   # (T * N³, 12)


# Time derivatives via central differences
U_inner = U_tns[1:-1]
U_dot   = (U_tns[2:] - U_tns[:-2]) / (2.0 * DT)   # (T-2, N, N, N, 3)
U_dot_flat = U_dot.reshape(-1, N_GRID**3, 3).reshape(-1, 3)  # (T_inner*N³, 3)

print("[DATA]  Building feature library (spectral) ...")
Theta = build_ns_library(U_inner)   # (T_inner * N³, 12)
print(f"[DATA]  Theta shape: {Theta.shape}  U_dot shape: {U_dot_flat.shape}")

# ---------------------------------------------------------------------------
# 4. KANDy model  (KAN = [12, 3])
# ---------------------------------------------------------------------------
ns_lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="ns_lift")

model = KANDy(
    lift=ns_lift,
    grid=5,
    k=3,
    steps=500,
    seed=SEED,
)

model.fit(
    X=Theta,
    X_dot=U_dot_flat,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas for (u_t, v_t, w_t) ...")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=2)
    for lab, f in zip(["u_t", "v_t", "w_t"], formulas):
        print(f"  {lab} = {f}")
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 6. Evaluation — point-wise MSE on test set
# ---------------------------------------------------------------------------
N_total = len(Theta)
n_test  = int(N_total * 0.15)
Th_test = Theta[N_total - n_test:]
Ud_test = U_dot_flat[N_total - n_test:]

Ud_pred = model.predict(Th_test)
mse     = np.mean((Ud_pred - Ud_test) ** 2)
print(f"\n[EVAL]  Test MSE: {mse:.6e}   RMSE: {mse**0.5:.6e}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/NavierStokes", exist_ok=True)

# 7a. Predicted vs true u_t scatter (subsample)
n_plt = min(5000, len(Ud_test))
idx   = np.random.choice(len(Ud_test), n_plt, replace=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
comp_names = ["u_t", "v_t", "w_t"]
for ci, ax in enumerate(axes):
    ax.scatter(Ud_test[idx, ci], Ud_pred[idx, ci],
               s=1.0, alpha=0.3, rasterized=True)
    lims = [min(Ud_test[idx, ci].min(), Ud_pred[idx, ci].min()),
            max(Ud_test[idx, ci].max(), Ud_pred[idx, ci].max())]
    ax.plot(lims, lims, "r--", lw=0.8)
    ax.set_xlabel(f"True {comp_names[ci]}")
    ax.set_ylabel(f"Predicted {comp_names[ci]}")
    ax.set_title(comp_names[ci])
    ax.grid(alpha=0.3, linestyle="--")
fig.suptitle("N-S: predicted vs true point-wise derivatives", fontsize=11)
fig.tight_layout()
fig.savefig("results/NavierStokes/pred_vs_true.png", dpi=300, bbox_inches="tight")
fig.savefig("results/NavierStokes/pred_vs_true.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Loss curves
if hasattr(model, "train_results_") and model.train_results_:
    fig, ax = plot_loss_curves(
        model.train_results_,
        title="N-S training loss",
        save="results/NavierStokes/loss_curves",
    )
    plt.close(fig)

# 7c. Edge activations (subsample)
n_sub   = min(4000, int(N_total * 0.70))
sub_idx = np.random.choice(int(N_total * 0.70), n_sub, replace=False)
train_t = torch.tensor(Theta[sub_idx], dtype=torch.float32)
fig = plot_all_edges(
    model.model_,
    X=train_t,
    input_names=FEATURE_NAMES,
    output_names=["u_t", "v_t", "w_t"],
    title="N-S KAN edge activations",
    save="results/NavierStokes/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/NavierStokes/")
