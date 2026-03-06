I'll now provide the complete final deliverables for both tasks based on all investigation and diagnostics performed.

---

## `outputs/kane_lorenz.py`

```python
"""
KANELift Lorenz-63 Experiment
==============================================
Investigates whether KANELift (KAN autoencoder Koopman lift) can learn an
interpretable, accurate Koopman lift phi(x) for the Lorenz-63 system.

Key fix: KANELift trains on GPU if available; we explicitly move all
sub-modules to CPU before calling KANDy (which operates in numpy/CPU space).

Steps:
  a) Generate Lorenz-63 trajectory (N=8000, dt=0.01, burn_in=500)
  b) Train KANELift (latent_dim=8, single-layer encoder)
  c) Plot & save loss curves -> results/KANE/loss_curves.png
  d) Populate encoder activations & extract symbolic formulas
  e) Plug into KANDy(lift=lift, grid=5, k=3).fit(X, X_dot)
  f) Rollout(x0, T=2000, dt=0.01) and compute RMSE
  g) Compare to PolynomialLift(degree=2)
  h) Print eigenvalues of lift.K_.weight (Koopman spectrum)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Output directories ─────────────────────────────────────────────────────
os.makedirs("results/KANE", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── KANDy imports ──────────────────────────────────────────────────────────
from kandy import KANELift, KANDy, PolynomialLift

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def get_lift_device(lift):
    """Return the torch device of the KANELift encoder parameters."""
    try:
        return next(lift.encoder_.parameters()).device
    except Exception:
        return torch.device("cpu")


def move_lift_to_cpu(lift):
    """Move all KANELift sub-modules to CPU in-place."""
    moved = []
    for attr in ("encoder_", "decoder_", "K_"):
        if hasattr(lift, attr):
            try:
                getattr(lift, attr).to("cpu")
                moved.append(attr)
            except Exception as e:
                print(f"    Could not move {attr} to CPU: {e}")
    return moved


# ──────────────────────────────────────────────────────────────────────────
# STEP A: Generate Lorenz-63 trajectory
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP A: Generating Lorenz-63 trajectory")
print("=" * 60)

sigma_L = 10.0
rho_L   = 28.0
beta_L  = 8.0 / 3.0


def lorenz_rhs(state):
    x, y, z = state
    return np.array([
        sigma_L * (y - x),
        x * (rho_L - z) - y,
        x * y - beta_L * z,
    ])


def rk4_step(s, h):
    k1 = lorenz_rhs(s)
    k2 = lorenz_rhs(s + 0.5 * h * k1)
    k3 = lorenz_rhs(s + 0.5 * h * k2)
    k4 = lorenz_rhs(s + h * k3)
    return s + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


N       = 8000
dt      = 0.01
burn_in = 500
total   = N + burn_in

np.random.seed(42)
state     = np.array([1.0, 1.0, 1.0])
traj_full = np.zeros((total, 3))
for i in range(total):
    traj_full[i] = state
    state = rk4_step(state, dt)

traj    = traj_full[burn_in:].astype("float32")   # (8000, 3)
X_state = traj[:-1]                                # (7999, 3)
X_next  = traj[1:]                                 # (7999, 3)
X_dot   = (X_next - X_state) / dt                 # (7999, 3)  finite-diff

print(f"  Trajectory shape : {traj.shape}")
print(f"  X_state          : {X_state.shape}")
print(f"  X_dot            : {X_dot.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP B: Train KANELift (latent_dim=8, single-layer encoder)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP B: Training KANELift (latent_dim=8)")
print("=" * 60)

lift = KANELift(latent_dim=8, hidden_dim=None, grid=5, k=3)

lift.train_koopman(
    traj,
    dt=dt,
    epochs=100,
    lr=5e-4,
    batch_size=512,
    alpha_latent=1.0,
    beta_recon=0.5,
    gamma_predx=1.0,
)

history     = lift.train_history_
train_total = history.get("train_total", [])
val_total   = history.get("val_total",   [])

if train_total:
    print(f"\n  Final train_total : {train_total[-1]:.6f}")
else:
    print("  No train loss recorded.")
if val_total:
    print(f"  Final val_total   : {val_total[-1]:.6f}")
else:
    print("  No val loss recorded.")

# ── Move everything to CPU for downstream numpy/KANDy operations ──────────
lift_device = get_lift_device(lift)
print(f"\n  Encoder device after training: {lift_device}")
if str(lift_device) != "cpu":
    moved = move_lift_to_cpu(lift)
    print(f"  Moved to CPU: {moved}")
else:
    print("  Already on CPU.")

# ──────────────────────────────────────────────────────────────────────────
# STEP C: Save loss curves
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP C: Saving loss curves -> results/KANE/loss_curves.png")
print("=" * 60)

plot_keys = ["train_total", "val_total",
             "train_predx", "train_latent", "train_recon",
             "val_predx",   "val_latent",   "val_recon"]
plot_data = {k: history[k] for k in plot_keys if k in history and history[k]}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax0 = axes[0]
for k in ["train_total", "val_total"]:
    if k in plot_data:
        ax0.semilogy(plot_data[k], label=k)
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Loss (log scale)")
ax0.legend()
ax0.set_title("Total Loss")
ax0.grid(True, alpha=0.3)

ax1 = axes[1]
for k in ["train_predx", "train_latent", "train_recon"]:
    if k in plot_data:
        ax1.semilogy(plot_data[k], label=k)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)")
ax1.legend()
ax1.set_title("Component Losses (train)")
ax1.grid(True, alpha=0.3)

fig.suptitle("KANELift Lorenz-63 Training", fontsize=13)
fig.tight_layout()
fig.savefig("results/KANE/loss_curves.png", dpi=300)
plt.close(fig)
print("  Saved results/KANE/loss_curves.png")

# ──────────────────────────────────────────────────────────────────────────
# STEP D: Populate activations & extract symbolic formulas
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP D: Extracting symbolic encoder formulas")
print("=" * 60)

formulas       = None
formula_status = "not attempted"

try:
    lift.encoder_.save_act = True
    # Encoder is now CPU; ensure sample is on CPU too
    X_sample_t = torch.tensor(X_state[:512], dtype=torch.float32)
    with torch.no_grad():
        _ = lift.encoder_(X_sample_t)
    formulas = lift.get_formula(var_names=["x", "y", "z"])
    formula_status = "success"
    print("  Symbolic encoder formulas:")
    for i, f in enumerate(formulas):
        print(f"    z_{i} = {f}")
except Exception as e:
    formula_status = f"failed: {e}"
    print(f"  get_formula() {formula_status}")
    print("  NOTE: This is expected if spline activations cannot be auto-symbolified.")
    print("        Increase epochs or use auto_symbolic_with_costs for better coverage.")

# ──────────────────────────────────────────────────────────────────────────
# STEP E: KANDy(lift=KANELift) — derivative supervision
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP E: KANDy(lift=KANELift) derivative supervision")
print("=" * 60)

kandy_kane        = None
kandy_kane_fitted = False

for opt_name, opt_kwargs in [
    ("LBFGS", dict(opt="LBFGS", lr=1.0)),
    ("Adam",  dict(opt="Adam",  lr=2e-3, batch=4096)),
]:
    try:
        kandy_kane = KANDy(lift=lift, grid=5, k=3)
        kandy_kane.fit(X_state, X_dot, **opt_kwargs)
        print(f"  KANDy(KANELift) fit OK using {opt_name}.")
        kandy_kane_fitted = True
        break
    except Exception as e:
        print(f"  {opt_name} failed: {e}")

if not kandy_kane_fitted:
    print("  WARNING: KANDy(KANELift) could not be fit. "
          "Check device placement or encoder stability.")

# ──────────────────────────────────────────────────────────────────────────
# STEP F: Rollout — KANELift
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP F: Rollout and RMSE — KANELift")
print("=" * 60)

T_rollout     = 2000
x0            = X_state[0].copy()
true_traj_ref = X_state[:T_rollout]

rmse_kane      = None
pred_traj_kane = None

if kandy_kane_fitted:
    try:
        pred_traj_kane = kandy_kane.rollout(x0, T=T_rollout, dt=dt)
        L = min(len(pred_traj_kane), len(true_traj_ref))
        rmse_kane = float(np.sqrt(np.mean(
            (pred_traj_kane[:L] - true_traj_ref[:L]) ** 2
        )))
        print(f"  KANELift rollout RMSE (T={T_rollout}, dt={dt}): {rmse_kane:.6f}")
    except Exception as e:
        print(f"  KANELift rollout FAILED: {e}")
else:
    print("  Skipped (fit failed).")

# ──────────────────────────────────────────────────────────────────────────
# STEP G: PolynomialLift(degree=2) baseline
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP G: PolynomialLift(degree=2) baseline")
print("=" * 60)

poly_lift      = PolynomialLift(degree=2)
kandy_poly     = None
poly_fitted    = False
rmse_poly      = None
pred_traj_poly = None

for opt_name, opt_kwargs in [
    ("LBFGS", dict(opt="LBFGS", lr=1.0)),
    ("Adam",  dict(opt="Adam",  lr=2e-3, batch=4096)),
]:
    try:
        kandy_poly = KANDy(lift=poly_lift, grid=5, k=3)
        kandy_poly.fit(X_state, X_dot, **opt_kwargs)
        print(f"  PolynomialLift(2) fit OK using {opt_name}.")
        poly_fitted = True
        break
    except Exception as e:
        print(f"  {opt_name} failed: {e}")

if poly_fitted:
    try:
        pred_traj_poly = kandy_poly.rollout(x0, T=T_rollout, dt=dt)
        L2 = min(len(pred_traj_poly), len(true_traj_ref))
        rmse_poly = float(np.sqrt(np.mean(
            (pred_traj_poly[:L2] - true_traj_ref[:L2]) ** 2
        )))
        print(f"  PolynomialLift(2) rollout RMSE (T={T_rollout}, dt={dt}): {rmse_poly:.6f}")
    except Exception as e:
        print(f"  PolynomialLift rollout FAILED: {e}")

# ──────────────────────────────────────────────────────────────────────────
# STEP H: Koopman eigenvalue spectrum
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP H: Koopman eigenvalue spectrum (lift.K_.weight)")
print("=" * 60)

eigenvalues_sorted = []
eig_table_rows     = []

try:
    K_mat = lift.K_.weight.detach().cpu().numpy()
    print(f"  K matrix shape: {K_mat.shape}")
    eigs = np.linalg.eigvals(K_mat)
    eigenvalues_sorted = sorted(eigs, key=lambda v: -abs(v))
    print("  Eigenvalues (sorted by |λ|):")
    for i, ev in enumerate(eigenvalues_sorted):
        mag   = abs(ev)
        angle = np.angle(ev)
        freq  = angle / (2.0 