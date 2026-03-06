Looking at the results, the key issue is a **CUDA/CPU device mismatch** — the KANELift encoder trained on GPU but inference is attempted on CPU. I need to fix the script to explicitly move tensors to CPU, and also fix the `plot_loss_curves` API call. Let me provide the complete final deliverables.

---

## Final Answer

### `outputs/kane_lorenz.py`

```python
"""
KANELift Lorenz-63 Experiment
==============================================
Investigates whether KANELift (KAN autoencoder Koopman lift) can learn an
interpretable, accurate Koopman lift phi(x) for the Lorenz-63 system.

Steps:
  a) Generate Lorenz-63 trajectory (N=8000, dt=0.01, burn_in=500)
  b) Train KANELift (latent_dim=8, single-layer encoder)
  c) Plot & save loss curves → results/KANE/loss_curves.png
  d) Populate encoder activations & extract symbolic formulas
  e) Plug into KANDy(lift=lift, grid=5, k=3).fit(X, X_dot)
  f) Rollout(x0, T=2000, dt=0.01) and compute RMSE
  g) Compare to PolynomialLift(degree=2)
  h) Print eigenvalues of lift.K_.weight
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
# Helper: detect device the KANELift encoder lives on and move data there
# ──────────────────────────────────────────────────────────────────────────

def get_lift_device(lift):
    """Return the torch device of the KANELift encoder parameters."""
    try:
        return next(lift.encoder_.parameters()).device
    except Exception:
        return torch.device("cpu")


def cpu_numpy(t):
    """Safely convert a tensor to a numpy array on CPU."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


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

def rk4_step(state, dt):
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

N       = 8000
dt      = 0.01
burn_in = 500
total   = N + burn_in

np.random.seed(42)
state = np.array([1.0, 1.0, 1.0])
traj_full = np.zeros((total, 3))
for i in range(total):
    traj_full[i] = state
    state = rk4_step(state, dt)

traj    = traj_full[burn_in:]           # (8000, 3)
X_state = traj[:-1].astype("float32")  # (7999, 3)
X_next  = traj[1:].astype("float32")   # (7999, 3)
X_dot   = ((X_next - X_state) / dt)    # finite-diff derivatives

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
    traj.astype("float32"),
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

print(f"\n  Final train_total : {train_total[-1]:.6f}" if train_total else "  No train loss.")
print(f"  Final val_total   : {val_total[-1]:.6f}"   if val_total   else "  No val   loss.")

# Move lift to CPU for all downstream numpy operations
lift_device = get_lift_device(lift)
print(f"  Lift encoder device: {lift_device}")

# Move the whole lift to CPU so KANDy (CPU-side) can call it
if str(lift_device) != "cpu":
    try:
        lift.encoder_.to("cpu")
        lift.decoder_.to("cpu")
        lift.K_.to("cpu")
        print("  Moved lift to CPU.")
    except Exception as e:
        print(f"  Could not move lift to CPU: {e}")

# ──────────────────────────────────────────────────────────────────────────
# STEP C: Save loss curves
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP C: Saving loss curves → results/KANE/loss_curves.png")
print("=" * 60)

plot_keys = ["train_total", "val_total",
             "train_predx", "train_latent", "train_recon",
             "val_predx",   "val_latent",   "val_recon"]
plot_data = {k: history[k] for k in plot_keys if k in history and history[k]}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: total losses
ax0 = axes[0]
for k in ["train_total", "val_total"]:
    if k in plot_data:
        ax0.plot(plot_data[k], label=k)
ax0.set_xlabel("Epoch"); ax0.set_ylabel("Loss"); ax0.set_yscale("log")
ax0.legend(); ax0.set_title("Total Loss")

# Right: component losses
ax1 = axes[1]
for k in ["train_predx", "train_latent", "train_recon"]:
    if k in plot_data:
        ax1.plot(plot_data[k], label=k)
ax1.set_xlabel("Epoch"); ax1.set_yscale("log")
ax1.legend(); ax1.set_title("Component Losses (train)")

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

formulas = None
formula_status = "not attempted"
try:
    lift.encoder_.save_act = True
    X_sample_t = torch.tensor(X_state[:512], dtype=torch.float32)  # CPU
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

# ──────────────────────────────────────────────────────────────────────────
# STEP E: KANDy(lift=KANELift) — derivative supervision
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP E: KANDy(lift=KANELift) derivative supervision")
print("=" * 60)

kandy_kane_fitted = False
kandy_kane = None

for opt_name, opt_kwargs in [
    ("LBFGS", dict(opt="LBFGS", lr=1.0)),
    ("Adam",  dict(opt="Adam",  lr=2e-3, batch=4096)),
]:
    try:
        kandy_kane = KANDy(lift=lift, grid=5, k=3)
        kandy_kane.fit(X_state, X_dot, **opt_kwargs)
        print(f"  KANDy fit OK ({opt_name}).")
        kandy_kane_fitted = True
        break
    except Exception as e:
        print(f"  {opt_name} failed: {e}")

# ──────────────────────────────────────────────────────────────────────────
# STEP F: Rollout — KANELift
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP F: Rollout and RMSE — KANELift")
print("=" * 60)

T_rollout     = 2000
x0            = X_state[0]
true_traj_ref = X_state[:T_rollout]

rmse_kane        = None
pred_traj_kane   = None
kane_rollout_err = "not run"

if kandy_kane_fitted:
    try:
        pred_traj_kane = kandy_kane.rollout(x0, T=T_rollout, dt=dt)
        L = min(len(pred_traj_kane), len(true_traj_ref))
        rmse_kane = float(np.sqrt(np.mean((pred_traj_kane[:L] - true_traj_ref[:L])**2)))
        kane_rollout_err = "ok"
        print(f"  KANELift rollout RMSE (T={T_rollout}): {rmse_kane:.6f}")
    except Exception as e:
        kane_rollout_err = str(e)
        print(f"  KANELift rollout FAILED: {e}")
else:
    print("  Skipped (KANDy fit failed).")

# ──────────────────────────────────────────────────────────────────────────
# STEP G: PolynomialLift(degree=2) baseline
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP G: PolynomialLift(degree=2) baseline")
print("=" * 60)

poly_lift       = PolynomialLift(degree=2)
kandy_poly      = None
poly_fitted     = False
rmse_poly       = None
pred_traj_poly  = None
poly_rollout_err = "not run"

for opt_name, opt_kwargs in [
    ("LBFGS", dict(opt="LBFGS", lr=1.0)),
    ("Adam",  dict(opt="Adam",  lr=2e-3, batch=4096)),
]:
    try:
        kandy_poly = KANDy(lift=poly_lift, grid=5, k=3)
        kandy_poly.fit(X_state, X_dot, **opt_kwargs)
        print(f"  PolynomialLift fit OK ({opt_name}).")
        poly_fitted = True
        break
    except Exception as e:
        print(f"  {opt_name} failed: {e}")

if poly_fitted:
    try:
        pred_traj_poly = kandy_poly.rollout(x0, T=T_rollout, dt=dt)
        L2 = min(len(pred_traj_poly), len(true_traj_ref))
        rmse_poly = float(np.sqrt(np.mean((pred_traj_poly[:L2] - true_traj_ref[:L2])**2)))
        poly_rollout_err = "ok"
        print(f"  PolynomialLift(2) rollout RMSE (T={T_rollout}): {rmse_poly:.6f}")
    except Exception as e:
        poly_rollout_err = str(e)
        print(f"  PolynomialLift rollout FAILED: {e}")

# ──────────────────────────────────────────────────────────────────────────
# STEP H: Koopman eigenvalue spectrum
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP H: Koopman eigenvalue spectrum")
print("=" * 60)

eigenvalues_sorted = []
try:
    K_mat = lift.K_.weight.detach().cpu().numpy()
    print(f"  K matrix shape: {K_mat.shape}")
    eigs = np.linalg.eigvals(K_mat)
    eigenvalues_sorted = sorted(eigs, key=lambda v: -abs(v))
    print("  Eigenvalues (sorted by |λ|):")
    for i, ev in enumerate(eigenvalues_sorted):
        mag   = abs(ev)
        angle = np.angle(ev)
        freq  = angle / (2 * np.pi * dt)
        print(f"    λ_{i}: {ev.real:+.6f}{ev.imag:+.6f}j  "
              f"|λ|={mag:.6f}  freq≈{freq:.3f} Hz")
except Exception as e:
    print(f"  Could not extract K_ eigenvalues: {e}")

# ──────────────────────────────────────────────────