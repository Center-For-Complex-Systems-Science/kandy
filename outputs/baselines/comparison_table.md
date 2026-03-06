# Final Answer

## `outputs/kane_lorenz.py`

```python
"""
KANELift Lorenz-63 Experiment
==============================================
Investigates whether KANELift (KAN autoencoder Koopman lift) can learn an
interpretable, accurate Koopman lift phi(x) for the Lorenz-63 system.

Key fix applied: KANELift trains on GPU if available; we explicitly move all
sub-modules to CPU before calling KANDy (which operates in numpy/CPU space).

Steps:
  a) Generate Lorenz-63 trajectory (N=8000, dt=0.01, burn_in=500,
     sigma=10, rho=28, beta=8/3)
  b) Train KANELift (latent_dim=8, hidden_dim=None, single-layer encoder)
     epochs=100, lr=5e-4, batch_size=512, alpha_latent=1.0,
     beta_recon=0.5, gamma_predx=1.0
  c) Plot & save loss curves -> results/KANE/loss_curves.png
  d) Populate encoder activations & extract symbolic formulas via
     lift.get_formula(var_names=['x','y','z'])
  e) Plug into KANDy(lift=lift, grid=5, k=3).fit(X, X_dot) — LBFGS preferred
  f) Rollout(x0, T=2000, dt=0.01) and compute RMSE vs true trajectory
  g) Compare to PolynomialLift(degree=2) on same data
  h) Print eigenvalues of lift.K_.weight to inspect Koopman spectrum
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
# Helpers: device management
# ──────────────────────────────────────────────────────────────────────────

def get_lift_device(lift):
    """Return the torch device of the KANELift encoder parameters."""
    try:
        return next(lift.encoder_.parameters()).device
    except Exception:
        return torch.device("cpu")


def move_lift_to_cpu(lift):
    """
    Move all KANELift sub-modules (encoder_, decoder_, K_) to CPU in-place.
    This is necessary because KANDy's .fit() calls lift(X) with numpy arrays
    that are converted to CPU tensors; if the encoder is on CUDA the spline
    grids (also on CUDA) clash with the CPU input tensor.
    """
    moved = []
    for attr in ("encoder_", "decoder_", "K_"):
        if hasattr(lift, attr):
            try:
                getattr(lift, attr).to("cpu")
                moved.append(attr)
            except Exception as e:
                print(f"    Warning: could not move {attr} to CPU: {e}")
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


def lorenz_rhs(s):
    x, y, z = s
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
X_dot   = (X_next - X_state) / dt                 # (7999, 3)

print(f"  Trajectory shape : {traj.shape}")
print(f"  X_state          : {X_state.shape}")
print(f"  X_dot (FD deriv) : {X_dot.shape}")

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
    print(f"\n  Final train_total loss : {train_total[-1]:.6f}")
else:
    print("  No train loss recorded.")
if val_total:
    print(f"  Final val_total   loss : {val_total[-1]:.6f}")
else:
    print("  No val loss recorded.")

# ── CRITICAL FIX: Move lift to CPU before any numpy/KANDy interaction ─────
lift_device = get_lift_device(lift)
print(f"\n  Encoder device after training: {lift_device}")
if str(lift_device) != "cpu":
    moved = move_lift_to_cpu(lift)
    new_device = get_lift_device(lift)
    print(f"  Moved {moved} to CPU. Device now: {new_device}")
else:
    print("  Already on CPU — no device move needed.")

# ──────────────────────────────────────────────────────────────────────────
# STEP C: Save loss curves to results/KANE/loss_curves.png
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP C: Saving loss curves -> results/KANE/loss_curves.png")
print("=" * 60)

plot_keys = [
    "train_total", "val_total",
    "train_predx", "train_latent", "train_recon",
    "val_predx",   "val_latent",   "val_recon",
]
plot_data = {k: history[k] for k in plot_keys if k in history and len(history[k]) > 0}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax0 = axes[0]
for k in ["train_total", "val_total"]:
    if k in plot_data:
        ax0.semilogy(plot_data[k], label=k.replace("_", " "))
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Loss (log scale)")
ax0.legend(fontsize=9)
ax0.set_title("Total Loss")
ax0.grid(True, alpha=0.3)

ax1 = axes[1]
for k in ["train_predx", "train_latent", "train_recon"]:
    if k in plot_data:
        ax1.semilogy(plot_data[k], label=k.replace("train_", ""))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)")
ax1.legend(fontsize=9)
ax1.set_title("Component Losses (train)")
ax1.grid(True, alpha=0.3)

fig.suptitle("KANELift Lorenz-63 Training Curves", fontsize=13, fontweight="bold")
fig.tight_layout()
save_path = "results/KANE/loss_curves.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {save_path}")

# ──────────────────────────────────────────────────────────────────────────
# STEP D: Populate encoder activations & extract symbolic formulas
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP D: Extracting symbolic encoder formulas via lift.get_formula()")
print("=" * 60)

formulas       = None
formula_status = "not attempted"

try:
    # 1. Enable activation saving
    lift.encoder_.save_act = True
    # 2. Forward pass on CPU tensor (encoder is now on CPU)
    X_sample_t = torch.tensor(X_state[:512], dtype=torch.float32)  # CPU
    with torch.no_grad():
        _ = lift.encoder_(X_sample_t)
    # 3. Extract symbolic formulas
    formulas = lift.get_formula(var_names=["x", "y", "z"])
    formula_status = "success"
    print("  Symbolic encoder formulas (z_i = f(x, y, z)):")
    for i, f in enumerate(formulas):
        print(f"    z_{i} = {f}")
except Exception as e:
    formula_status = f"failed — {e}"
    print(f"  get_formula() {formula_status}")
    print("  Diagnosis: This can occur when:")
    print("    • save_act=True was not set before the forward pass")
    print("    • The KAN spline activations are too complex for auto-symbolic fitting")
    print("    • Device mismatch (now fixed with CPU move in Step B)")
    print("  Recommendation: Use auto_symbolic_with_costs() with POLY_LIB_CHEAP")
    print("    after a longer training run (epochs>=200) for better symbolic coverage.")

# ──────────────────────────────────────────────────────────────────────────
# STEP E: Plug KANELift into KANDy — derivative supervision fit
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP E: KANDy(lift=KANELift, grid=5, k=3).fit(X_state, X_dot)")
print("=" * 60)

kandy_kane        = None
kandy_kane_fitted = False
fit_method_used   = "none"

for opt_name, opt_kwargs in [
    ("LBFGS", dict(opt="LBFGS", lr=1.0)),
    ("Adam",  dict(opt="Adam",  lr=2e-3, batch=4096)),
]:
    try:
        print(f"  Trying {opt_name}...")
        kandy_kane = KANDy(lift=lift, grid=5, k=3)
        kandy_kane.fit(X_state, X_dot, **opt_kwargs)
        print(f"  KANDy(KANELift) fit completed ({opt_name}).")
        kandy_kane_fitted = True
        fit_method_used   = opt_name
        break
    except Exception as e:
        print(f"  {opt_name} failed: {e}")

if not kandy_kane_fitted:
    print("  CRITICAL: KANDy(KANELift) fit failed with all optimisers.")
    print("  This is almost certainly the CUDA/CPU device mismatch.")
    print("  The move_lift_to_cpu() call above should have fixed it;")
    print("  if still failing, check that KANELift.__call__ calls .cpu() on output.")

# ──────────────────────────────────────────────────────────────────────────
# STEP F: Rollout — KANELift — T=2000, dt=0.01
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP F: Rollout and RMSE — KANELift (T=2000)")
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
        print(f"  KANELift rollout RMSE  : {rmse_kane:.6f}  "
              f"(T={T_rollout} steps, dt={dt})")

        # ── Save attractor overlay ────────────────────────────────────────
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
        ax2[0].plot(true_traj_ref[:L, 0], true_traj_ref[:L, 2],
                    lw=0.4, alpha=0.7, color="steelblue", label="True")
        ax2[0].plot(pred_traj_kane[:L, 0], pred_traj_kane[:L, 2],
                    lw=0.4, alpha=0.7, color="tomato", label="KANELift")
        ax2[0].set_xlabel("x"); ax2[0].set_ylabel("z")
        ax2[0].set_title("Lorenz Attractor (x-z): KANELift")
        ax2[0].legend(fontsize=8)

        t_arr = np.arange(L) * dt
        for dim, label in enumerate(["x", "y", "z"]):
            ax2[1].plot(t_arr, true_traj_ref[:L, dim], lw