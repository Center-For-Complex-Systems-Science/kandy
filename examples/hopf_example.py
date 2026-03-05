#!/usr/bin/env python3
"""KANDy example: Hopf fibration  (S³ → S²).

The Hopf fibration maps unit quaternions q = (x1, x2, x3, x4) ∈ S³ to
points on the 2-sphere S² via:
    p1 = 2*(x1*x3 + x2*x4)
    p2 = 2*(x2*x3 - x1*x4)
    p3 = x1² + x2² - x3² - x4²

Two KANDy models are trained:
  (A) Raw model  — lift is identity on S³ (4 inputs), KAN = [4, 3]
  (B) Engineered — lift is the five bilinear/quadratic Hopf features, KAN = [5, 3]

Model B demonstrates how a hand-crafted lift that pre-encodes all cross-terms
recovers the exact structure of the Hopf map.
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
# 1. Data generation — uniform sampling on S³
# ---------------------------------------------------------------------------
N_SAMPLES = 20_000


def sample_s3(n: int, rng=None) -> np.ndarray:
    """Uniform random points on the 3-sphere via Gaussian normalisation."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def hopf_map(q: np.ndarray) -> np.ndarray:
    """True Hopf fibration  S³ -> S²."""
    x1, x2, x3, x4 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    p1 = 2.0 * (x1 * x3 + x2 * x4)
    p2 = 2.0 * (x2 * x3 - x1 * x4)
    p3 = x1**2 + x2**2 - x3**2 - x4**2
    return np.column_stack([p1, p2, p3]).astype(np.float32)


rng = np.random.default_rng(SEED)
Q   = sample_s3(N_SAMPLES, rng=rng)   # (N, 4) — points on S³
P   = hopf_map(Q)                     # (N, 3) — target on S²

print(f"[DATA]  N={N_SAMPLES} samples on S³, target dim=3 (S²)")
print(f"        S³ radius check: max |q|={np.linalg.norm(Q, axis=1).max():.6f}")
print(f"        S² radius check: max |p|={np.linalg.norm(P, axis=1).max():.6f}")

# ---------------------------------------------------------------------------
# 2A. Raw lift  phi: R^4 -> R^4  (identity)
# ---------------------------------------------------------------------------
RAW_FEATURE_NAMES = ["x1", "x2", "x3", "x4"]
raw_lift = CustomLift(fn=lambda X: X, output_dim=4, name="identity_s3")

# ---------------------------------------------------------------------------
# 2B. Engineered lift  phi: R^4 -> R^5
#     theta = [x1*x3 + x2*x4,  x2*x3 - x1*x4,  x1²+x2²-x3²-x4²,  x1²+x2²,  x3²+x4²]
#
#  The first three are exactly the Hopf components (up to the factor of 2).
#  Adding the Hopf-fiber invariants (x1²+x2² and x3²+x4²) is optional but
#  helps the KAN learn the exact linear map on the first three features.
# ---------------------------------------------------------------------------
ENG_FEATURE_NAMES = ["x1x3+x2x4", "x2x3-x1x4", "x1sq+x2sq-x3sq-x4sq",
                     "x1sq+x2sq", "x3sq+x4sq"]


def engineered_hopf_lift(X: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    f1 = x1 * x3 + x2 * x4
    f2 = x2 * x3 - x1 * x4
    f3 = x1**2 + x2**2 - x3**2 - x4**2
    f4 = x1**2 + x2**2
    f5 = x3**2 + x4**2
    return np.column_stack([f1, f2, f3, f4, f5])


eng_lift = CustomLift(fn=engineered_hopf_lift, output_dim=5, name="hopf_engineered")

# ---------------------------------------------------------------------------
# 3. Train both models
# ---------------------------------------------------------------------------
# In "map mode" (like Hénon), X is the input and P is the "derivative" (target).
# We reuse KANDy.fit() with X=Q and X_dot=P.

print("\n--- Model A: raw identity lift (KAN=[4,3]) ---")
model_raw = KANDy(lift=raw_lift, grid=5, k=3, steps=500, seed=SEED)
model_raw.fit(X=Q, X_dot=P, val_frac=0.15, test_frac=0.15, lamb=0.0)

print("\n--- Model B: engineered Hopf lift (KAN=[5,3]) ---")
model_eng = KANDy(lift=eng_lift, grid=5, k=3, steps=500, seed=SEED)
model_eng.fit(X=Q, X_dot=P, val_frac=0.15, test_frac=0.15, lamb=0.0)

# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
N         = len(Q)
n_test    = int(N * 0.15)
Q_test    = Q[N - n_test:]
P_test    = P[N - n_test:]

pred_raw  = model_raw.predict(Q_test)
pred_eng  = model_eng.predict(Q_test)

rmse_raw  = np.sqrt(np.mean((pred_raw - P_test) ** 2))
rmse_eng  = np.sqrt(np.mean((pred_eng - P_test) ** 2))
print(f"\n[EVAL]  Raw model RMSE:        {rmse_raw:.6f}")
print(f"[EVAL]  Engineered model RMSE: {rmse_eng:.6f}")

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
for name, mdl, fnames in [
    ("Raw",        model_raw, RAW_FEATURE_NAMES),
    ("Engineered", model_eng, ENG_FEATURE_NAMES),
]:
    print(f"\n[SYMBOLIC] {name} model:")
    try:
        formulas = mdl.get_formula(var_names=fnames, round_places=2)
        for comp, f in zip(["p1", "p2", "p3"], formulas):
            print(f"  {comp} = {f}")
    except Exception as exc:
        print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Hopf", exist_ok=True)

# 6a. S² scatter — true vs predicted (engineered model)
fig = plt.figure(figsize=(10, 4))
for col_idx, (pts, title) in enumerate([
    (P_test,    "True S² (Hopf map)"),
    (pred_eng,  "KANDy (engineered lift)"),
]):
    ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=1.0, alpha=0.3, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("p1"); ax.set_ylabel("p2"); ax.set_zlabel("p3")
fig.suptitle("Hopf Fibration  S³ → S²", fontsize=12)
fig.tight_layout()
fig.savefig("results/Hopf/s2_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Hopf/s2_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. RMSE bar chart
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["Raw lift\n[4→3]", "Engineered lift\n[5→3]"],
       [rmse_raw, rmse_eng], color=["#1f77b4", "#2ca02c"], width=0.5)
ax.set_ylabel("Test RMSE")
ax.set_title("Hopf: KAN model comparison")
ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout()
fig.savefig("results/Hopf/rmse_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Hopf/rmse_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Edge activations for engineered model
train_theta = torch.tensor(
    engineered_hopf_lift(Q[: int(N * 0.70)]), dtype=torch.float32
)
fig = plot_all_edges(
    model_eng.model_,
    X=train_theta,
    input_names=ENG_FEATURE_NAMES,
    output_names=["p1", "p2", "p3"],
    title="Hopf (engineered) KAN edge activations",
    save="results/Hopf/edge_activations_eng",
)
plt.close(fig)

print("[FIGS]  Saved results/Hopf/")
