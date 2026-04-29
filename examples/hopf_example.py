#!/usr/bin/env python3
"""KANDy example: Hopf fibration  (S³ → S²).

The Hopf fibration maps unit quaternions q = (x1, x2, x3, x4) ∈ S³ to
points on the 2-sphere S² via:
    p1 = 2*(x1*x3 + x2*x4)
    p2 = 2*(x2*x3 - x1*x4)
    p3 = x1² + x2² - x3² - x4²

Two KANDy models are trained:
  (A) Raw model  — lift is identity on S³ (4 inputs), KAN = [4, 3]
  (B) Engineered — lift gives 5 bilinear features, KAN = [5, 3]

Model B demonstrates how a hand-crafted lift that pre-encodes all cross-terms
recovers the exact structure of the Hopf map.
"""

import os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kan import KAN

from kandy.training import fit_kan
from kandy.plotting import (
    plot_all_edges,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)

device = torch.device("cpu")
rbf = lambda x: torch.exp(-x**2)

# ---------------------------------------------------------------------------
# 1. Data generation — uniform sampling on S³
# ---------------------------------------------------------------------------
TRAIN_N = 20_000
TEST_N  = 5_000


def sample_s3(n, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(n, 4, generator=g, device=device)
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


def hopf_map(x4):
    x1, x2, x3, x4v = x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3]
    z1 = torch.complex(x1, x2)
    z2 = torch.complex(x3, x4v)
    w = z1 * torch.conj(z2)
    y1 = 2.0 * torch.real(w)
    y2 = 2.0 * torch.imag(w)
    y3 = torch.abs(z1)**2 - torch.abs(z2)**2
    return torch.stack([y1, y2, y3], dim=1)


X_train = sample_s3(TRAIN_N, seed=0)
X_test  = sample_s3(TEST_N, seed=1)
Y_train = hopf_map(X_train)
Y_test  = hopf_map(X_test)

print(f"[DATA]  Train: {TRAIN_N}, Test: {TEST_N}")
print(f"        S³ radius check: max |q|={X_train.norm(dim=1).max():.6f}")
print(f"        S² radius check: max |p|={Y_train.norm(dim=1).max():.6f}")

# ---------------------------------------------------------------------------
# 2A. Raw model — identity lift, label normalization
# ---------------------------------------------------------------------------
y_mu = Y_train.mean(dim=0, keepdim=True)
y_std = Y_train.std(dim=0, keepdim=True) + 1e-12
Y_train_n = (Y_train - y_mu) / y_std
Y_test_n = (Y_test - y_mu) / y_std

dataset_raw = {
    "train_input": X_train,
    "train_label": Y_train_n,
    "test_input":  X_test,
    "test_label":  Y_test_n,
}

# ---------------------------------------------------------------------------
# 2B. Engineered lift — 5 bilinear features, input normalization
#     [x1*x3, x2*x4, x2*x3, x1*x4, x1²+x2²-x3²-x4²]
# ---------------------------------------------------------------------------
ENG_FEATURE_NAMES = ["x1x3", "x2x4", "x2x3", "x1x4", "x1sq+x2sq-x3sq-x4sq"]


def hopf_features(X):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    u1 = x1 * x3
    u2 = x2 * x4
    u3 = x2 * x3
    u4 = x1 * x4
    u5 = x1**2 + x2**2 - x3**2 - x4**2
    return torch.stack([u1, u2, u3, u4, u5], dim=1)


Phi_train = hopf_features(X_train)
Phi_test = hopf_features(X_test)
phi_mu = Phi_train.mean(dim=0, keepdim=True)
phi_std = Phi_train.std(dim=0, keepdim=True) + 1e-12
Phi_train_n = (Phi_train - phi_mu) / phi_std
Phi_test_n = (Phi_test - phi_mu) / phi_std

dataset_phi = {
    "train_input": Phi_train_n,
    "train_label": Y_train,
    "test_input":  Phi_test_n,
    "test_label":  Y_test,
}

# ---------------------------------------------------------------------------
# 3. Train both models
# ---------------------------------------------------------------------------
print("\n--- Model A: raw identity lift (KAN=[4,3]) ---")
model_raw = KAN(width=[4, 3], grid=64, k=3, base_fun=rbf, seed=SEED)
fit_kan(model_raw, dataset_raw, steps=400, patience=0)

print("\n--- Model B: engineered Hopf lift (KAN=[5,3]) ---")
model_phi = KAN(width=[5, 3], grid=64, k=3, base_fun=rbf, seed=SEED)
fit_kan(model_phi, dataset_phi, steps=400, patience=0)

# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
with torch.no_grad():
    pred_raw = model_raw(X_test) * y_std + y_mu
    mse_raw = torch.mean((pred_raw - Y_test) ** 2).item()

    pred_phi = model_phi(Phi_test_n)
    mse_phi = torch.mean((pred_phi - Y_test) ** 2).item()

print(f"\n[EVAL]  Raw model MSE:        {mse_raw:.3e}   RMSE: {mse_raw**0.5:.6f}")
print(f"[EVAL]  Engineered model MSE: {mse_phi:.3e}   RMSE: {mse_phi**0.5:.6f}")

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
# ---------------------------------------------------------------------------
for name, mdl, data_in, fnames in [
    ("Raw", model_raw, X_test[:2000], ["x1", "x2", "x3", "x4"]),
    ("Engineered", model_phi, Phi_test_n[:2000], ENG_FEATURE_NAMES),
]:
    print(f"\n[SYMBOLIC] {name} model:")

    mdl.save_act = True
    with torch.no_grad():
        _ = mdl(data_in)
    mdl.auto_symbolic()
    exprs, inputs = mdl.symbolic_formula()
    import sympy as sp
    sub_map = {
        sp.Symbol(str(inputs[i])): sp.Symbol(fnames[i])
        for i in range(len(inputs))
    }
    for comp, expr_str in zip(["p1", "p2", "p3"], exprs):
        sym = sp.sympify(expr_str).xreplace(sub_map)
        print(f"  {comp} = {sp.expand(sym)}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Hopf", exist_ok=True)

# 6a. S² scatter — true vs predicted (engineered model)
pred_eng_np = pred_phi.numpy()
Y_test_np = Y_test.numpy()

fig = plt.figure(figsize=(10, 4))
for col_idx, (pts, title) in enumerate([
    (Y_test_np, "True S² (Hopf map)"),
    (pred_eng_np, "KANDy (engineered lift)"),
]):
    ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=1.0, alpha=0.3, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("p3")
fig.tight_layout()
fig.savefig("results/Hopf/s2_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Hopf/s2_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. RMSE bar chart
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["Raw lift\n[4→3]", "Engineered lift\n[5→3]"],
       [mse_raw**0.5, mse_phi**0.5], color=["#1f77b4", "#2ca02c"], width=0.5)
ax.set_ylabel("Test RMSE")
ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout()
fig.savefig("results/Hopf/rmse_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Hopf/rmse_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Edge activations for engineered model
fig, axes = plot_all_edges(
    model_phi,
    X=Phi_train_n[:5000],
    in_var_names=ENG_FEATURE_NAMES,
    out_var_names=["p1", "p2", "p3"],
    save="results/Hopf/edge_activations_eng",
)
plt.close(fig)

print("[FIGS]  Saved results/Hopf/")
