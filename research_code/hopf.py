#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pykan')


# In[2]:


# ============================================================
# One-cell script: train BOTH KANs, use SAME fibers, ONE figure,
# then plot each KAN graph + pretty-print symbolic formulas.
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN

# (optional) prettier formulas if sympy is available
try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

# -----------------------------
# 0) Setup
# -----------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)

rbf = lambda x: torch.exp(-x**2)

# -----------------------------
# 1) Core geometry: S^3 -> S^2 (Hopf map) + helpers
# -----------------------------
def sample_s3(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(n, 4, generator=g, device=device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x

def hopf_map_torch(x4: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4v = x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3]
    z1 = torch.complex(x1, x2)
    z2 = torch.complex(x3, x4v)
    w  = z1 * torch.conj(z2)
    y1 = 2.0 * torch.real(w)
    y2 = 2.0 * torch.imag(w)
    y3 = (torch.abs(z1)**2) - (torch.abs(z2)**2)
    return torch.stack([y1, y2, y3], dim=1)

def stereographic_s3_to_r3_torch(x4: torch.Tensor) -> torch.Tensor:
    denom = (1.0 - x4[:, 3]).clamp_min(1e-9)
    return x4[:, :3] / denom[:, None]

def section_s2_to_s3_torch(y: torch.Tensor) -> torch.Tensor:
    # a simple section avoiding the south pole
    a, b, c = y[0], y[1], y[2]
    z1 = torch.sqrt(((1.0 + c).clamp_min(1e-12)) / 2.0)  # real >=0
    z2 = torch.complex(a, b) / (2.0 * z1)
    x = torch.stack([z1, torch.zeros_like(z1), torch.real(z2), torch.imag(z2)])
    return x / (x.norm() + 1e-12)

def hopf_fiber_from_x0_torch(x0: torch.Tensor, m: int = 700) -> torch.Tensor:
    z1 = torch.complex(x0[0], x0[1])
    z2 = torch.complex(x0[2], x0[3])
    t = torch.linspace(0, 2*torch.pi, m, device=x0.device)
    eit = torch.exp(1j * t)
    z1t = eit * z1
    z2t = eit * z2
    return torch.stack([torch.real(z1t), torch.imag(z1t), torch.real(z2t), torch.imag(z2t)], dim=1)

def sample_s2_basepoints_torch(n_lat=4, n_phi=8, device="cpu"):
    # avoid exact south pole
    cs = torch.linspace(0.85, -0.85, n_lat, device=device)
    pts = []
    for c in cs:
        r = torch.sqrt(torch.clamp(1.0 - c*c, min=0.0))
        phis = torch.linspace(0, 2*torch.pi, n_phi+1, device=device)[:-1]
        for phi in phis:
            pts.append(torch.stack([r*torch.cos(phi), r*torch.sin(phi), c]))
    pts = torch.stack(pts, dim=0)
    pts = pts / (pts.norm(dim=1, keepdim=True) + 1e-12)
    return pts

def angle_error(yhat, ytrue, eps=1e-12):
    yhat = yhat / (yhat.norm(dim=1, keepdim=True) + eps)
    ytrue = ytrue / (ytrue.norm(dim=1, keepdim=True) + eps)
    dot = (yhat * ytrue).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(dot)

# -----------------------------
# 2) Engineered 5D features (Option A)
# -----------------------------
def hopf_features_option_A(X: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    u1 = x1 * x3
    u2 = x2 * x4
    u3 = x2 * x3
    u4 = x1 * x4
    u5 = x1 * x1 + x2 * x2 - x3 * x3 - x4 * x4
    return torch.stack([u1, u2, u3, u4, u5], dim=1)

def to_phi_norm(X4: torch.Tensor, phi_mu: torch.Tensor, phi_std: torch.Tensor) -> torch.Tensor:
    Phi = hopf_features_option_A(X4)
    return (Phi - phi_mu) / (phi_std + 1e-12)

# -----------------------------
# 3) Data (shared)
# -----------------------------
train_n = 20_000
test_n  = 5_000

X_train = sample_s3(train_n, device=device, seed=0)
X_test  = sample_s3(test_n,  device=device, seed=1)

Y_train = hopf_map_torch(X_train)
Y_test  = hopf_map_torch(X_test)

# A) Raw-input experiment uses label normalization (your first version)
y_mu  = Y_train.mean(dim=0, keepdim=True)
y_std = Y_train.std(dim=0, keepdim=True) + 1e-12
Y_train_n = (Y_train - y_mu) / y_std
Y_test_n  = (Y_test  - y_mu) / y_std

dataset_raw = {
    "train_input": X_train,
    "train_label": Y_train_n,
    "test_input":  X_test,
    "test_label":  Y_test_n,
    "normalize_label": True,
    "normalize_label_mu": y_mu,
    "normalize_label_sigma": y_std,
}

# B) Engineered-input experiment uses Phi normalization (your second version)
Phi_train = hopf_features_option_A(X_train)
Phi_test  = hopf_features_option_A(X_test)
phi_mu  = Phi_train.mean(dim=0, keepdim=True)
phi_std = Phi_train.std(dim=0, keepdim=True) + 1e-12

Phi_train_n = (Phi_train - phi_mu) / phi_std
Phi_test_n  = (Phi_test  - phi_mu) / phi_std

dataset_phi = {
    "train_input": Phi_train_n,
    "train_label": Y_train,   # NOT normalized labels in your second experiment
    "test_input":  Phi_test_n,
    "test_label":  Y_test,
}

# -----------------------------
# 4) Train BOTH models
# -----------------------------
# Raw-input deep KAN (outputs are normalized labels)
model_raw = KAN(
    width=[4, 3],
    grid=64,
    k=3,
    base_fun=rbf,
    seed=42,
    device=device,
)
losss1 = model_raw.fit(dataset_raw, steps=400)

# Engineered-input small KAN (outputs are direct Hopf coords)
model_phi = KAN(
    width=[5, 3],
    grid=64,
    k=3,
    base_fun=rbf,
    seed=42,
    device=device,
)
loss2 = model_phi.fit(dataset_phi, steps=400)

# quick sanity numbers
with torch.no_grad():
    # raw model needs un-normalization
    pred_raw = model_raw(dataset_raw["test_input"]) * y_std + y_mu
    mse_raw  = torch.mean((pred_raw - Y_test) ** 2).item()

    pred_phi = model_phi(dataset_phi["test_input"])
    mse_phi  = torch.mean((pred_phi - Y_test) ** 2).item()

print(f"test mse (raw-input, unnormed): {mse_raw:.3e}")
print(f"test mse (engineered-5D):       {mse_phi:.3e}")

# -----------------------------
# 5) SAME fibers, ONE figure with 3 panels
#    Left: stereographic fibers in R^3
#    Mid:  raw-model outputs along fibers (should collapse on S^2)
#    Right: engineered-model outputs along fibers (should collapse on S^2)
# -----------------------------
def make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    num_fibers=18,
    m=700,
    s2_cloud_points_per_fiber=120,
    seed=0,
    savepath=None,
):
    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()

    # choose SAME basepoints once
    base = sample_s2_basepoints_torch(n_lat=4, n_phi=8, device=device)
    if num_fibers < base.shape[0]:
        idx = torch.randperm(base.shape[0], device=device)[:num_fibers]
        base = base[idx]

    # precompute fiber geometry + predictions
    fibers_r3 = []
    fiber_colors = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []

    take = torch.linspace(0, m-1, s2_cloud_points_per_fiber, device=device).long()

    with torch.no_grad():
        for y in base:
            x0 = section_s2_to_s3_torch(y)
            fiber_s3 = hopf_fiber_from_x0_torch(x0, m=m)  # (m,4)

            # left panel: stereographic curve
            r3 = stereographic_s3_to_r3_torch(fiber_s3)   # (m,3)
            fibers_r3.append(r3.detach().cpu().numpy())

            # consistent fiber color from basepoint
            col = ((y.clamp(-1, 1) + 1.0) / 2.0).detach().cpu().numpy()
            fiber_colors.append(col)

            # raw model prediction (needs unnorm)
            yhat_raw = model_raw(fiber_s3) * y_std + y_mu
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())

            # engineered model prediction (needs Phi transform)
            Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)  # (m,5)
            yhat_phi = model_phi(Phi_f_n)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())

    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)

    # compute angular stats on a random batch for both models
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)

        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw  = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))

        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi  = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))

    # ---- plotting ----
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
    })

    fig = plt.figure(figsize=(16.2, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.12)

    axL = fig.add_subplot(gs[0, 0], projection="3d")
    axM = fig.add_subplot(gs[0, 1], projection="3d")
    axR = fig.add_subplot(gs[0, 2], projection="3d")

    # Left: fibers in R^3
    for r3, col in zip(fibers_r3, fiber_colors):
        axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=2.0, alpha=0.95, color=col)
    #axL.set_title("Hopf fibers in $\\mathbb{R}^3$ (stereo proj. of $S^3$)")
    axL.set_xlabel("X"); axL.set_ylabel("Y"); axL.set_zlabel("Z")
    axL.set_box_aspect((1, 1, 1))
    axL.view_init(elev=18, azim=35)
    axL.grid(False)

    # Sphere wireframe for middle/right panels
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 16)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    for ax in (axM, axR):
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.25)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=35)
        ax.grid(False)
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$"); ax.set_zlabel("$y_3$")

    # Middle: raw model outputs on S^2
    for cloud, col, meanpt in zip(clouds_raw, fiber_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, alpha=0.55, color=col)
        axM.scatter(meanpt[0], meanpt[1], meanpt[2], s=60, alpha=0.95, color=col)
    #axM.set_title("KAN outputs along fibers (raw 4D input)")

    # Right: engineered model outputs on S^2
    for cloud, col, meanpt in zip(clouds_phi, fiber_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, alpha=0.55, color=col)
        axR.scatter(meanpt[0], meanpt[1], meanpt[2], s=60, alpha=0.95, color=col)
    #axR.set_title("KAN outputs along fibers (engineered $\\Phi\\in\\mathbb{R}^5$)")

    # # Title with key metrics
    msg = (
        "Same fibers for both models.\n"
        f"Raw-input: mean ang err={np.mean(ang_raw):.2e} rad (p95={np.quantile(ang_raw,0.95):.2e}), "
        f"mean fiber RMS={rms_raw.mean():.2e} (max={rms_raw.max():.2e}), mean |‖ŷ‖−1|={norm_dev_raw:.2e}\n"
        f"5D-phi:    mean ang err={np.mean(ang_phi):.2e} rad (p95={np.quantile(ang_phi,0.95):.2e}), "
        f"mean fiber RMS={rms_phi.mean():.2e} (max={rms_phi.max():.2e}), mean |‖ŷ‖−1|={norm_dev_phi:.2e}"
    )
    print(msg)
    #fig.suptitle(msg, y=0.98, fontsize=10)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        print("Saved:", savepath)

    plt.show()

make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device=device,
    num_fibers=18,
    m=1700,
    s2_cloud_points_per_fiber=120,
    seed=0,
    savepath="hopf_fibres_from_kans.pdf"
)

# -----------------------------
# 6) Plot each KAN graph + print symbolic formulas (pretty)
# -----------------------------
def pretty_print_symbolic(model, name="model"):
    print("\n" + "="*70)
    print(f"{name}: auto_symbolic() + symbolic_formula()")
    print("="*70)
    model.auto_symbolic()
    f = model.symbolic_formula()

    # Try to pretty print if sympy is around; otherwise just print raw.
    if _HAS_SYMPY:
        try:
            # common cases: list/tuple of expressions, sympy Matrix, etc.
            if isinstance(f, (list, tuple)):
                for i, fi in enumerate(f):
                    print(f"\n{name}[{i}] =")
                    print(sp.pretty(fi))
            else:
                print(sp.pretty(f))
        except Exception:
            print(f)
    else:
        print(f)

# (a) KAN graph plot + formula for raw-input model
model_raw.save_acts = True
_ = model_raw(dataset_raw["train_input"][:2048])  # run once so plot has activations
model_raw.plot()
plt.savefig("4D_kan_model.pdf", dpi=300, bbox_inches="tight")
plt.close()
pretty_print_symbolic(model_raw, name="KAN (raw 4D input)")

# (b) KAN graph plot + formula for engineered-5D model
model_phi.save_acts = True
_ = model_phi(dataset_phi["train_input"][:2048])
model_phi.plot()
plt.savefig("5D_kan_model.pdf", dpi=300, bbox_inches="tight")
plt.close()
pretty_print_symbolic(model_phi, name="KAN (engineered Phi 5D)")


# In[3]:


# ============================================================
# Minimal publication-grade loss plots (robust to dict logs)
# One figure per loss curve
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

def extract_loss_series(obj):
    """
    Returns 1D numpy array of losses from:
      - list/np/torch
      - dict with keys like 'loss', 'train_loss', etc.
      - list of dicts with per-step loss entries
    """
    # torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().ravel().astype(float)
    except Exception:
        pass

    # list of dicts (common)
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict):
        # prefer common keys
        keys_priority = ["loss", "train_loss", "test_loss", "l"]
        k = None
        for kk in keys_priority:
            if kk in obj[0]:
                k = kk
                break
        if k is None:
            # fallback: first numeric-looking key
            for kk, vv in obj[0].items():
                if isinstance(vv, (int, float, np.number)):
                    k = kk
                    break
        if k is None:
            raise TypeError("Couldn't find a numeric loss key in list-of-dicts log.")
        return np.array([d[k] for d in obj], dtype=float).ravel()

    # dict of arrays / scalars
    if isinstance(obj, dict):
        keys_priority = ["loss", "train_loss", "test_loss", "l"]
        for k in keys_priority:
            if k in obj:
                return extract_loss_series(obj[k])
        # fallback: first value that can be converted
        for v in obj.values():
            try:
                return extract_loss_series(v)
            except Exception:
                continue
        raise TypeError("Couldn't extract a loss series from dict log.")

    # plain list/np array
    arr = np.asarray(obj)
    if arr.dtype == object:
        # sometimes it's a nested structure; try flatten by mapping
        try:
            return np.array([float(x) for x in obj], dtype=float).ravel()
        except Exception as e:
            raise TypeError(f"Loss object not plottable (dtype=object): {e}")
    return arr.ravel().astype(float)

def plot_loss_curve(loss_obj, title, outname, logy=True):
    y = extract_loss_series(loss_obj)
    x = np.arange(len(y))

    # publication-ish, minimal
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, y)

    #ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    if logy:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
    fig.tight_layout()

    fig.savefig(f"{outname}.pdf", bbox_inches="tight")
    fig.savefig(f"{outname}.png", bbox_inches="tight")
    plt.show()

# ---- one figure per curve ----
plot_loss_curve(losss1, "Training loss — raw 4D input KAN", "loss_raw_input_kan", logy=True)
plot_loss_curve(loss2,  "Training loss — engineered Φ (5D) KAN", "loss_engineered_phi_kan", logy=True)


# In[5]:


def hopf_fiber_parametric(theta: float, phi: float, m: int = 500, device="cpu") -> torch.Tensor:
    """
    Generate a Hopf fiber for a basepoint on S^2 given by spherical coords (theta, phi).
    theta in [0, pi], phi in [0, 2*pi]
    Returns points on S^3 (before stereographic projection).
    """
    # Basepoint on S^2
    y1 = np.sin(theta) * np.cos(phi)
    y2 = np.sin(theta) * np.sin(phi)
    y3 = np.cos(theta)
    
    # Section: lift to S^3
    # Using standard section: z1 = sqrt((1+y3)/2), z2 = (y1 + i*y2)/(2*z1)
    z1_mag = np.sqrt(max((1.0 + y3) / 2.0, 1e-12))
    z2 = complex(y1, y2) / (2.0 * z1_mag) if z1_mag > 1e-9 else complex(0, 0)
    
    # Fiber: multiply by e^{i*t}
    t = torch.linspace(0, 2*np.pi, m, device=device)
    eit = torch.exp(1j * t)
    
    z1_fiber = eit * z1_mag
    z2_fiber = eit * z2
    
    x1 = torch.real(z1_fiber)
    x2 = torch.imag(z1_fiber)
    x3 = torch.real(z2_fiber)
    x4 = torch.imag(z2_fiber)
    
    return torch.stack([x1, x2, x3, x4], dim=1)


def generate_nested_tori_fibers(
    n_tori: int = 3,
    fibers_per_torus: int = 12,
    m: int = 500,
    device: str = "cpu"
) -> list:
    """
    Generate fibers organized as nested tori.
    Each torus corresponds to a latitude circle on S^2.
    Returns list of (fiber_s3, torus_idx, fiber_idx, theta, phi) tuples.
    """
    fibers = []
    
    # Thetas for different latitude circles (tori)
    # Avoid poles for numerical stability
    thetas = np.linspace(0.3 * np.pi, 0.7 * np.pi, n_tori)
    
    for torus_idx, theta in enumerate(thetas):
        phis = np.linspace(0, 2 * np.pi, fibers_per_torus, endpoint=False)
        for fiber_idx, phi in enumerate(phis):
            fiber = hopf_fiber_parametric(theta, phi, m=m, device=device)
            fibers.append({
                'fiber_s3': fiber,
                'torus_idx': torus_idx,
                'fiber_idx': fiber_idx,
                'theta': theta,
                'phi': phi,
            })
    
    return fibers


def generate_villarceau_circles(
    center_theta: float = np.pi / 2,
    n_circles: int = 8,
    m: int = 500,
    device: str = "cpu"
) -> list:
    """
    Generate Villarceau circles - linked fibers that form beautiful interlocking patterns.
    These are fibers whose basepoints lie on a great circle of S^2.
    """
    fibers = []
    phis = np.linspace(0, 2 * np.pi, n_circles, endpoint=False)
    
    for idx, phi in enumerate(phis):
        fiber = hopf_fiber_parametric(center_theta, phi, m=m, device=device)
        fibers.append({
            'fiber_s3': fiber,
            'circle_idx': idx,
            'phi': phi,
        })
    
    return fibers


# -----------------------------
# ENHANCED: Custom Color Maps
# -----------------------------
def create_hopf_colormap():
    """Create a sophisticated colormap for Hopf fibers."""
    colors = [
        '#1a1a2e',  # Deep navy
        '#16213e',  # Dark blue
        '#0f3460',  # Royal blue
        '#e94560',  # Coral red
        '#ff6b6b',  # Salmon
        '#feca57',  # Gold
        '#48dbfb',  # Cyan
        '#1dd1a1',  # Teal
        '#5f27cd',  # Purple
    ]
    return LinearSegmentedColormap.from_list('hopf', colors, N=256)


def create_torus_colormaps(n_tori: int):
    """Create distinct colormaps for each torus."""
    base_maps = [
        ['#0a0a23', '#1e3a5f', '#3d5a80', '#98c1d9', '#e0fbfc'],  # Ocean blues
        ['#1a0a2e', '#3d1e6d', '#8338ec', '#c77dff', '#e0aaff'],  # Purple/violet
        ['#0d1b0d', '#1e441e', '#2d6a4f', '#40916c', '#74c69d'],  # Forest greens
        ['#2b0a0a', '#5a1a1a', '#9d0208', '#dc2f02', '#ffba08'],  # Fire
        ['#0a1628', '#1b3a4b', '#006466', '#065a60', '#0b525b'],  # Teal depths
    ]
    
    cmaps = []
    for i in range(n_tori):
        colors = base_maps[i % len(base_maps)]
        cmaps.append(LinearSegmentedColormap.from_list(f'torus_{i}', colors, N=256))
    
    return cmaps


# In[6]:


model_raw = KAN(
    width=[4, 3],
    grid=64,
    k=3,
    base_fun=rbf,
    seed=42,
    device=device,
)
losss1 = model_raw.fit(dataset_raw, steps=200)

# Engineered-input small KAN (outputs are direct Hopf coords)
model_phi = KAN(
    width=[5, 3],
    grid=64,
    k=3,
    base_fun=rbf,
    seed=42,
    device=device,
)
loss2 = model_phi.fit(dataset_phi, steps=200)

def make_trefoil_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    n_fibers: int = 24,
    m: int = 900,
    s2_cloud_points_per_fiber: int = 140,
    p: int = 2, q: int = 3,              # trefoil-ish torus-knot params on S^2
    theta_amp: float = 0.40,             # modulation of theta
    phi_amp: float = 0.30,               # modulation of phi
    seed: int = 42,
    dark_theme: bool = True,
    glow_effect: bool = True,
    savepath: str = None,
):
    """
    OPTION A (model-aware):
      Panel 1: trefoil-chosen Hopf fibers in R^3 (stereographic S^3->R^3)
      Panel 2: raw-input KAN outputs along those fibers (should collapse to points on S^2)
      Panel 3: engineered-phi KAN outputs along those fibers (should collapse to points on S^2)

    Uses the SAME fibers for both models.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()

    # -----------------------------
    # Theme / rcParams
    # -----------------------------
    if dark_theme:
        plt.style.use("dark_background")
        bg_color = "#050510"
        text_color = "#e8e8e8"
        grid_color = "#1a1a2e"
        wireframe_color = "#2a2a4a"
    else:
        plt.style.use("default")
        bg_color = "#ffffff"
        text_color = "#1a1a1a"
        grid_color = "#e0e0e0"
        wireframe_color = "#cccccc"

    plt.rcParams.update({
        "figure.dpi": 180,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
    })

    # -----------------------------
    # Build trefoil-chosen basepoints on S^2 and corresponding Hopf fibers
    # -----------------------------
    t = np.linspace(0, 2*np.pi, n_fibers, endpoint=False)

    fibers_r3 = []
    fiber_colors = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []

    cmap = plt.cm.twilight_shifted
    take = torch.linspace(0, m - 1, s2_cloud_points_per_fiber, device=device).long()

    with torch.no_grad():
        for i, ti in enumerate(t):
            # "trefoil path" on S^2 via spherical coords
            theta = np.pi/2 + theta_amp * np.sin(q * ti)
            phi   = p * ti + phi_amp * np.cos(q * ti)

            fiber_s3 = hopf_fiber_parametric(theta, phi, m=m, device=device)   # (m,4)

            # R^3 geometry
            r3 = stereographic_s3_to_r3_torch(fiber_s3)                        # (m,3)
            fibers_r3.append(r3.detach().cpu().numpy())

            # consistent fiber color
            col = cmap(i / n_fibers)
            fiber_colors.append(col)

            # raw model outputs (unnormalize labels)
            yhat_raw = model_raw(fiber_s3) * y_std + y_mu                      # (m,3)
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())

            # engineered model outputs (compute Phi then predict)
            Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)                   # (m,5)
            yhat_phi = model_phi(Phi_f_n)                                      # (m,3)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())

    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)

    # -----------------------------
    # Also compute batch stats (angular error + norm deviation) for both models
    # -----------------------------
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)

        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))

        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))

    # -----------------------------
    # Figure: 1x3
    # -----------------------------
    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.10)

    axL = fig.add_subplot(gs[0, 0], projection="3d")
    axM = fig.add_subplot(gs[0, 1], projection="3d")
    axR = fig.add_subplot(gs[0, 2], projection="3d")

    for ax in (axL, axM, axR):
        ax.set_facecolor(bg_color)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.08, color=grid_color)

    # ----- Panel 1: Trefoil-chosen fibers in R^3 -----
    for r3, col in zip(fibers_r3, fiber_colors):
        axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=2.0, alpha=0.90, color=col)
        if glow_effect:
            axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=4.8, alpha=0.18, color=col)
            axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=3.2, alpha=0.25, color=col)

    axL.set_title("Trefoil-selected Hopf fibers in $\\mathbb{R}^3$")
    axL.set_xlabel("X"); axL.set_ylabel("Y"); axL.set_zlabel("Z")
    axL.set_box_aspect((1, 1, 1))
    axL.view_init(elev=18, azim=60)

    all_r3 = np.concatenate(fibers_r3, axis=0)
    max_ext = np.abs(all_r3).max() * 1.08
    axL.set_xlim(-max_ext, max_ext)
    axL.set_ylim(-max_ext, max_ext)
    axL.set_zlim(-max_ext, max_ext)

    # ----- Sphere wireframe for S^2 panels -----
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    for ax in (axM, axR):
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2,
                          linewidth=0.35, alpha=0.18, color=wireframe_color)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=60)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$"); ax.set_zlabel("$y_3$")

    # ----- Panel 2: raw KAN outputs -----
    for cloud, col, meanpt in zip(clouds_raw, fiber_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                    s=9, alpha=0.50, color=col, edgecolors="none")
        axM.scatter(meanpt[0], meanpt[1], meanpt[2],
                    s=85, alpha=0.95, color=col, edgecolors="white", linewidths=0.4)
    axM.set_title("KAN $\\mathbf{x}\\in\\mathbb{R}^5$)")

    # ----- Panel 3: engineered KAN outputs -----
    for cloud, col, meanpt in zip(clouds_phi, fiber_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                    s=9, alpha=0.50, color=col, edgecolors="none")
        axR.scatter(meanpt[0], meanpt[1], meanpt[2],
                    s=85, alpha=0.95, color=col, edgecolors="white", linewidths=0.4)
    axR.set_title("KANDy $\\Phi\\in\\mathbb{R}^5$)")

    # # ----- Title / metrics -----
    # supt = (
    #     "Trefoil fiber set (same fibers for both models)\n"
    #     f"Raw: mean ang={np.mean(ang_raw):.2e} (p95={np.quantile(ang_raw,0.95):.2e}), "
    #     f"mean RMS={rms_raw.mean():.2e} (max={rms_raw.max():.2e}), mean |‖ŷ‖−1|={norm_dev_raw:.2e}\n"
    #     f"Phi: mean ang={np.mean(ang_phi):.2e} (p95={np.quantile(ang_phi,0.95):.2e}), "
    #     f"mean RMS={rms_phi.mean():.2e} (max={rms_phi.max():.2e}), mean |‖ŷ‖−1|={norm_dev_phi:.2e}"
    # )
    # fig.suptitle(supt, y=0.98, fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor=bg_color)
        print(f"Saved: {savepath}")

    plt.show()

    return {
        "rms_raw": rms_raw,
        "rms_phi": rms_phi,
        "ang_raw": ang_raw,
        "ang_phi": ang_phi,
        "fibers_r3": fibers_r3,
    }

# Example usage:
results = make_trefoil_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device=device,
    n_fibers=24, m=800,
    dark_theme=False, glow_effect=True,
    savepath="hopf_trefoil_compare.pdf"
)


# In[19]:


# -----------------------------
# NEW: (p,q)-torus knot in S^3  (trefoil = (2,3) or (3,2))
# Represent S^3 via (z1,z2) with |z1|^2+|z2|^2=1:
#   z1 = cos(alpha) * e^{i (p t + phi1)}
#   z2 = sin(alpha) * e^{i (q t + phi2)}
# This is a torus knot/link on the Clifford torus in S^3.
# -----------------------------
def torus_knot_s3_torch(p:int, q:int, alpha:float, m:int, device, phi1:float=0.0, phi2:float=0.0):
    t = torch.linspace(0, 2*torch.pi, m, device=device)
    z1 = torch.cos(torch.tensor(alpha, device=device)) * torch.exp(1j*(p*t + phi1))
    z2 = torch.sin(torch.tensor(alpha, device=device)) * torch.exp(1j*(q*t + phi2))
    x = torch.stack([torch.real(z1), torch.imag(z1), torch.real(z2), torch.imag(z2)], dim=1)
    # already on S^3 by construction
    return x

def linked_trefoils_s3_torch(
    m:int, device,
    trefoil=(2,3),
    alpha=np.pi/4,
    # phase offsets determine how the two components sit relative to each other
    delta_phi1=np.pi/2,
    delta_phi2=np.pi/2,
):
    p,q = trefoil
    K1 = torus_knot_s3_torch(p, q, alpha, m, device, phi1=0.0,       phi2=0.0)
    K2 = torus_knot_s3_torch(p, q, alpha, m, device, phi1=delta_phi1, phi2=delta_phi2)
    return [K1, K2]

# -----------------------------
# MODIFIED: one figure, but can draw trefoils instead of sampled fibers
# -----------------------------
def make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    num_fibers=18,
    m=700,
    s2_cloud_points_per_fiber=120,
    seed=0,
    savepath=None,
    geometry_mode="fibers",   # "fibers" or "trefoils"
    evaluate_mode="fibers",   # "fibers" or "geometry"
):
    """
    geometry_mode:
      - "fibers": left panel shows Hopf fibers (circles / Hopf links)
      - "trefoils": left panel shows linked trefoils (2 components)

    evaluate_mode:
      - "fibers": middle/right computed along true Hopf fibers (collapse demo)
      - "geometry": middle/right computed along whatever is drawn on the left
                   (trefoil trajectories; won't collapse)
    """
    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()

    # choose SAME Hopf basepoints once (for fiber evaluation and/or colors)
    base = sample_s2_basepoints_torch(n_lat=4, n_phi=8, device=device)
    if num_fibers < base.shape[0]:
        idx = torch.randperm(base.shape[0], device=device)[:num_fibers]
        base = base[idx]

    take = torch.linspace(0, m-1, s2_cloud_points_per_fiber, device=device).long()

    # ---------- build geometry curves on S^3 ----------
    curves_s3 = []
    curve_colors = []

    if geometry_mode == "trefoils":
        # two linked trefoils
        trefoils = linked_trefoils_s3_torch(
            m=m, device=device,
            trefoil=(2,3),
            alpha=np.pi/4,
            delta_phi1=np.pi/2,
            delta_phi2=np.pi/2,
        )
        curves_s3 = trefoils

        # nice distinct colors for the 2 components
        curve_colors = [np.array([0.85, 0.25, 0.25]), np.array([0.25, 0.45, 0.85])]

    else:
        # Hopf fibers from those basepoints
        with torch.no_grad():
            for y in base:
                x0 = section_s2_to_s3_torch(y)
                fiber_s3 = hopf_fiber_from_x0_torch(x0, m=m)
                curves_s3.append(fiber_s3)
                col = ((y.clamp(-1, 1) + 1.0) / 2.0).detach().cpu().numpy()
                curve_colors.append(col)

    # ---------- decide which curves to use for evaluation ----------
    if evaluate_mode == "geometry":
        eval_curves_s3 = curves_s3
        eval_colors = curve_colors
    else:
        # always evaluate on true Hopf fibers (best for “collapse” story)
        eval_curves_s3 = []
        eval_colors = []
        with torch.no_grad():
            for y in base:
                x0 = section_s2_to_s3_torch(y)
                fiber_s3 = hopf_fiber_from_x0_torch(x0, m=m)
                eval_curves_s3.append(fiber_s3)
                col = ((y.clamp(-1, 1) + 1.0) / 2.0).detach().cpu().numpy()
                eval_colors.append(col)

    # ---------- compute geometry curves in R^3 + predictions ----------
    curves_r3 = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []

    with torch.no_grad():
        # left panel uses curves_s3
        for C, col in zip(curves_s3, curve_colors):
            r3 = stereographic_s3_to_r3_torch(C)
            curves_r3.append(r3.detach().cpu().numpy())

        # middle/right use eval_curves_s3
        for C, col in zip(eval_curves_s3, eval_colors):
            # raw model
            yhat_raw = model_raw(C) * y_std + y_mu
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())

            # engineered model
            Phi_n = to_phi_norm(C, phi_mu, phi_std)
            yhat_phi = model_phi(Phi_n)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())

    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)

    # ---------- batch angular stats ----------
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)

        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw  = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))

        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi  = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))

    # ---------- plot ----------
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig = plt.figure(figsize=(16.2, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.12)
    axL = fig.add_subplot(gs[0, 0], projection="3d")
    axM = fig.add_subplot(gs[0, 1], projection="3d")
    axR = fig.add_subplot(gs[0, 2], projection="3d")

    # Left: geometry in R^3
    for r3, col in zip(curves_r3, curve_colors):
        axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=2.2, alpha=0.98, color=col)

    if geometry_mode == "trefoils":
        axL.set_title("Linked trefoils in $\\mathbb{R}^3$ (stereographic projection of $S^3$)")
    else:
        axL.set_title("Hopf fibers in $\\mathbb{R}^3$ (stereographic projection of $S^3$)")

    axL.set_xlabel("X"); axL.set_ylabel("Y"); axL.set_zlabel("Z")
    axL.set_box_aspect((1, 1, 1))
    axL.view_init(elev=18, azim=35)
    axL.grid(False)

    # wireframe sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 16)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    for ax in (axM, axR):
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.25)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=35)
        ax.grid(False)
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$"); ax.set_zlabel("$y_3$")

    # Middle: raw model outputs
    for cloud, col, meanpt in zip(clouds_raw, eval_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, alpha=0.55, color=col)
        axM.scatter(meanpt[0], meanpt[1], meanpt[2], s=60, alpha=0.95, color=col)
    axM.set_title("Raw-input KAN outputs along evaluation curves")

    # Right: engineered model outputs
    for cloud, col, meanpt in zip(clouds_phi, eval_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=10, alpha=0.55, color=col)
        axR.scatter(meanpt[0], meanpt[1], meanpt[2], s=60, alpha=0.95, color=col)
    axR.set_title("Engineered-5D KAN outputs along evaluation curves")

    msg = (
        f"Eval mode: {evaluate_mode} | Geometry mode: {geometry_mode}\n"
        f"Raw: mean ang err={np.mean(ang_raw):.2e} rad (p95={np.quantile(ang_raw,0.95):.2e}), "
        f"mean RMS={rms_raw.mean():.2e} (max={rms_raw.max():.2e}), mean |‖ŷ‖−1|={norm_dev_raw:.2e}\n"
        f"5D:  mean ang err={np.mean(ang_phi):.2e} rad (p95={np.quantile(ang_phi,0.95):.2e}), "
        f"mean RMS={rms_phi.mean():.2e} (max={rms_phi.max():.2e}), mean |‖ŷ‖−1|={norm_dev_phi:.2e}"
    )
    fig.suptitle(msg, y=0.98, fontsize=10)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        print("Saved:", savepath)

    plt.show()


# In[20]:


make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device=device,
    geometry_mode="trefoils",
    evaluate_mode="fibers",
)


# In[22]:


# ============================================================
# Enhanced Hopf Fibration Visualization with Nested Tori
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

# (optional) prettier formulas if sympy is available
try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

# -----------------------------
# 0) Setup
# -----------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)

rbf = lambda x: torch.exp(-x**2)

# -----------------------------
# 1) Core geometry: S^3 -> S^2 (Hopf map) + helpers
# -----------------------------
def sample_s3(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(n, 4, generator=g, device=device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x

def hopf_map_torch(x4: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4v = x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3]
    z1 = torch.complex(x1, x2)
    z2 = torch.complex(x3, x4v)
    w  = z1 * torch.conj(z2)
    y1 = 2.0 * torch.real(w)
    y2 = 2.0 * torch.imag(w)
    y3 = (torch.abs(z1)**2) - (torch.abs(z2)**2)
    return torch.stack([y1, y2, y3], dim=1)

def stereographic_s3_to_r3_torch(x4: torch.Tensor) -> torch.Tensor:
    denom = (1.0 - x4[:, 3]).clamp_min(1e-9)
    return x4[:, :3] / denom[:, None]

def section_s2_to_s3_torch(y: torch.Tensor) -> torch.Tensor:
    # a simple section avoiding the south pole
    a, b, c = y[0], y[1], y[2]
    z1 = torch.sqrt(((1.0 + c).clamp_min(1e-12)) / 2.0)
    z2 = torch.complex(a, b) / (2.0 * z1)
    x = torch.stack([z1, torch.zeros_like(z1), torch.real(z2), torch.imag(z2)])
    return x / (x.norm() + 1e-12)

def hopf_fiber_from_x0_torch(x0: torch.Tensor, m: int = 700) -> torch.Tensor:
    z1 = torch.complex(x0[0], x0[1])
    z2 = torch.complex(x0[2], x0[3])
    t = torch.linspace(0, 2*torch.pi, m, device=x0.device)
    eit = torch.exp(1j * t)
    z1t = eit * z1
    z2t = eit * z2
    return torch.stack([torch.real(z1t), torch.imag(z1t), torch.real(z2t), torch.imag(z2t)], dim=1)

def sample_s2_basepoints_torch(n_lat=4, n_phi=8, device="cpu"):
    cs = torch.linspace(0.85, -0.85, n_lat, device=device)
    pts = []
    for c in cs:
        r = torch.sqrt(torch.clamp(1.0 - c*c, min=0.0))
        phis = torch.linspace(0, 2*torch.pi, n_phi+1, device=device)[:-1]
        for phi in phis:
            pts.append(torch.stack([r*torch.cos(phi), r*torch.sin(phi), c]))
    pts = torch.stack(pts, dim=0)
    pts = pts / (pts.norm(dim=1, keepdim=True) + 1e-12)
    return pts

def angle_error(yhat, ytrue, eps=1e-12):
    yhat = yhat / (yhat.norm(dim=1, keepdim=True) + eps)
    ytrue = ytrue / (ytrue.norm(dim=1, keepdim=True) + eps)
    dot = (yhat * ytrue).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(dot)

# -----------------------------
# 2) Engineered 5D features (Option A)
# -----------------------------
def hopf_features_option_A(X: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    u1 = x1 * x3
    u2 = x2 * x4
    u3 = x2 * x3
    u4 = x1 * x4
    u5 = x1 * x1 + x2 * x2 - x3 * x3 - x4 * x4
    return torch.stack([u1, u2, u3, u4, u5], dim=1)

def to_phi_norm(X4: torch.Tensor, phi_mu: torch.Tensor, phi_std: torch.Tensor) -> torch.Tensor:
    Phi = hopf_features_option_A(X4)
    return (Phi - phi_mu) / (phi_std + 1e-12)


# -----------------------------
# ENHANCED: Nested Tori Hopf Fiber Generation
# -----------------------------
def hopf_fiber_parametric(theta: float, phi: float, m: int = 500, device="cpu") -> torch.Tensor:
    """
    Generate a Hopf fiber for a basepoint on S^2 given by spherical coords (theta, phi).
    theta in [0, pi], phi in [0, 2*pi]
    Returns points on S^3 (before stereographic projection).
    """
    # Basepoint on S^2
    y1 = np.sin(theta) * np.cos(phi)
    y2 = np.sin(theta) * np.sin(phi)
    y3 = np.cos(theta)
    
    # Section: lift to S^3
    # Using standard section: z1 = sqrt((1+y3)/2), z2 = (y1 + i*y2)/(2*z1)
    z1_mag = np.sqrt(max((1.0 + y3) / 2.0, 1e-12))
    z2 = complex(y1, y2) / (2.0 * z1_mag) if z1_mag > 1e-9 else complex(0, 0)
    
    # Fiber: multiply by e^{i*t}
    t = torch.linspace(0, 2*np.pi, m, device=device)
    eit = torch.exp(1j * t)
    
    z1_fiber = eit * z1_mag
    z2_fiber = eit * z2
    
    x1 = torch.real(z1_fiber)
    x2 = torch.imag(z1_fiber)
    x3 = torch.real(z2_fiber)
    x4 = torch.imag(z2_fiber)
    
    return torch.stack([x1, x2, x3, x4], dim=1)


def generate_nested_tori_fibers(
    n_tori: int = 3,
    fibers_per_torus: int = 12,
    m: int = 500,
    device: str = "cpu"
) -> list:
    """
    Generate fibers organized as nested tori.
    Each torus corresponds to a latitude circle on S^2.
    Returns list of (fiber_s3, torus_idx, fiber_idx, theta, phi) tuples.
    """
    fibers = []
    
    # Thetas for different latitude circles (tori)
    # Avoid poles for numerical stability
    thetas = np.linspace(0.3 * np.pi, 0.7 * np.pi, n_tori)
    
    for torus_idx, theta in enumerate(thetas):
        phis = np.linspace(0, 2 * np.pi, fibers_per_torus, endpoint=False)
        for fiber_idx, phi in enumerate(phis):
            fiber = hopf_fiber_parametric(theta, phi, m=m, device=device)
            fibers.append({
                'fiber_s3': fiber,
                'torus_idx': torus_idx,
                'fiber_idx': fiber_idx,
                'theta': theta,
                'phi': phi,
            })
    
    return fibers


def generate_villarceau_circles(
    center_theta: float = np.pi / 2,
    n_circles: int = 8,
    m: int = 500,
    device: str = "cpu"
) -> list:
    """
    Generate Villarceau circles - linked fibers that form beautiful interlocking patterns.
    These are fibers whose basepoints lie on a great circle of S^2.
    """
    fibers = []
    phis = np.linspace(0, 2 * np.pi, n_circles, endpoint=False)
    
    for idx, phi in enumerate(phis):
        fiber = hopf_fiber_parametric(center_theta, phi, m=m, device=device)
        fibers.append({
            'fiber_s3': fiber,
            'circle_idx': idx,
            'phi': phi,
        })
    
    return fibers


# -----------------------------
# ENHANCED: Custom Color Maps
# -----------------------------
def create_hopf_colormap():
    """Create a sophisticated colormap for Hopf fibers."""
    colors = [
        '#1a1a2e',  # Deep navy
        '#16213e',  # Dark blue
        '#0f3460',  # Royal blue
        '#e94560',  # Coral red
        '#ff6b6b',  # Salmon
        '#feca57',  # Gold
        '#48dbfb',  # Cyan
        '#1dd1a1',  # Teal
        '#5f27cd',  # Purple
    ]
    return LinearSegmentedColormap.from_list('hopf', colors, N=256)


def create_torus_colormaps(n_tori: int):
    """Create distinct colormaps for each torus."""
    base_maps = [
        ['#0a0a23', '#1e3a5f', '#3d5a80', '#98c1d9', '#e0fbfc'],  # Ocean blues
        ['#1a0a2e', '#3d1e6d', '#8338ec', '#c77dff', '#e0aaff'],  # Purple/violet
        ['#0d1b0d', '#1e441e', '#2d6a4f', '#40916c', '#74c69d'],  # Forest greens
        ['#2b0a0a', '#5a1a1a', '#9d0208', '#dc2f02', '#ffba08'],  # Fire
        ['#0a1628', '#1b3a4b', '#006466', '#065a60', '#0b525b'],  # Teal depths
    ]
    
    cmaps = []
    for i in range(n_tori):
        colors = base_maps[i % len(base_maps)]
        cmaps.append(LinearSegmentedColormap.from_list(f'torus_{i}', colors, N=256))
    
    return cmaps


# -----------------------------
# ENHANCED: Main Visualization Function
# -----------------------------
def make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    # Nested tori parameters
    n_tori: int = 4,
    fibers_per_torus: int = 16,
    # Visualization parameters
    m: int = 800,
    s2_cloud_points_per_fiber: int = 120,
    seed: int = 0,
    # Style parameters
    style: str = 'nested_tori',  # 'nested_tori', 'villarceau', 'classic'
    show_linking: bool = True,
    tube_radius_factor: float = 0.015,
    glow_effect: bool = True,
    dark_theme: bool = True,
    savepath: str = None,
):
    """
    Enhanced Hopf fibration visualization with nested tori structure.
    
    Parameters:
    -----------
    style : str
        'nested_tori' - Fibers organized by latitude circles (nested tori)
        'villarceau' - Interlocking Villarceau circles
        'classic' - Original scattered basepoints
    show_linking : bool
        Highlight the linking structure of fibers
    glow_effect : bool
        Add glow/bloom effect to fibers
    dark_theme : bool
        Use dark background theme
    """
    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()
    
    # ========== Generate Fibers Based on Style ==========
    if style == 'nested_tori':
        fiber_data = generate_nested_tori_fibers(
            n_tori=n_tori,
            fibers_per_torus=fibers_per_torus,
            m=m,
            device=device
        )
        torus_cmaps = create_torus_colormaps(n_tori)
        
    elif style == 'villarceau':
        fiber_data = generate_villarceau_circles(
            center_theta=np.pi / 2,
            n_circles=fibers_per_torus,
            m=m,
            device=device
        )
        
    else:  # classic
        base = sample_s2_basepoints_torch(n_lat=4, n_phi=8, device=device)
        fiber_data = []
        for idx, y in enumerate(base):
            x0 = section_s2_to_s3_torch(y)
            fiber = hopf_fiber_from_x0_torch(x0, m=m)
            fiber_data.append({
                'fiber_s3': fiber,
                'torus_idx': idx // 8,
                'fiber_idx': idx % 8,
                'basepoint': y.detach().cpu().numpy(),
            })
        torus_cmaps = create_torus_colormaps(4)
    
    # ========== Process Fibers ==========
    fibers_r3 = []
    fiber_colors = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []
    
    take = torch.linspace(0, m-1, s2_cloud_points_per_fiber, device=device).long()
    
    with torch.no_grad():
        for i, fd in enumerate(fiber_data):
            fiber_s3 = fd['fiber_s3']
            
            # Stereographic projection to R^3
            r3 = stereographic_s3_to_r3_torch(fiber_s3)
            fibers_r3.append(r3.detach().cpu().numpy())
            
            # Color based on style
            if style == 'nested_tori':
                torus_idx = fd['torus_idx']
                fiber_idx = fd['fiber_idx']
                cmap = torus_cmaps[torus_idx]
                t_param = fiber_idx / fibers_per_torus
                col = cmap(t_param)
            elif style == 'villarceau':
                t_param = fd['circle_idx'] / len(fiber_data)
                col = plt.cm.twilight_shifted(t_param)
            else:
                torus_idx = fd.get('torus_idx', 0)
                fiber_idx = fd.get('fiber_idx', 0)
                cmap = torus_cmaps[torus_idx]
                col = cmap(fiber_idx / 8)
            
            fiber_colors.append(col)
            
            # Raw model prediction
            yhat_raw = model_raw(fiber_s3) * y_std + y_mu
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())
            
            # Engineered model prediction
            Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)
            yhat_phi = model_phi(Phi_f_n)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())
    
    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)
    
    # ========== Angular Statistics ==========
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)
        
        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))
        
        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))
    
    # ========== Figure Setup ==========
    if dark_theme:
        plt.style.use('dark_background')
        bg_color = '#0a0a0f'
        text_color = '#e8e8e8'
        grid_color = '#1a1a2e'
        wireframe_color = '#2a2a4a'
    else:
        bg_color = '#fafafa'
        text_color = '#1a1a1a'
        grid_color = '#e0e0e0'
        wireframe_color = '#cccccc'
    
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "font.family": "sans-serif",
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
    })
    
    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.08)
    
    axL = fig.add_subplot(gs[0, 0], projection="3d")
    axM = fig.add_subplot(gs[0, 1], projection="3d")
    axR = fig.add_subplot(gs[0, 2], projection="3d")
    
    for ax in [axL, axM, axR]:
        ax.set_facecolor(bg_color)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(grid_color)
        ax.yaxis.pane.set_edgecolor(grid_color)
        ax.zaxis.pane.set_edgecolor(grid_color)
        ax.grid(True, alpha=0.1, color=grid_color)
    
    # ========== Left Panel: Nested Tori in R^3 ==========
    
    # Draw fibers with varying thickness for depth
    for idx, (r3, col) in enumerate(zip(fibers_r3, fiber_colors)):
        # Main fiber
        axL.plot(r3[:, 0], r3[:, 1], r3[:, 2], 
                linewidth=1.8, alpha=0.9, color=col, zorder=10)
        
        # Glow effect (multiple layers with increasing alpha)
        if glow_effect:
            for lw, alpha in [(4.0, 0.15), (2.8, 0.25)]:
                axL.plot(r3[:, 0], r3[:, 1], r3[:, 2],
                        linewidth=lw, alpha=alpha, color=col, zorder=5)
    
    # Add torus surface hints (transparent shells)
    if style == 'nested_tori' and show_linking:
        for torus_idx in range(n_tori):
            # Get fibers for this torus
            torus_fibers = [fibers_r3[i] for i, fd in enumerate(fiber_data) 
                          if fd['torus_idx'] == torus_idx]
            if len(torus_fibers) > 2:
                # Create a surface from the fibers
                all_pts = np.concatenate(torus_fibers, axis=0)
                # Just plot a subtle bounding effect
                cmap = torus_cmaps[torus_idx]
                center = all_pts.mean(axis=0)
                axL.scatter([center[0]], [center[1]], [center[2]], 
                           s=50, alpha=0.3, color=cmap(0.5), marker='o')
    
    axL.set_title(f"Hopf Fibers as Nested Tori\n({style} style, {len(fiber_data)} fibers)", 
                  fontsize=11, fontweight='bold', pad=10)
    axL.set_xlabel("X", labelpad=5)
    axL.set_ylabel("Y", labelpad=5)
    axL.set_zlabel("Z", labelpad=5)
    
    # Set consistent view and aspect
    axL.set_box_aspect((1, 1, 1))
    axL.view_init(elev=22, azim=45)
    
    # Auto-scale based on fiber extent
    all_r3 = np.concatenate(fibers_r3, axis=0)
    max_extent = np.abs(all_r3).max() * 1.1
    axL.set_xlim(-max_extent, max_extent)
    axL.set_ylim(-max_extent, max_extent)
    axL.set_zlim(-max_extent, max_extent)
    
    # ========== Middle & Right Panels: S^2 Sphere ==========
    
    # High-quality sphere wireframe
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    
    for ax in (axM, axR):
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, 
                         linewidth=0.3, alpha=0.2, color=wireframe_color)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=45)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel("$y_1$", labelpad=5)
        ax.set_ylabel("$y_2$", labelpad=5)
        ax.set_zlabel("$y_3$", labelpad=5)
    
    # Plot predictions on S^2
    for cloud, col, meanpt in zip(clouds_raw, fiber_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
                   s=8, alpha=0.5, color=col, edgecolors='none')
        axM.scatter(meanpt[0], meanpt[1], meanpt[2], 
                   s=80, alpha=0.95, color=col, edgecolors='white', linewidths=0.5)
    
    for cloud, col, meanpt in zip(clouds_phi, fiber_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
                   s=8, alpha=0.5, color=col, edgecolors='none')
        axR.scatter(meanpt[0], meanpt[1], meanpt[2], 
                   s=80, alpha=0.95, color=col, edgecolors='white', linewidths=0.5)
    
    axM.set_title("KAN Outputs (Raw 4D Input)\nFiber → Point Collapse Quality", 
                  fontsize=10, fontweight='bold', pad=10)
    axR.set_title("KAN Outputs (Engineered $\\Phi \\in \\mathbb{R}^5$)\nFiber → Point Collapse Quality", 
                  fontsize=10, fontweight='bold', pad=10)
    
    # ========== Statistics Box ==========
    stats_text = (
        f"{'─'*60}\n"
        f"RAW INPUT MODEL\n"
        f"  Angular Error: mean={np.mean(ang_raw):.2e} rad, p95={np.quantile(ang_raw, 0.95):.2e} rad\n"
        f"  Fiber RMS:     mean={rms_raw.mean():.2e}, max={rms_raw.max():.2e}\n"
        f"  Norm Deviation: mean |‖ŷ‖−1| = {norm_dev_raw:.2e}\n"
        f"{'─'*60}\n"
        f"ENGINEERED Φ MODEL\n"
        f"  Angular Error: mean={np.mean(ang_phi):.2e} rad, p95={np.quantile(ang_phi, 0.95):.2e} rad\n"
        f"  Fiber RMS:     mean={rms_phi.mean():.2e}, max={rms_phi.max():.2e}\n"
        f"  Norm Deviation: mean |‖ŷ‖−1| = {norm_dev_phi:.2e}"
    )
    
    # fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=8,
    #          family='monospace', color=text_color,
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, 
    #                   edgecolor=grid_color, alpha=0.9))
    
    # ========== Legend for Tori ==========
    if style == 'nested_tori':
        legend_patches = []
        for i in range(n_tori):
            cmap = torus_cmaps[i]
            patch = mpatches.Patch(color=cmap(0.5), label=f'Torus {i+1} (θ={fiber_data[i*fibers_per_torus]["theta"]/np.pi:.2f}π)')
            legend_patches.append(patch)
        axL.legend(handles=legend_patches, loc='upper left', fontsize=7, 
                  framealpha=0.7, facecolor=bg_color)
    
    plt.suptitle("Hopf Fibration: $S^3 \\to S^2$ — Fiber Bundle Structure Learned by KANs",
                 fontsize=13, fontweight='bold', y=0.98, color=text_color)
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300, facecolor=bg_color)
        print(f"Saved: {savepath}")
    
    plt.show()
    
    return {
        'rms_raw': rms_raw,
        'rms_phi': rms_phi,
        'ang_raw': ang_raw,
        'ang_phi': ang_phi,
    }


# -----------------------------
# ALTERNATIVE: Trefoil Knot View
# -----------------------------
def make_trefoil_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    n_fibers: int = 24,
    m: int = 800,
    savepath: str = None,
):
    """
    Visualize Hopf fibers as trefoil-linked circles.
    The Hopf fibration naturally produces linked circles that form trefoil-like structures.
    """
    torch.manual_seed(42)
    model_raw.eval()
    model_phi.eval()
    
    # Generate fibers on a trefoil path on S^2
    t = np.linspace(0, 2*np.pi, n_fibers, endpoint=False)
    
    # Trefoil knot parametrization on S^2
    # Map trefoil (p,q) torus knot to spherical coords
    p, q = 2, 3
    r_base = 0.6
    
    fibers_r3 = []
    fiber_colors = []
    
    cmap = plt.cm.twilight_shifted
    
    with torch.no_grad():
        for i, ti in enumerate(t):
            # Point on trefoil projected to S^2
            theta = np.pi/2 + 0.4 * np.sin(q * ti)
            phi = p * ti + 0.3 * np.cos(q * ti)
            
            fiber = hopf_fiber_parametric(theta, phi, m=m, device=device)
            r3 = stereographic_s3_to_r3_torch(fiber)
            fibers_r3.append(r3.detach().cpu().numpy())
            fiber_colors.append(cmap(i / n_fibers))
    
    # Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050510')
    
    for r3, col in zip(fibers_r3, fiber_colors):
        ax.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=2.0, alpha=0.9, color=col)
        # Glow
        ax.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=4.5, alpha=0.2, color=col)
    
    ax.set_title("Hopf Fibers: Trefoil-Linked Circles", fontsize=14, fontweight='bold')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=60)
    
    all_r3 = np.concatenate(fibers_r3, axis=0)
    max_ext = np.abs(all_r3).max() * 1.1
    ax.set_xlim(-max_ext, max_ext)
    ax.set_ylim(-max_ext, max_ext)
    ax.set_zlim(-max_ext, max_ext)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.1)
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='#050510')
        print(f"Saved: {savepath}")
    
    plt.show()


# -----------------------------
# Demo: Hopf Fiber Artistic Render
# -----------------------------
def render_hopf_artistic(
    n_tori: int = 5,
    fibers_per_torus: int = 20,
    m: int = 600,
    device: str = "cpu",
    savepath: str = None,
):
    """
    Standalone artistic rendering of nested tori Hopf fibers.
    No model required - pure geometry.
    """
    fiber_data = generate_nested_tori_fibers(
        n_tori=n_tori,
        fibers_per_torus=fibers_per_torus,
        m=m,
        device=device
    )
    
    torus_cmaps = create_torus_colormaps(n_tori)
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#020208')
    fig.patch.set_facecolor('#020208')
    
    all_r3 = []
    
    for fd in fiber_data:
        fiber_s3 = fd['fiber_s3']
        r3 = stereographic_s3_to_r3_torch(fiber_s3).detach().cpu().numpy()
        all_r3.append(r3)
        
        torus_idx = fd['torus_idx']
        fiber_idx = fd['fiber_idx']
        cmap = torus_cmaps[torus_idx]
        col = cmap(fiber_idx / fibers_per_torus)
        
        # Main line
        ax.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=1.5, alpha=0.85, color=col)
        
        # Glow layers
        for lw, alpha in [(4.0, 0.12), (2.5, 0.2)]:
            ax.plot(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=lw, alpha=alpha, color=col)
    
    all_r3 = np.concatenate(all_r3, axis=0)
    max_ext = np.abs(all_r3).max() * 1.05
    ax.set_xlim(-max_ext, max_ext)
    ax.set_ylim(-max_ext, max_ext)
    ax.set_zlim(-max_ext, max_ext)
    
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=45)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_axis_off()
    
    ax.set_title("Hopf Fibration: Nested Tori Structure\n$\\pi: S^3 \\to S^2$",
                 fontsize=16, fontweight='bold', color='#e8e8e8', pad=20)
    
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='#020208')
        print(f"Saved: {savepath}")
    
    plt.show()


# ============================================================
# MAIN EXECUTION - Example usage
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Enhanced Hopf Fibration Visualization")
    print("="*60)
    
    # First, let's render the pure geometric visualization
    print("\n[1] Rendering artistic Hopf fiber visualization...")
    render_hopf_artistic(
        n_tori=4,
        fibers_per_torus=18,
        m=600,
        device=str(device),
        savepath=None  # Set to "hopf_artistic.png" to save
    )
    
    print("\n[2] To use with your KAN models, run:")
    print("""
    # After training your models (model_raw, model_phi):
    
    results = make_joint_hopf_figure(
        model_raw, y_mu, y_std,
        model_phi, phi_mu, phi_std,
        device=device,
        n_tori=4,
        fibers_per_torus=16,
        style='nested_tori',  # or 'villarceau', 'classic'
        glow_effect=True,
        dark_theme=True,
        savepath="hopf_comparison.png"
    )
    
    # Or for trefoil visualization:
    make_trefoil_hopf_figure(
        model_raw, y_mu, y_std,
        model_phi, phi_mu, phi_std,
        device=device,
        n_fibers=24,
        savepath="hopf_trefoil.png"
    )
    """)
results = make_joint_hopf_figure(
        model_raw, y_mu, y_std,
        model_phi, phi_mu, phi_std,
        device=device,
        n_tori=4,
        fibers_per_torus=16,
        style='nested_tori',  # or 'villarceau', 'classic'
        glow_effect=True,
        dark_theme=False,
        savepath="hopf_comparison.png"
    )
    
# Or for trefoil visualization:
make_trefoil_hopf_figure(
        model_raw, y_mu, y_std,
        model_phi, phi_mu, phi_std,
        device=device,
        n_fibers=24,
        savepath="hopf_trefoil.png"
    )


# In[21]:


make_trefoil_hopf_figure(
        model_raw, y_mu, y_std,
        model_phi, phi_mu, phi_std,
        device=device,
        n_fibers=24,
        savepath="hopf_trefoil.png"
    )


# In[10]:


# ============================================================
# Minimal publication-grade loss plots
# One figure per loss curve
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

def to_np(loss):
    try:
        import torch
        if isinstance(loss, torch.Tensor):
            return loss.detach().cpu().numpy().ravel()
    except Exception:
        pass
    if isinstance(loss, dict):
        for k in ['train_loss', 'loss', 'loss_train', 'l']:
            if k in loss:
                return np.asarray(loss[k]).ravel()
        return np.asarray(list(loss.values())[0]).ravel()
    return np.asarray(loss).ravel()

def plot_single_loss(loss, title, filename, logy=True):
    y = to_np(loss)
    x = np.arange(len(y))

    # clean, journal-style defaults
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")

    if logy:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    fig.tight_layout()
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.png", bbox_inches="tight")
    plt.show()

# ---- plots ----
plot_single_loss(
    losss1,
    title="Training loss — raw 4D input KAN",
    filename="loss_raw_input_kan",
    logy=True,
)

plot_single_loss(
    loss2,
    title="Training loss — engineered Φ (5D) KAN",
    filename="loss_engineered_phi_kan",
    logy=True,
)


# In[5]:


# ============================================================
# Publication-grade loss curves for losss1 and loss2
# Run this cell after training cell
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

def _to_1d_np(loss_obj):
    """
    Robustly convert KAN.fit() return into a 1D numpy array.
    Handles list, numpy array, torch tensor, or dict-like logs.
    """
    if loss_obj is None:
        raise ValueError("Loss object is None.")
    # dict-like: try common keys
    if isinstance(loss_obj, dict):
        for k in ["loss", "train_loss", "loss_train", "l", "losss"]:
            if k in loss_obj:
                loss_obj = loss_obj[k]
                break
        else:
            # last resort: take first value that looks like a sequence
            for v in loss_obj.values():
                if hasattr(v, "__len__"):
                    loss_obj = v
                    break

    # torch tensor
    try:
        import torch
        if isinstance(loss_obj, torch.Tensor):
            return loss_obj.detach().cpu().flatten().numpy()
    except Exception:
        pass

    # numpy / list
    arr = np.asarray(loss_obj).reshape(-1)
    return arr.astype(np.float64)

def ema(x, alpha=0.15):
    """Exponential moving average smoothing (alpha in (0,1])."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def publication_loss_plot(loss_a, loss_b,
                          label_a="Raw-input KAN (4D→3D)",
                          label_b="Engineered Φ KAN (5D→3D)",
                          title="Training loss vs steps",
                          logy=True,
                          smooth_alpha=0.12,   # set to None to disable smoothing
                          drop_first=0,        # e.g. 1-5 if step0 is weird
                          save_prefix="kan_losses",
                          show=True):
    la = _to_1d_np(loss_a)
    lb = _to_1d_np(loss_b)

    if drop_first > 0:
        la = la[drop_first:]
        lb = lb[drop_first:]

    xa = np.arange(len(la))
    xb = np.arange(len(lb))

    if smooth_alpha is not None:
        la_s = ema(la, alpha=smooth_alpha)
        lb_s = ema(lb, alpha=smooth_alpha)
    else:
        la_s, lb_s = la, lb

    # ---- publication styling ----
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # plot raw lines lightly + smoothed strongly
    ax.plot(xa, la, alpha=0.25)
    ax.plot(xb, lb, alpha=0.25)
    ax.plot(xa, la_s, label=label_a)
    ax.plot(xb, lb_s, label=label_b)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")

    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    # tighten + export
    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_prefix}.png", bbox_inches="tight")
    print(f"Saved: {save_prefix}.pdf and {save_prefix}.png")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ---- call it ----
publication_loss_plot(losss1, loss2,
                      label_a="Raw-input KAN (4D input, normalized labels)",
                      label_b="Engineered Φ KAN (5D features, direct labels)",
                      title="KAN training loss (EMA-smoothed)",
                      logy=True,
                      smooth_alpha=0.12,
                      drop_first=0,
                      save_prefix="hopf_kan_loss_curves")


# In[7]:


"""
Publication-grade Hopf Fibration Visualization
===============================================
Drop-in replacement for make_joint_hopf_figure with:
- Nested tori structure (fibers organized by latitude circles on S²)
- Dark theme matching classic mathematical visualizations
- Improved color scheme and rendering
- Better axis styling for publication
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import Axes3D

# Torch import (required for model evaluation)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_joint_hopf_figure(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    # New parameters for nested tori
    num_tori=6,
    fibers_per_torus=14,
    m=700,
    s2_cloud_points_per_fiber=100,
    seed=0,
    savepath=None,
    # Style options
    style='dark',  # 'dark' or 'light'
    show_axis_fiber=True,
    view_elev=22,
    view_azim=42,
    # Pass in your helper functions
    sample_s2_basepoints_torch=None,
    section_s2_to_s3_torch=None,
    hopf_fiber_from_x0_torch=None,
    stereographic_s3_to_r3_torch=None,
    to_phi_norm=None,
    hopf_map_torch=None,
    angle_error=None,
):
    """
    Publication-grade Hopf fibration figure with nested tori structure.
    
    Creates a 3-panel figure:
    - Left: Hopf fibers in R³ (stereographic projection) organized as nested tori
    - Middle: Raw model outputs along fibers on S²
    - Right: Engineered model outputs along fibers on S²
    
    Parameters
    ----------
    num_tori : int
        Number of nested tori (latitude circles on S²)
    fibers_per_torus : int
        Number of fibers per torus
    style : str
        'dark' for dark background (like reference), 'light' for white background
    show_axis_fiber : bool
        Whether to show the vertical axis fiber (over poles)
    """
    
    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()
    
    # =========================================================================
    # Generate basepoints organized into nested tori
    # =========================================================================
    # Instead of random sampling, we sample along latitude circles on S²
    # Each latitude circle's fibers form a torus in S³
    
    torus_latitudes = np.linspace(0.15 * np.pi, 0.85 * np.pi, num_tori)
    
    all_basepoints = []
    torus_membership = []
    
    for torus_idx, theta in enumerate(torus_latitudes):
        z = np.cos(theta)
        r = np.sin(theta)
        phi_angles = np.linspace(0, 2 * np.pi, fibers_per_torus, endpoint=False)
        
        for phi in phi_angles:
            bp = torch.tensor(
                [r * np.cos(phi), r * np.sin(phi), z],
                dtype=torch.float32, device=device
            )
            all_basepoints.append(bp)
            torus_membership.append(torus_idx)
    
    base = torch.stack(all_basepoints)
    num_fibers = len(base)
    
    # =========================================================================
    # Color palette: warm inner → cool outer (like reference image)
    # =========================================================================
    inner_to_outer_colors = [
        '#ff6d00',  # Deep orange (innermost - near equator fibers)
        '#ffab00',  # Amber
        '#c6ff00',  # Lime
        '#00e676',  # Green
        '#00bcd4',  # Cyan
        '#2979ff',  # Blue
        '#7c4dff',  # Purple (outermost - near pole fibers)
    ]
    
    # Interpolate colors for our number of tori
    torus_colors = []
    for i in range(num_tori):
        t = i / max(1, num_tori - 1)
        idx = t * (len(inner_to_outer_colors) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(inner_to_outer_colors) - 1)
        frac = idx - idx_low
        c1 = np.array(to_rgb(inner_to_outer_colors[idx_low]))
        c2 = np.array(to_rgb(inner_to_outer_colors[idx_high]))
        torus_colors.append(c1 * (1 - frac) + c2 * frac)
    
    # =========================================================================
    # Precompute fiber geometry + predictions
    # =========================================================================
    fibers_r3 = []
    fiber_colors = []
    fiber_torus_idx = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []
    
    take = torch.linspace(0, m - 1, s2_cloud_points_per_fiber, device=device).long()
    
    with torch.no_grad():
        for i, y in enumerate(base):
            torus_idx = torus_membership[i]
            fiber_idx_in_torus = i % fibers_per_torus
            
            x0 = section_s2_to_s3_torch(y)
            fiber_s3 = hopf_fiber_from_x0_torch(x0, m=m)
            
            # Left panel: stereographic projection
            r3 = stereographic_s3_to_r3_torch(fiber_s3)
            fibers_r3.append(r3.detach().cpu().numpy())
            
            # Color based on torus membership with slight phase variation
            base_color = torus_colors[torus_idx]
            phase = fiber_idx_in_torus / fibers_per_torus
            variation = 0.08 * np.sin(2 * np.pi * phase)
            color = np.clip(base_color * (1 + variation), 0, 1)
            fiber_colors.append(color)
            fiber_torus_idx.append(torus_idx)
            
            # Raw model prediction
            yhat_raw = model_raw(fiber_s3) * y_std + y_mu
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())
            
            # Engineered model prediction
            Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)
            yhat_phi = model_phi(Phi_f_n)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())
    
    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)
    
    # =========================================================================
    # Compute angular statistics on random batch
    # =========================================================================
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)
        
        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(
            Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))
        
        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(
            Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))
    
    # =========================================================================
    # Publication-quality plotting setup
    # =========================================================================
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.6,
    })
    
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.06)
    
    axL = fig.add_subplot(gs[0, 0], projection='3d')
    axM = fig.add_subplot(gs[0, 1], projection='3d')
    axR = fig.add_subplot(gs[0, 2], projection='3d')
    
    # =========================================================================
    # Style configuration
    # =========================================================================
    if style == 'dark':
        bg_color = '#08080f'
        pane_color = '#0c0c18'
        grid_color = '#1a1a2e'
        text_color = '#e8e8e8'
        axis_color = '#666677'
        sphere_wire_color = '#555566'
        sphere_wire_alpha = 0.18
    else:
        bg_color = '#ffffff'
        pane_color = '#fafafa'
        grid_color = '#dddddd'
        text_color = '#222222'
        axis_color = '#444444'
        sphere_wire_color = '#888888'
        sphere_wire_alpha = 0.25
    
    fig.patch.set_facecolor(bg_color)
    
    for ax in [axL, axM, axR]:
        ax.set_facecolor(bg_color)
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor(pane_color)
        ax.yaxis.pane.set_facecolor(pane_color)
        ax.zaxis.pane.set_facecolor(pane_color)
        ax.xaxis.pane.set_edgecolor(grid_color)
        ax.yaxis.pane.set_edgecolor(grid_color)
        ax.zaxis.pane.set_edgecolor(grid_color)
        ax.xaxis._axinfo['grid']['color'] = grid_color
        ax.yaxis._axinfo['grid']['color'] = grid_color
        ax.zaxis._axinfo['grid']['color'] = grid_color
        ax.tick_params(colors=axis_color, labelsize=8)
        ax.xaxis.label.set_color(axis_color)
        ax.yaxis.label.set_color(axis_color)
        ax.zaxis.label.set_color(axis_color)
        ax.grid(True, alpha=0.25)
    
    # =========================================================================
    # Left panel: Hopf fibers in R³ with nested tori
    # =========================================================================
    
    # Plot fibers with depth-based line properties
    for r3, col, tidx in zip(fibers_r3, fiber_colors, fiber_torus_idx):
        # Filter extreme values from stereographic projection
        mask = np.all(np.abs(r3) < 10, axis=1)
        if mask.sum() < 20:
            continue
        pts = r3[mask]
        
        # Vary line properties by torus layer (outer tori = thinner, more transparent)
        depth_factor = tidx / max(1, num_tori - 1)
        lw = 1.4 - 0.4 * depth_factor
        alpha = 0.88 - 0.25 * depth_factor
        
        axL.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                linewidth=lw, alpha=alpha, color=col,
                solid_capstyle='round', solid_joinstyle='round')
    
    # Add axis fiber (fiber over north pole) - this is the vertical line
    if show_axis_fiber:
        # Fiber over point very close to north pole
        y_pole = torch.tensor([0.0, 0.0, 0.9999], dtype=torch.float32, device=device)
        with torch.no_grad():
            x0_pole = section_s2_to_s3_torch(y_pole)
            fiber_pole = hopf_fiber_from_x0_torch(x0_pole, m=m)
            r3_pole = stereographic_s3_to_r3_torch(fiber_pole).cpu().numpy()
        
        mask = np.all(np.abs(r3_pole) < 15, axis=1)
        if mask.sum() > 10:
            pts = r3_pole[mask]
            axL.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    linewidth=2.2, alpha=0.95, color='#ff3d00',
                    solid_capstyle='round', label='Axis fiber')
    
    axL.set_title('Hopf Fibers in $\\mathbb{R}^3$\n(Stereographic Projection of $S^3$)',
                 color=text_color, fontsize=11, pad=12)
    axL.set_xlabel('$x$', labelpad=4)
    axL.set_ylabel('$y$', labelpad=4)
    axL.set_zlabel('$z$', labelpad=4)
    axL.set_xlim([-4.5, 4.5])
    axL.set_ylim([-4.5, 4.5])
    axL.set_zlim([-4.5, 4.5])
    axL.view_init(elev=view_elev, azim=view_azim)
    axL.set_box_aspect([1, 1, 1])
    
    # =========================================================================
    # Unit sphere wireframe for middle/right panels
    # =========================================================================
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 22)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    
    for ax in [axM, axR]:
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2,
                         linewidth=0.3, alpha=sphere_wire_alpha,
                         color=sphere_wire_color)
        ax.set_xlabel('$y_1$', labelpad=3)
        ax.set_ylabel('$y_2$', labelpad=3)
        ax.set_zlabel('$y_3$', labelpad=3)
        ax.set_xlim([-1.45, 1.45])
        ax.set_ylim([-1.45, 1.45])
        ax.set_zlim([-1.45, 1.45])
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_box_aspect([1, 1, 1])
    
    # =========================================================================
    # Middle panel: Raw model outputs on S²
    # =========================================================================
    for cloud, col, meanpt in zip(clouds_raw, fiber_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                   s=7, alpha=0.5, color=col, edgecolors='none')
        axM.scatter(*meanpt, s=50, alpha=0.92, color=col,
                   edgecolors='white', linewidths=0.4)
    
    axM.set_title('KAN Outputs: Raw $x \\in S^3$\n(Fiber collapse to $S^2$)',
                 color=text_color, fontsize=11, pad=12)
    
    # =========================================================================
    # Right panel: Engineered model outputs on S²
    # =========================================================================
    for cloud, col, meanpt in zip(clouds_phi, fiber_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                   s=7, alpha=0.5, color=col, edgecolors='none')
        axR.scatter(*meanpt, s=50, alpha=0.92, color=col,
                   edgecolors='white', linewidths=0.4)
    
    axR.set_title('KAN Outputs: $\\Phi(x) \\in \\mathbb{R}^5$\n(Fiber-invariant features)',
                 color=text_color, fontsize=11, pad=12)
    
    # =========================================================================
    # Metrics annotation
    # =========================================================================
    metrics_text = (
        f"Raw 4D:  ⟨θ⟩ = {np.mean(ang_raw):.2e} rad  "
        f"(p₉₅ = {np.quantile(ang_raw, 0.95):.2e})   "
        f"fiber RMS = {rms_raw.mean():.2e}   "
        f"|‖ŷ‖−1| = {norm_dev_raw:.2e}\n"
        f"Φ-eng:   ⟨θ⟩ = {np.mean(ang_phi):.2e} rad  "
        f"(p₉₅ = {np.quantile(ang_phi, 0.95):.2e})   "
        f"fiber RMS = {rms_phi.mean():.2e}   "
        f"|‖ŷ‖−1| = {norm_dev_phi:.2e}"
    )
    
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom',
            color=axis_color, fontsize=8.5, family='monospace',
            linespacing=1.4)
    
    # Add subtle annotation about structure
    fig.text(0.02, 0.01,
            f'{num_tori} nested tori × {fibers_per_torus} fibers = {num_fibers} total',
            ha='left', va='bottom', color=axis_color, fontsize=8, style='italic')
    
    plt.tight_layout(rect=[0, 0.045, 1, 0.98])
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=400,
                   facecolor=fig.get_facecolor(), edgecolor='none',
                   pad_inches=0.08)
        print(f"Saved: {savepath}")
    
    plt.show()
    
    return fig


# =============================================================================
# Standalone demo (without model evaluation)
# =============================================================================

def demo_hopf_fibers_standalone(
    num_tori=7,
    fibers_per_torus=18,
    m=800,
    style='dark',
    view_elev=20,
    view_azim=38,
    savepath=None,
):
    """
    Create standalone Hopf fibration visualization (no models needed).
    """
    
    def hopf_fiber_s3(base_point, num_points=500):
        """Generate Hopf fiber over a point on S²."""
        x, y, z = base_point
        norm = np.sqrt(x*x + y*y + z*z)
        x, y, z = x/norm, y/norm, z/norm
        
        if z > -0.99:
            r1 = np.sqrt((1 + z) / 2)
            r2 = np.sqrt((1 - z) / 2)
            phase = np.arctan2(y, x) if r1 * r2 > 1e-10 else 0
            z1_base = r1 + 0j
            z2_base = r2 * np.exp(-1j * phase)
        else:
            z1_base = np.sqrt((1 + z) / 2) * np.exp(1j * np.arctan2(y, x) / 2)
            z2_base = np.sqrt((1 - z) / 2)
        
        theta = np.linspace(0, 2 * np.pi, num_points)
        rotation = np.exp(1j * theta)
        z1 = rotation * z1_base
        z2 = rotation * z2_base
        return np.stack([z1.real, z1.imag, z2.real, z2.imag], axis=1)
    
    def stereographic_projection(points_s3):
        """Stereographic projection from S³ to R³."""
        x1, x2, x3, x4 = points_s3[:, 0], points_s3[:, 1], points_s3[:, 2], points_s3[:, 3]
        denom = 1 - x4
        denom = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom + 1e-10), denom)
        return np.stack([x1 / denom, x2 / denom, x3 / denom], axis=1)
    
    # Generate basepoints for nested tori
    torus_latitudes = np.linspace(0.12 * np.pi, 0.88 * np.pi, num_tori)
    
    fibers = []
    torus_membership = []
    
    for torus_idx, theta in enumerate(torus_latitudes):
        z = np.cos(theta)
        r = np.sin(theta)
        phi_angles = np.linspace(0, 2 * np.pi, fibers_per_torus, endpoint=False)
        
        for phi in phi_angles:
            bp = np.array([r * np.cos(phi), r * np.sin(phi), z])
            fiber_s3 = hopf_fiber_s3(bp, m)
            fiber_r3 = stereographic_projection(fiber_s3)
            fibers.append(fiber_r3)
            torus_membership.append(torus_idx)
    
    # Color palette
    inner_to_outer = [
        '#ff6d00', '#ffab00', '#c6ff00', '#00e676',
        '#00bcd4', '#2979ff', '#7c4dff'
    ]
    
    torus_colors = []
    for i in range(num_tori):
        t = i / max(1, num_tori - 1)
        idx = t * (len(inner_to_outer) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(inner_to_outer) - 1)
        frac = idx - idx_low
        c1 = np.array(to_rgb(inner_to_outer[idx_low]))
        c2 = np.array(to_rgb(inner_to_outer[idx_high]))
        torus_colors.append(c1 * (1 - frac) + c2 * frac)
    
    # Plotting
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 11,
    })
    
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if style == 'dark':
        bg = '#08080f'
        pane = '#0c0c18'
        grid = '#1a1a2e'
        text = '#e8e8e8'
    else:
        bg = '#ffffff'
        pane = '#fafafa'
        grid = '#dddddd'
        text = '#222222'
    
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.xaxis.pane.set_facecolor(pane)
    ax.yaxis.pane.set_facecolor(pane)
    ax.zaxis.pane.set_facecolor(pane)
    ax.xaxis.pane.set_edgecolor(grid)
    ax.yaxis.pane.set_edgecolor(grid)
    ax.zaxis.pane.set_edgecolor(grid)
    
    # Plot fibers
    for i, (fiber, tidx) in enumerate(zip(fibers, torus_membership)):
        fidx = i % fibers_per_torus
        
        mask = np.all(np.abs(fiber) < 10, axis=1)
        if mask.sum() < 20:
            continue
        pts = fiber[mask]
        
        base_color = torus_colors[tidx]
        phase = fidx / fibers_per_torus
        color = np.clip(base_color * (1 + 0.08 * np.sin(2 * np.pi * phase)), 0, 1)
        
        depth = tidx / max(1, num_tori - 1)
        lw = 1.3 - 0.35 * depth
        alpha = 0.85 - 0.2 * depth
        
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
               color=color, linewidth=lw, alpha=alpha)
    
    # Axis fiber
    pole_fiber = hopf_fiber_s3((0, 0, 0.9999), m)
    pole_r3 = stereographic_projection(pole_fiber)
    mask = np.all(np.abs(pole_r3) < 15, axis=1)
    if mask.sum() > 10:
        ax.plot(pole_r3[mask, 0], pole_r3[mask, 1], pole_r3[mask, 2],
               color='#ff3d00', linewidth=2.2, alpha=0.95)
    
    ax.set_title('The Hopf Fibration\n$\\pi: S^3 \\to S^2$',
                color=text, fontsize=14, pad=15)
    ax.set_xlabel('$x$', color='#777', labelpad=6)
    ax.set_ylabel('$y$', color='#777', labelpad=6)
    ax.set_zlabel('$z$', color='#777', labelpad=6)
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_zlim([-4.5, 4.5])
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(colors='#666')
    ax.grid(True, alpha=0.25)
    
    fig.text(0.02, 0.02,
            f'{num_tori} nested tori × {fibers_per_torus} fibers\nStereographic projection to $\\mathbb{{R}}^3$',
            color='#666', fontsize=9, style='italic', ha='left', va='bottom')
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=400,
                   facecolor=fig.get_facecolor(), pad_inches=0.1)
        print(f"Saved: {savepath}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("Generating standalone Hopf fibration demo...")
    demo_hopf_fibers_standalone(
        num_tori=7,
        fibers_per_torus=20,
        m=800,
        style='dark',
        savepath='hopf_nested_tori.png'
    )


# In[6]:


"""
Publication-grade Hopf Fibration Visualization
===============================================
Creates a topologically interesting visualization with nested tori (Villarceau circles)
similar to classic mathematical visualizations of the Hopf fibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch

# =============================================================================
# Core Hopf Fibration Geometry
# =============================================================================

def hopf_fiber_s3(base_point, num_points=500):
    """
    Generate a Hopf fiber (great circle on S^3) over a given base point on S^2.
    
    The Hopf map π: S³ → S² is given by:
    π(z₁, z₂) = (2Re(z₁z̄₂), 2Im(z₁z̄₂), |z₁|² - |z₂|²)
    
    where we identify S³ ⊂ ℂ² with (z₁, z₂) satisfying |z₁|² + |z₂|² = 1.
    
    The fiber over a point (x, y, z) ∈ S² with z = |z₁|² - |z₂|² consists of all
    (e^{iθ} z₁, e^{iθ} z₂) for θ ∈ [0, 2π].
    """
    x, y, z = base_point
    # Normalize to ensure on S^2
    norm = np.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm
    
    # From the Hopf map inverse, find a point (z1, z2) in the fiber
    # |z1|² = (1+z)/2, |z2|² = (1-z)/2
    # For z close to -1, use different parametrization to avoid singularity
    
    if z > -0.99:
        # Standard parametrization
        r1 = np.sqrt((1 + z) / 2)
        r2 = np.sqrt((1 - z) / 2)
        # Phase of z1*conj(z2) determines (x, y)
        # x + iy = 2 * z1 * conj(z2)
        if r1 * r2 > 1e-10:
            phase = np.arctan2(y, x)
        else:
            phase = 0
        # Choose z1 real positive, z2 carries the phase
        z1_base = r1 + 0j
        z2_base = r2 * np.exp(-1j * phase)
    else:
        # Near south pole, different parametrization
        z1_base = np.sqrt((1 + z) / 2) * np.exp(1j * np.arctan2(y, x) / 2)
        z2_base = np.sqrt((1 - z) / 2)
    
    # Generate the fiber by rotating (z1, z2) → (e^{iθ}z1, e^{iθ}z2)
    theta = np.linspace(0, 2 * np.pi, num_points)
    rotation = np.exp(1j * theta)
    
    z1 = rotation * z1_base
    z2 = rotation * z2_base
    
    # Convert to real coordinates on S^3 ⊂ R^4
    fiber_s3 = np.stack([z1.real, z1.imag, z2.real, z2.imag], axis=1)
    return fiber_s3


def stereographic_projection(points_s3):
    """
    Stereographic projection from S³ to R³.
    Project from the point (0, 0, 0, 1) ∈ S³.
    
    For a point (x₁, x₂, x₃, x₄) on S³:
    π(x) = (x₁, x₂, x₃) / (1 - x₄)
    """
    x1, x2, x3, x4 = points_s3[:, 0], points_s3[:, 1], points_s3[:, 2], points_s3[:, 3]
    
    # Avoid division by zero near the projection point
    denom = 1 - x4
    denom = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom + 1e-10), denom)
    
    return np.stack([x1 / denom, x2 / denom, x3 / denom], axis=1)


def sample_torus_basepoints(major_angle, num_points, radius_on_s2=None):
    """
    Sample basepoints along a latitude circle on S² to create a torus in R³.
    
    When we take fibers over a circle on S², the union forms a torus in S³,
    which projects to a torus in R³.
    
    major_angle: angle from north pole (0 = north, π = south)
    """
    if radius_on_s2 is None:
        radius_on_s2 = np.sin(major_angle)
    
    z = np.cos(major_angle)
    r = np.sin(major_angle)
    
    phi = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    basepoints = np.stack([r * np.cos(phi), r * np.sin(phi), np.full_like(phi, z)], axis=1)
    return basepoints


def create_nested_tori_fibers(num_tori=5, fibers_per_torus=12, points_per_fiber=400):
    """
    Create fibers organized into nested tori for the classic Hopf fibration look.
    
    Returns list of (fiber_points_r3, torus_idx, fiber_idx) tuples
    """
    fibers = []
    
    # Angles for different "latitudes" on S² - these create nested tori
    # Distribute from near-north-pole to near-south-pole for variety
    torus_angles = np.linspace(0.15 * np.pi, 0.85 * np.pi, num_tori)
    
    for torus_idx, angle in enumerate(torus_angles):
        basepoints = sample_torus_basepoints(angle, fibers_per_torus)
        
        for fiber_idx, bp in enumerate(basepoints):
            fiber_s3 = hopf_fiber_s3(bp, points_per_fiber)
            fiber_r3 = stereographic_projection(fiber_s3)
            fibers.append({
                'points': fiber_r3,
                'torus_idx': torus_idx,
                'fiber_idx': fiber_idx,
                'basepoint': bp,
                'angle': angle
            })
    
    return fibers


def create_linked_rings_fibers(num_rings=6, points_per_fiber=500):
    """
    Create a set of linked fibers that showcase the linking number property.
    Any two distinct Hopf fibers are linked exactly once.
    """
    fibers = []
    
    # Choose basepoints spread across S²
    phi = np.linspace(0, 2 * np.pi, num_rings, endpoint=False)
    theta = np.linspace(0.2 * np.pi, 0.8 * np.pi, num_rings)
    
    for i in range(num_rings):
        # Spiral pattern on S²
        angle = theta[i]
        azimuth = phi[i] + 0.3 * i
        
        bp = np.array([
            np.sin(angle) * np.cos(azimuth),
            np.sin(angle) * np.sin(azimuth),
            np.cos(angle)
        ])
        
        fiber_s3 = hopf_fiber_s3(bp, points_per_fiber)
        fiber_r3 = stereographic_projection(fiber_s3)
        fibers.append({
            'points': fiber_r3,
            'basepoint': bp,
            'index': i
        })
    
    return fibers


# =============================================================================
# Visualization Functions
# =============================================================================

def create_publication_colormap():
    """Create an elegant colormap for the fibers."""
    # Deep blues to teals to gold - inspired by scientific visualization
    colors = [
        '#1a237e',  # Deep indigo
        '#0d47a1',  # Dark blue
        '#00838f',  # Teal
        '#00897b',  # Green-teal
        '#43a047',  # Green
        '#c0ca33',  # Lime
        '#fdd835',  # Yellow
        '#ffb300',  # Amber
    ]
    return LinearSegmentedColormap.from_list('hopf', colors, N=256)


def create_torus_colormap():
    """Colormap where each torus gets a distinct hue band."""
    # Distinct colors for each torus level
    return plt.cm.viridis


def plot_fiber_with_depth(ax, points, color, linewidth=1.2, alpha=0.85, zorder_base=0):
    """
    Plot a fiber with depth-based rendering for better 3D perception.
    """
    # Simple approach: plot as a single line with given properties
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 
            color=color, linewidth=linewidth, alpha=alpha, zorder=zorder_base)


def make_publication_hopf_figure(
    model_raw=None, y_mu=None, y_std=None,
    model_phi=None, phi_mu=None, phi_std=None,
    device=None,
    num_tori=6,
    fibers_per_torus=16,
    points_per_fiber=600,
    show_model_outputs=True,
    savepath=None,
):
    """
    Create a publication-grade Hopf fibration figure.
    
    If models are provided, creates a 3-panel figure:
    - Left: Hopf fibers in R³
    - Middle: Raw model outputs
    - Right: Engineered model outputs
    
    If no models provided, creates a single beautiful Hopf fibration plot.
    """
    
    # Generate nested tori fibers
    fibers = create_nested_tori_fibers(
        num_tori=num_tori,
        fibers_per_torus=fibers_per_torus,
        points_per_fiber=points_per_fiber
    )
    
    # Also add the special "axis" fiber (over north pole) and a few artistic fibers
    # The fiber over (0, 0, 1) is a vertical line through origin in stereographic proj
    special_fibers = []
    
    # Fibers over poles create vertical lines
    for pole in [(0, 0, 0.999), (0, 0, -0.999)]:
        fiber_s3 = hopf_fiber_s3(pole, points_per_fiber)
        fiber_r3 = stereographic_projection(fiber_s3)
        special_fibers.append({
            'points': fiber_r3,
            'basepoint': np.array(pole),
            'is_pole': True
        })
    
    # ==========================================================================
    # Publication-quality plot styling
    # ==========================================================================
    
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
    })
    
    # Determine figure layout based on whether models are provided
    if model_raw is not None and model_phi is not None and show_model_outputs:
        fig = plt.figure(figsize=(16, 5.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.08)
        ax_hopf = fig.add_subplot(gs[0, 0], projection='3d')
        ax_raw = fig.add_subplot(gs[0, 1], projection='3d')
        ax_phi = fig.add_subplot(gs[0, 2], projection='3d')
        axes = [ax_hopf, ax_raw, ax_phi]
    else:
        fig = plt.figure(figsize=(10, 9))
        ax_hopf = fig.add_subplot(111, projection='3d')
        axes = [ax_hopf]
        ax_raw = ax_phi = None
    
    # ==========================================================================
    # Color scheme based on torus membership and position within torus
    # ==========================================================================
    
    # Create a sophisticated color scheme
    torus_cmap = plt.cm.plasma  # Warm colors for inner to outer tori
    fiber_variation = 0.15  # Amount of color variation within each torus
    
    # Dark background for dramatic effect (like the reference image)
    for ax in axes:
        ax.set_facecolor('#0a0a12')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.xaxis._axinfo['grid']['color'] = '#222230'
        ax.yaxis._axinfo['grid']['color'] = '#222230'
        ax.zaxis._axinfo['grid']['color'] = '#222230'
        ax.grid(True, alpha=0.3)
    
    fig.patch.set_facecolor('#0a0a12')
    
    # ==========================================================================
    # Plot the Hopf fibers
    # ==========================================================================
    
    # Color palette inspired by the reference image
    # Outer tori: cyan/teal, Middle: green/yellow, Inner: orange/amber
    inner_to_outer_colors = [
        '#ff6f00',  # Amber (innermost)
        '#ffc107',  # Yellow
        '#8bc34a',  # Light green  
        '#4caf50',  # Green
        '#00bcd4',  # Cyan
        '#2196f3',  # Blue (outermost)
    ]
    
    # Interpolate for the number of tori we have
    from matplotlib.colors import to_rgb
    torus_colors = []
    for i in range(num_tori):
        t = i / max(1, num_tori - 1)
        idx = t * (len(inner_to_outer_colors) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(inner_to_outer_colors) - 1)
        frac = idx - idx_low
        c1 = np.array(to_rgb(inner_to_outer_colors[idx_low]))
        c2 = np.array(to_rgb(inner_to_outer_colors[idx_high]))
        torus_colors.append(c1 * (1 - frac) + c2 * frac)
    
    # Plot each fiber with smooth color gradients
    for fiber in fibers:
        torus_idx = fiber['torus_idx']
        fiber_idx = fiber['fiber_idx']
        points = fiber['points']
        
        # Base color from torus membership
        base_color = torus_colors[torus_idx]
        
        # Slight variation based on position within torus
        phase = fiber_idx / fibers_per_torus
        variation = 0.1 * np.sin(2 * np.pi * phase)
        color = np.clip(base_color + variation, 0, 1)
        
        # Vary linewidth based on torus (outer tori slightly thinner)
        lw = 1.3 - 0.1 * (torus_idx / num_tori)
        
        # Vary alpha for depth perception
        alpha = 0.75 + 0.2 * (1 - torus_idx / num_tori)
        
        ax_hopf.plot(points[:, 0], points[:, 1], points[:, 2],
                    color=color, linewidth=lw, alpha=alpha)
    
    # Plot pole fibers in contrasting color (red/orange axis)
    for sf in special_fibers:
        points = sf['points']
        # Filter out extreme values from stereographic projection
        mask = np.all(np.abs(points) < 10, axis=1)
        if mask.sum() > 10:
            pts = points[mask]
            ax_hopf.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color='#ff5722', linewidth=2.0, alpha=0.95)
    
    # ==========================================================================
    # Style the main Hopf plot
    # ==========================================================================
    
    ax_hopf.set_title('Hopf Fibration: $S^3 \\to S^2$\n(Stereographic Projection to $\\mathbb{R}^3$)',
                     color='white', fontsize=12, pad=15)
    
    # Set axis limits to frame the structure well
    ax_hopf.set_xlim([-3.5, 3.5])
    ax_hopf.set_ylim([-3.5, 3.5])
    ax_hopf.set_zlim([-3.5, 3.5])
    
    ax_hopf.set_xlabel('$x$', color='#888888', labelpad=5)
    ax_hopf.set_ylabel('$y$', color='#888888', labelpad=5)
    ax_hopf.set_zlabel('$z$', color='#888888', labelpad=5)
    
    # Tick colors
    ax_hopf.tick_params(colors='#666666')
    
    # Viewing angle for best nested torus visibility
    ax_hopf.view_init(elev=22, azim=45)
    ax_hopf.set_box_aspect([1, 1, 1])
    
    # ==========================================================================
    # If models provided, plot their outputs
    # ==========================================================================
    
    if ax_raw is not None and ax_phi is not None and model_raw is not None:
        model_raw.eval()
        model_phi.eval()
        
        # Use the same fiber basepoints but compute model predictions
        s2_cloud_points_per_fiber = 80
        
        clouds_raw, clouds_phi = [], []
        means_raw, means_phi = [], []
        rms_raw_list, rms_phi_list = [], []
        cloud_colors = []
        
        # Sample from each torus
        sampled_fibers = fibers[::max(1, len(fibers) // 20)]  # Subsample for clarity
        
        with torch.no_grad():
            for fiber in sampled_fibers:
                bp = fiber['basepoint']
                torus_idx = fiber['torus_idx']
                
                # Generate fiber on S³
                fiber_s3_np = hopf_fiber_s3(bp, points_per_fiber)
                fiber_s3 = torch.tensor(fiber_s3_np, dtype=torch.float32, device=device)
                
                # Subsample for plotting
                take = torch.linspace(0, points_per_fiber - 1, s2_cloud_points_per_fiber).long()
                
                # Raw model prediction
                yhat_raw = model_raw(fiber_s3) * y_std + y_mu
                mean_raw = yhat_raw.mean(dim=0, keepdim=True)
                rms_raw_list.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
                clouds_raw.append(yhat_raw[take].cpu().numpy())
                means_raw.append(mean_raw.squeeze(0).cpu().numpy())
                
                # Engineered model prediction
                # Assuming to_phi_norm is available
                from hopf_utils import to_phi_norm  # You'll need this
                Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)
                yhat_phi = model_phi(Phi_f_n)
                mean_phi = yhat_phi.mean(dim=0, keepdim=True)
                rms_phi_list.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
                clouds_phi.append(yhat_phi[take].cpu().numpy())
                means_phi.append(mean_phi.squeeze(0).cpu().numpy())
                
                cloud_colors.append(torus_colors[torus_idx])
        
        # Unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        
        for ax in [ax_raw, ax_phi]:
            ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2,
                            linewidth=0.3, alpha=0.2, color='#666666')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_xlabel('$y_1$', color='#888888')
            ax.set_ylabel('$y_2$', color='#888888')
            ax.set_zlabel('$y_3$', color='#888888')
            ax.tick_params(colors='#666666')
            ax.view_init(elev=22, azim=45)
            ax.set_box_aspect([1, 1, 1])
        
        # Plot raw model outputs
        for cloud, color, mean_pt in zip(clouds_raw, cloud_colors, means_raw):
            ax_raw.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                          s=8, alpha=0.5, color=color)
            ax_raw.scatter(*mean_pt, s=50, alpha=0.9, color=color,
                          edgecolors='white', linewidths=0.5)
        
        ax_raw.set_title('Raw 4D Input: $f(x) \\mapsto S^2$\nFiber collapse quality',
                        color='white', fontsize=11, pad=10)
        
        # Plot engineered model outputs
        for cloud, color, mean_pt in zip(clouds_phi, cloud_colors, means_phi):
            ax_phi.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                          s=8, alpha=0.5, color=color)
            ax_phi.scatter(*mean_pt, s=50, alpha=0.9, color=color,
                          edgecolors='white', linewidths=0.5)
        
        ax_phi.set_title('Engineered $\\Phi \\in \\mathbb{R}^5$: $g(\\Phi(x)) \\mapsto S^2$\nFiber collapse quality',
                        color='white', fontsize=11, pad=10)
        
        # Add metrics annotation
        rms_raw_arr = np.array(rms_raw_list)
        rms_phi_arr = np.array(rms_phi_list)
        
        metrics_text = (
            f"Mean fiber RMS — Raw: {rms_raw_arr.mean():.2e}  |  "
            f"Engineered: {rms_phi_arr.mean():.2e}"
        )
        fig.text(0.5, 0.02, metrics_text, ha='center', color='#aaaaaa',
                fontsize=9, family='monospace')
    
    # ==========================================================================
    # Add legend/annotation
    # ==========================================================================
    
    # Create a subtle annotation about the structure
    annotation = (
        "Fibers organized into nested tori\n"
        "Each torus: fibers over a latitude circle on $S^2$"
    )
    fig.text(0.02, 0.02, annotation, ha='left', va='bottom',
            color='#666666', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300,
                   facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved: {savepath}")
    
    plt.show()
    return fig


def make_standalone_hopf_visualization(
    num_tori=7,
    fibers_per_torus=20,
    points_per_fiber=800,
    savepath=None,
    view_angle=(20, 35),
    figsize=(12, 10),
    style='dark',  # 'dark' or 'light'
):
    """
    Create a standalone, publication-quality Hopf fibration visualization.
    
    This creates a beautiful nested torus structure similar to classic
    mathematical visualizations.
    """
    
    # Generate nested tori fibers
    fibers = create_nested_tori_fibers(
        num_tori=num_tori,
        fibers_per_torus=fibers_per_torus,
        points_per_fiber=points_per_fiber
    )
    
    # Publication styling
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
    })
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Color and style setup
    if style == 'dark':
        bg_color = '#050510'
        text_color = '#e0e0e0'
        grid_color = '#1a1a2e'
        pane_color = '#0a0a15'
        
        # Vibrant colors on dark background
        inner_to_outer = [
            '#ff9800',  # Amber
            '#ffeb3b',  # Yellow
            '#8bc34a',  # Light green
            '#26c6da',  # Cyan
            '#42a5f5',  # Blue
            '#7e57c2',  # Purple
            '#ab47bc',  # Magenta
        ]
    else:
        bg_color = '#fafafa'
        text_color = '#333333'
        grid_color = '#dddddd'
        pane_color = '#f5f5f5'
        
        # Deeper colors on light background
        inner_to_outer = [
            '#e65100',
            '#f57c00',
            '#388e3c',
            '#0097a7',
            '#1976d2',
            '#512da8',
            '#7b1fa2',
        ]
    
    # Set background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Configure panes
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_facecolor(pane_color)
    ax.xaxis.pane.set_edgecolor(grid_color)
    ax.yaxis.pane.set_edgecolor(grid_color)
    ax.zaxis.pane.set_edgecolor(grid_color)
    
    # Grid styling
    ax.xaxis._axinfo['grid']['color'] = grid_color
    ax.yaxis._axinfo['grid']['color'] = grid_color
    ax.zaxis._axinfo['grid']['color'] = grid_color
    ax.grid(True, alpha=0.3 if style == 'dark' else 0.5)
    
    # Interpolate colors for tori
    from matplotlib.colors import to_rgb
    torus_colors = []
    for i in range(num_tori):
        t = i / max(1, num_tori - 1)
        idx = t * (len(inner_to_outer) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(inner_to_outer) - 1)
        frac = idx - idx_low
        c1 = np.array(to_rgb(inner_to_outer[idx_low]))
        c2 = np.array(to_rgb(inner_to_outer[idx_high]))
        torus_colors.append(tuple(c1 * (1 - frac) + c2 * frac))
    
    # Plot fibers
    for fiber in fibers:
        torus_idx = fiber['torus_idx']
        fiber_idx = fiber['fiber_idx']
        points = fiber['points']
        
        # Filter extreme points from stereographic projection
        mask = np.all(np.abs(points) < 8, axis=1)
        if mask.sum() < 10:
            continue
        pts = points[mask]
        
        # Color with subtle variation
        base_color = np.array(torus_colors[torus_idx])
        phase = fiber_idx / fibers_per_torus
        variation = 0.08 * np.sin(2 * np.pi * phase)
        color = tuple(np.clip(base_color * (1 + variation), 0, 1))
        
        # Line properties vary by torus layer
        lw = 1.0 + 0.3 * (1 - torus_idx / num_tori)
        alpha = 0.7 + 0.25 * (1 - torus_idx / num_tori)
        
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
               color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')
    
    # Add the vertical "axis" fiber (fiber over north pole)
    axis_fiber = hopf_fiber_s3((0, 0, 0.9999), points_per_fiber)
    axis_r3 = stereographic_projection(axis_fiber)
    mask = np.all(np.abs(axis_r3) < 15, axis=1)
    if mask.sum() > 10:
        ax.plot(axis_r3[mask, 0], axis_r3[mask, 1], axis_r3[mask, 2],
               color='#ff3d00' if style == 'dark' else '#d32f2f',
               linewidth=2.5, alpha=0.9, label='Axis fiber')
    
    # Axis configuration
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    
    ax.set_xlabel('$x$', color=text_color, fontsize=12, labelpad=8)
    ax.set_ylabel('$y$', color=text_color, fontsize=12, labelpad=8)
    ax.set_zlabel('$z$', color=text_color, fontsize=12, labelpad=8)
    
    ax.tick_params(colors=text_color if style == 'light' else '#888888', labelsize=9)
    
    # Title
    ax.set_title(
        'The Hopf Fibration\n'
        '$\\pi: S^3 \\rightarrow S^2$',
        color=text_color, fontsize=14, pad=20, fontweight='normal'
    )
    
    # View angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.set_box_aspect([1, 1, 1])
    
    # Annotation
    description = (
        f"Stereographic projection of {num_tori} nested tori\n"
        f"({fibers_per_torus} fibers each, {num_tori * fibers_per_torus} total)"
    )
    fig.text(0.02, 0.02, description, ha='left', va='bottom',
            color='#888888' if style == 'dark' else '#666666',
            fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=400,
                   facecolor=fig.get_facecolor(), edgecolor='none',
                   pad_inches=0.1)
        print(f"Saved: {savepath}")
    
    plt.show()
    return fig


# =============================================================================
# Modified version of the original function for integration
# =============================================================================

def make_joint_hopf_figure_publication(
    model_raw, y_mu, y_std,
    model_phi, phi_mu, phi_std,
    device,
    num_tori=6,
    fibers_per_torus=14,
    points_per_fiber=600,
    s2_cloud_points_per_fiber=80,
    seed=0,
    savepath=None,
    # Import these from your codebase:
    hopf_fiber_from_x0_torch=None,
    section_s2_to_s3_torch=None,
    stereographic_s3_to_r3_torch=None,
    sample_s2_basepoints_torch=None,
    to_phi_norm=None,
    hopf_map_torch=None,
    angle_error=None,
):
    """
    Publication-grade version of make_joint_hopf_figure with nested tori structure.
    
    Drop-in replacement that produces more topologically interesting visualizations.
    """
    torch.manual_seed(seed)
    model_raw.eval()
    model_phi.eval()
    
    # Generate basepoints organized into nested tori (latitude circles on S²)
    torus_angles = np.linspace(0.2 * np.pi, 0.8 * np.pi, num_tori)
    
    all_basepoints = []
    torus_membership = []
    
    for torus_idx, angle in enumerate(torus_angles):
        z = np.cos(angle)
        r = np.sin(angle)
        phi_angles = np.linspace(0, 2 * np.pi, fibers_per_torus, endpoint=False)
        
        for phi in phi_angles:
            bp = torch.tensor([r * np.cos(phi), r * np.sin(phi), z],
                            dtype=torch.float32, device=device)
            all_basepoints.append(bp)
            torus_membership.append(torus_idx)
    
    base = torch.stack(all_basepoints)
    
    # Precompute fiber geometry + predictions
    fibers_r3 = []
    fiber_colors = []
    clouds_raw, means_raw, rms_raw = [], [], []
    clouds_phi, means_phi, rms_phi = [], [], []
    
    take_indices = torch.linspace(0, points_per_fiber - 1, s2_cloud_points_per_fiber,
                                  device=device).long()
    
    # Color palette for tori (inner to outer)
    inner_to_outer = [
        '#ff9800', '#ffc107', '#8bc34a', '#26c6da', '#42a5f5', '#7e57c2'
    ]
    from matplotlib.colors import to_rgb
    torus_colors = []
    for i in range(num_tori):
        t = i / max(1, num_tori - 1)
        idx = t * (len(inner_to_outer) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(inner_to_outer) - 1)
        frac = idx - idx_low
        c1 = np.array(to_rgb(inner_to_outer[idx_low]))
        c2 = np.array(to_rgb(inner_to_outer[idx_high]))
        torus_colors.append(c1 * (1 - frac) + c2 * frac)
    
    with torch.no_grad():
        for i, y in enumerate(base):
            torus_idx = torus_membership[i]
            
            x0 = section_s2_to_s3_torch(y)
            fiber_s3 = hopf_fiber_from_x0_torch(x0, m=points_per_fiber)
            
            # Stereographic projection for R³ plot
            r3 = stereographic_s3_to_r3_torch(fiber_s3)
            fibers_r3.append(r3.detach().cpu().numpy())
            
            # Color based on torus membership
            fiber_colors.append(torus_colors[torus_idx])
            
            # Raw model prediction
            yhat_raw = model_raw(fiber_s3) * y_std + y_mu
            mean_raw = yhat_raw.mean(dim=0, keepdim=True)
            rms_raw.append(torch.sqrt(((yhat_raw - mean_raw) ** 2).mean()).item())
            clouds_raw.append(yhat_raw[take_indices].detach().cpu().numpy())
            means_raw.append(mean_raw.squeeze(0).detach().cpu().numpy())
            
            # Engineered model prediction
            Phi_f_n = to_phi_norm(fiber_s3, phi_mu, phi_std)
            yhat_phi = model_phi(Phi_f_n)
            mean_phi = yhat_phi.mean(dim=0, keepdim=True)
            rms_phi.append(torch.sqrt(((yhat_phi - mean_phi) ** 2).mean()).item())
            clouds_phi.append(yhat_phi[take_indices].detach().cpu().numpy())
            means_phi.append(mean_phi.squeeze(0).detach().cpu().numpy())
    
    rms_raw = np.array(rms_raw)
    rms_phi = np.array(rms_phi)
    
    # Compute angular statistics
    with torch.no_grad():
        X = torch.randn(60000, 4, device=device)
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        Ytrue = hopf_map_torch(X)
        
        Yhat_raw = model_raw(X) * y_std + y_mu
        ang_raw = angle_error(Yhat_raw, Ytrue).detach().cpu().numpy()
        norm_dev_raw = float(np.mean(np.abs(
            Yhat_raw.norm(dim=1).detach().cpu().numpy() - 1.0)))
        
        Phi_n = to_phi_norm(X, phi_mu, phi_std)
        Yhat_phi = model_phi(Phi_n)
        ang_phi = angle_error(Yhat_phi, Ytrue).detach().cpu().numpy()
        norm_dev_phi = float(np.mean(np.abs(
            Yhat_phi.norm(dim=1).detach().cpu().numpy() - 1.0)))
    
    # ==========================================================================
    # Publication-quality plotting
    # ==========================================================================
    
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })
    
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.06)
    
    axL = fig.add_subplot(gs[0, 0], projection='3d')
    axM = fig.add_subplot(gs[0, 1], projection='3d')
    axR = fig.add_subplot(gs[0, 2], projection='3d')
    
    # Dark theme
    bg_color = '#06060f'
    fig.patch.set_facecolor(bg_color)
    
    for ax in [axL, axM, axR]:
        ax.set_facecolor(bg_color)
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('#0a0a15')
        ax.yaxis.pane.set_facecolor('#0a0a15')
        ax.zaxis.pane.set_facecolor('#0a0a15')
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_edgecolor('#222233')
            axis._axinfo['grid']['color'] = '#1a1a2a'
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='#666666', labelsize=8)
    
    # Left panel: Hopf fibers in R³
    for r3, col, torus_idx in zip(fibers_r3, fiber_colors, torus_membership):
        # Filter extreme values
        mask = np.all(np.abs(r3) < 8, axis=1)
        if mask.sum() > 10:
            pts = r3[mask]
            lw = 1.2 - 0.05 * torus_idx
            alpha = 0.8 - 0.05 * torus_idx
            axL.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    linewidth=lw, alpha=alpha, color=col, solid_capstyle='round')
    
    axL.set_title('Hopf Fibers in $\\mathbb{R}^3$\n(Stereographic Projection)',
                 color='#e0e0e0', fontsize=11, pad=12)
    axL.set_xlabel('$x$', color='#888888', labelpad=5)
    axL.set_ylabel('$y$', color='#888888', labelpad=5)
    axL.set_zlabel('$z$', color='#888888', labelpad=5)
    axL.set_xlim([-4, 4])
    axL.set_ylim([-4, 4])
    axL.set_zlim([-4, 4])
    axL.view_init(elev=20, azim=40)
    axL.set_box_aspect([1, 1, 1])
    
    # Unit sphere wireframe for S² plots
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    
    for ax in [axM, axR]:
        ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2,
                         linewidth=0.25, alpha=0.15, color='#666666')
        ax.set_xlabel('$y_1$', color='#888888', labelpad=3)
        ax.set_ylabel('$y_2$', color='#888888', labelpad=3)
        ax.set_zlabel('$y_3$', color='#888888', labelpad=3)
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])
        ax.set_zlim([-1.4, 1.4])
        ax.view_init(elev=20, azim=40)
        ax.set_box_aspect([1, 1, 1])
    
    # Middle panel: Raw model outputs
    for cloud, col, meanpt in zip(clouds_raw, fiber_colors, means_raw):
        axM.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                   s=6, alpha=0.45, color=col, edgecolors='none')
        axM.scatter(*meanpt, s=45, alpha=0.9, color=col,
                   edgecolors='white', linewidths=0.3)
    
    axM.set_title('KAN Outputs: Raw $x \\in S^3$\n(Should collapse to points on $S^2$)',
                 color='#e0e0e0', fontsize=11, pad=12)
    
    # Right panel: Engineered model outputs
    for cloud, col, meanpt in zip(clouds_phi, fiber_colors, means_phi):
        axR.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                   s=6, alpha=0.45, color=col, edgecolors='none')
        axR.scatter(*meanpt, s=45, alpha=0.9, color=col,
                   edgecolors='white', linewidths=0.3)
    
    axR.set_title('KAN Outputs: Engineered $\\Phi(x) \\in \\mathbb{R}^5$\n(Fiber-invariant features)',
                 color='#e0e0e0', fontsize=11, pad=12)
    
    # Metrics annotation at bottom
    metrics_raw = (f"Raw: ⟨θ⟩={np.mean(ang_raw):.2e} rad, "
                   f"p₉₅={np.quantile(ang_raw, 0.95):.2e}, "
                   f"RMS={rms_raw.mean():.2e}, "
                   f"|‖ŷ‖−1|={norm_dev_raw:.2e}")
    metrics_phi = (f"Φ-eng: ⟨θ⟩={np.mean(ang_phi):.2e} rad, "
                   f"p₉₅={np.quantile(ang_phi, 0.95):.2e}, "
                   f"RMS={rms_phi.mean():.2e}, "
                   f"|‖ŷ‖−1|={norm_dev_phi:.2e}")
    
    fig.text(0.5, 0.01, f"{metrics_raw}    |    {metrics_phi}",
            ha='center', color='#999999', fontsize=8.5, family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=400,
                   facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved: {savepath}")
    
    plt.show()
    return fig


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    # Create standalone visualization
    print("Generating publication-grade Hopf fibration visualization...")
    
    fig = make_standalone_hopf_visualization(
        num_tori=7,
        fibers_per_torus=18,
        points_per_fiber=700,
        view_angle=(22, 40),
        figsize=(11, 9),
        style='dark',
        savepath='hopf_fibration_publication.png'
    )
    
    print("\nTo use with your models, call:")
    print("  make_joint_hopf_figure_publication(model_raw, y_mu, y_std, ...)")


# In[ ]:




