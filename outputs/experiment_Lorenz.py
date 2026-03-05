I now have all four sections of the script from the tool calls. I will assemble the complete, final, runnable script by combining all parts.

---

```python
#!/usr/bin/env python3
"""
lorenz_kanddy.py
================
KANDy experiment for the Lorenz system.

Algorithm: x_dot = A * Psi(phi(x))
  - phi : Koopman lift  R^3 -> R^9  (full degree-2 polynomial basis)
  - Psi : separable spline map from a SINGLE-LAYER KAN  width=[9, 3]
  - A   : linear mixing matrix extracted from KAN output weights

Lorenz system:
  x_dot = sigma*(y - x)
  y_dot = rho*x - x*z - y
  z_dot = x*y - beta*z

  sigma=10, rho=28, beta=8/3

phi(x,y,z) = [x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]   dim=9
KAN width = [9, 3]

Author : KANDy Research Team
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import os
import random
import json
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    warnings.warn("SymPy not found. Symbolic extraction will be skipped.")

try:
    from kan import KAN
except ImportError:
    raise ImportError(
        "PyKAN not found. Install with: pip install pykan"
    )

# ===========================================================================
# GLOBAL CONSTANTS
# ===========================================================================
SEED        = 42
SIGMA       = 10.0
RHO         = 28.0
BETA        = 8.0 / 3.0
T_TRAIN_END = 10.0
N_POINTS    = 2000
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC = 0.15  (remainder)
IC_TRAIN    = [1.0, 1.0, 1.0]
IC_TEST     = [1.5, 1.5, 1.5]
T_TEST_END  = 20.0
N_TEST_PTS  = 4000
GRID        = 5
K_SPLINE    = 3
STEPS       = 500
RESULTS_DIR = os.path.join('results', 'Lorenz')
PHI_DIM     = 9
STATE_DIM   = 3

PHI_FEATURE_NAMES = [
    'x',   'y',   'z',
    'x^2', 'x*y', 'x*z',
    'y^2', 'y*z', 'z^2',
]

assert len(PHI_FEATURE_NAMES) == PHI_DIM, "PHI_FEATURE_NAMES length mismatch"


# ===========================================================================
# SECTION 0 — REPRODUCIBILITY
# ===========================================================================

def set_all_seeds(seed: int = SEED) -> None:
    """Fix ALL random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[SEED] All random seeds fixed to {seed}.")


# ===========================================================================
# SECTION 1 — DATA GENERATION
# ===========================================================================

def lorenz_rhs(t: float,
               state: np.ndarray,
               sigma: float = SIGMA,
               rho:   float = RHO,
               beta:  float = BETA) -> List[float]:
    """
    True Lorenz right-hand side.

      x_dot = sigma*(y - x)
      y_dot = rho*x - x*z - y
      z_dot = x*y - beta*z
    """
    x, y, z = state
    xdot = sigma * (y - x)
    ydot = rho * x - x * z - y
    zdot = x * y - beta * z
    return [xdot, ydot, zdot]


def generate_lorenz_data(
        ic:      List[float] = IC_TRAIN,
        t_end:   float       = T_TRAIN_END,
        n_pts:   int         = N_POINTS,
        sigma:   float       = SIGMA,
        rho:     float       = RHO,
        beta:    float       = BETA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the Lorenz system with RK45 at high tolerance.

    Returns
    -------
    t     : shape (n_pts,)
    traj  : shape (n_pts, 3)   columns = [x, y, z]
    xdot  : shape (n_pts, 3)   evaluated from true RHS (exact derivatives)
    """
    t_eval = np.linspace(0.0, t_end, n_pts)

    sol = solve_ivp(
        fun       = lambda t, s: lorenz_rhs(t, s, sigma, rho, beta),
        t_span    = (0.0, t_end),
        y0        = ic,
        method    = 'RK45',
        t_eval    = t_eval,
        rtol      = 1e-10,
        atol      = 1e-12,
        dense_output = False,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    traj = sol.y.T                           # (n_pts, 3)

    # Exact derivatives via true RHS — avoids finite-difference noise
    xdot = np.array([
        lorenz_rhs(t_eval[i], traj[i], sigma, rho, beta)
        for i in range(n_pts)
    ])                                       # (n_pts, 3)

    print(f"[DATA] Generated {n_pts} points over t=[0, {t_end}].")
    print(f"       Trajectory range — "
          f"x:[{traj[:,0].min():.2f},{traj[:,0].max():.2f}]  "
          f"y:[{traj[:,1].min():.2f},{traj[:,1].max():.2f}]  "
          f"z:[{traj[:,2].min():.2f},{traj[:,2].max():.2f}]")
    return t_eval, traj, xdot


def split_data(
        traj:  np.ndarray,
        xdot:  np.ndarray,
        train_frac: float = TRAIN_FRAC,
        val_frac:   float = VAL_FRAC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological 70/15/15 train/val/test split.
    (No shuffling — preserves temporal order.)

    Returns
    -------
    x_train, x_val, x_test  : state splits,      shapes (N_*, 3)
    d_train, d_val, d_test   : derivative splits, shapes (N_*, 3)
    """
    n   = len(traj)
    n1  = int(n * train_frac)
    n2  = int(n * (train_frac + val_frac))

    x_train, d_train = traj[:n1],    xdot[:n1]
    x_val,   d_val   = traj[n1:n2],  xdot[n1:n2]
    x_test,  d_test  = traj[n2:],    xdot[n2:]

    print(f"[SPLIT] train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")
    return x_train, x_val, x_test, d_train, d_val, d_test


# ===========================================================================
# SECTION 2 — KOOPMAN LIFT   phi: R^3 -> R^9
# ===========================================================================

def phi(state: np.ndarray) -> np.ndarray:
    """
    Full degree-2 polynomial Koopman lift for the Lorenz system.

    phi(x,y,z) = [x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]

    Term-by-term justification from the Lorenz RHS:
      x     — appears in x_dot = sigma*(y-x)  and  y_dot = rho*x - y
      y     — appears in x_dot = sigma*(y-x)  and  z_dot = x*y - beta*z
      z     — appears in z_dot = -beta*z
      x^2   — included for completeness of degree-2 basis
      x*y   — MANDATORY: appears in z_dot = x*y - beta*z
      x*z   — MANDATORY: appears in y_dot = rho*x - x*z - y
      y^2   — included for completeness of degree-2 basis
      y*z   — included for completeness of degree-2 basis
      z^2   — included for completeness of degree-2 basis

    dim(phi) = 9  =>  KAN width = [9, 3]

    Parameters
    ----------
    state : np.ndarray, shape (3,) or (N, 3)

    Returns
    -------
    lifted : np.ndarray, shape (9,) or (N, 9)
    """
    scalar = state.ndim == 1
    if scalar:
        state = state[None, :]               # (1, 3)

    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]

    lifted = np.column_stack([
        x,           # feature 0 : x       (x_dot, y_dot)
        y,           # feature 1 : y       (x_dot, z_dot)
        z,           # feature 2 : z       (z_dot: -beta*z)
        x * x,       # feature 3 : x^2    (degree-2 completeness)
        x * y,       # feature 4 : x*y    MANDATORY — z_dot = x*y - beta*z
        x * z,       # feature 5 : x*z    MANDATORY — y_dot = rho*x - x*z - y
        y * y,       # feature 6 : y^2    (degree-2 completeness)
        y * z,       # feature 7 : y*z    (degree-2 completeness)
        z * z,       # feature 8 : z^2    (degree-2 completeness)
    ])               # shape (N, 9)

    return lifted[0] if scalar else lifted


def apply_phi_to_splits(
        x_train: np.ndarray,
        x_val:   np.ndarray,
        x_test:  np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply phi to all data splits."""
    theta_train = phi(x_train)   # (N_train, 9)
    theta_val   = phi(x_val)     # (N_val,   9)
    theta_test  = phi(x_test)    # (N_test,  9)

    print(f"[LIFT] phi applied: "
          f"train {theta_train.shape}, "
          f"val {theta_val.shape}, "
          f"test {theta_test.shape}.")
    return theta_train, theta_val, theta_test


# ===========================================================================
# SECTION 3 — DATASET CONSTRUCTION
# ===========================================================================

def build_dataset(
        theta_train: np.ndarray, d_train: np.ndarray,
        theta_val:   np.ndarray, d_val:   np.ndarray,
        theta_test:  np.ndarray, d_test:  np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Build the PyKAN dataset dict with torch.FloatTensors.

    Keys
    ----
    train_input  : (N_train, 9)   lifted state
    train_label  : (N_train, 3)   time derivatives  x_dot
    val_input    : (N_val,   9)
    val_label    : (N_val,   3)
    test_input   : (N_test,  9)
    test_label   : (N_test,  3)
    """
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    dataset = {
        'train_input': _t(theta_train),
        'train_label': _t(d_train),
        'val_input':   _t(theta_val),
        'val_label':   _t(d_val),
        'test_input':  _t(theta_test),
        'test_label':  _t(d_test),
    }

    print(f"[DATASET] train_input  : {dataset['train_input'].shape}")
    print(f"[DATASET] train_label  : {dataset['train_label'].shape}")
    print(f"[DATASET] val_input    : {dataset['val_input'].shape}")
    print(f"[DATASET] val_label    : {dataset['val_label'].shape}")
    print(f"[DATASET] test_input   : {dataset['test_input'].shape}")
    print(f"[DATASET] test_label   : {dataset['test_label'].shape}")
    return dataset


# ===========================================================================
# SECTION 4 — MODEL TRAINING
# ===========================================================================

def build_model(seed: int = SEED) -> KAN:
    """
    Build single-layer KAN for KANDy Lorenz.

    width=[9, 3]:
      - Input  dim = 9  (dim of phi)
      - Output dim = 3  (dim of x_dot: x, y, z equations)
      - Single layer — NO hidden layers (KANDy requirement)

    Multi-layer KANs (e.g., [9, 16, 3]) are INCORRECT:
      they destroy the separability of Psi and prevent clean
      symbolic extraction of individual psi_i functions.
    """
    model = KAN(
        width = [PHI_DIM, STATE_DIM],   # [9, 3]  — single layer
        grid  = GRID,                   # 5 grid points per spline
        k     = K_SPLINE,               # cubic splines
        seed  = seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] KAN width=[{PHI_DIM}, {STATE_DIM}], "
          f"grid={GRID}, k={K_SPLINE}, params={n_params}")
    return model


def train_model(
        model:   KAN,
        dataset: Dict[str, torch.Tensor],
        steps:   int = STEPS,
) -> Tuple[KAN, List[float]]:
    """
    Train KAN with LBFGS optimizer.

    Returns
    -------
    model      : trained KAN
    loss_hist  : list of training loss values per step
    """
    print(f"\n[TRAIN] Starting LBFGS training for {steps} steps