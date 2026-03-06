"""kandy.tools — crewAI tools that give agents access to the KANDy library.

Each tool is a thin wrapper: it either executes a Python script in a
subprocess (capturing stdout/stderr) or performs file I/O.  This lets
LLM agents actually run experiments, not just write code.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]   # repo root (kandy/)


def _run_script(script_path: str, timeout: int = 300) -> str:
    """Execute a Python file; return combined stdout+stderr (truncated to 8 KB)."""
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(_ROOT),
    )
    out = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr.strip() else "")
    # Truncate to avoid flooding the LLM context
    if len(out) > 8000:
        out = out[:3500] + "\n...[truncated]...\n" + out[-3500:]
    return out or "(no output)"


# ---------------------------------------------------------------------------
# Tool: write a Python script to disk
# ---------------------------------------------------------------------------

class WriteScriptInput(BaseModel):
    path: str = Field(..., description="Relative path from repo root (e.g. 'outputs/my_exp.py').")
    code: str = Field(..., description="Complete Python source code to write.")


class WriteScriptTool(BaseTool):
    name: str = "write_python_script"
    description: str = (
        "Write a Python script to disk at the given path (relative to the repo root). "
        "Use this to save experiment code before running it."
    )
    args_schema: Type[BaseModel] = WriteScriptInput

    def _run(self, path: str, code: str) -> str:
        target = _ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(code))
        return f"Written {len(code)} chars to {target}"


# ---------------------------------------------------------------------------
# Tool: run a Python script
# ---------------------------------------------------------------------------

class RunScriptInput(BaseModel):
    path: str = Field(..., description="Relative path from repo root to the script to run.")
    timeout: int = Field(300, description="Maximum run time in seconds (default 300).")


class RunScriptTool(BaseTool):
    name: str = "run_python_script"
    description: str = (
        "Execute a Python script and return its stdout/stderr output (up to 8 KB). "
        "Use this to actually run an experiment and see training loss, RMSE, and "
        "symbolic formulas.  The script must already exist on disk."
    )
    args_schema: Type[BaseModel] = RunScriptInput

    def _run(self, path: str, timeout: int = 300) -> str:
        target = _ROOT / path
        if not target.exists():
            return f"ERROR: {target} does not exist.  Write it first with write_python_script."
        try:
            return _run_script(str(target), timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: script exceeded {timeout}s timeout."
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool: read a file
# ---------------------------------------------------------------------------

class ReadFileInput(BaseModel):
    path: str = Field(..., description="Relative path from repo root of the file to read.")
    max_chars: int = Field(6000, description="Maximum characters to return (default 6000).")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read a file and return its contents (up to max_chars characters). "
        "Use this to inspect existing experiment scripts, results, or source code."
    )
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, path: str, max_chars: int = 6000) -> str:
        target = _ROOT / path
        if not target.exists():
            return f"ERROR: {target} does not exist."
        content = target.read_text(errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n...[truncated at {max_chars} chars]..."
        return content


# ---------------------------------------------------------------------------
# Tool: KANDy API reference (static knowledge, no execution)
# ---------------------------------------------------------------------------

class KANDyAPIInput(BaseModel):
    topic: str = Field(
        "all",
        description=(
            "Topic to look up. One of: 'all', 'lifts', 'training', 'symbolic', "
            "'numerics', 'kanelift', 'plotting'."
        ),
    )


class KANDyAPITool(BaseTool):
    name: str = "kandy_api_reference"
    description: str = (
        "Look up the KANDy library API reference.  Returns function signatures, "
        "parameter descriptions, and code examples for the requested topic. "
        "Always consult this before writing KANDy experiment code."
    )
    args_schema: Type[BaseModel] = KANDyAPIInput

    _DOCS = {
        "lifts": """\
LIFTS  (from kandy import ...)
─────────────────────────────
PolynomialLift(degree, include_bias=True)
    Maps R^n → all monomials up to `degree`.  For Lorenz use degree=2 (gives xy, xz).

FourierLift(n_modes)
    Maps a periodic PDE snapshot u∈R^N → [DC, Re(û_1), Im(û_1), ..., Re(û_k), Im(û_k)].
    output_dim = 1 + 2*n_modes.

RadialBasisLift(n_centers, sigma=None, center_method='random')
    Gaussian RBF dictionary.  Requires lift.fit(X) first.
    output_dim = n_centers.

DMDLift(n_modes, dictionary=None, sort_by='magnitude')
    EDMD Koopman eigenfunctions.  Requires lift.fit(X_trajectory) first.

DelayEmbedding(delays)
    Takens delay embedding for scalar time series.

CustomLift(fn, output_dim, name='custom')
    Wrap any callable: fn(X: ndarray (N,n)) → ndarray (N,m).

KANELift(latent_dim, hidden_dim=None, grid=5, k=3)  [EXPERIMENTAL]
    KAN autoencoder lift.  Train with lift.train_koopman(traj, dt, epochs=50, lr=1e-3).
    After training, lift(X) applies encoder only.  lift.get_formula() extracts symbolic
    expressions for each Koopman observable.
    Architecture: default single-layer [n, latent_dim]; set hidden_dim for [n, h, latent_dim].
""",
        "training": """\
TRAINING  (from kandy import KANDy, fit_kan, make_windows, angle_mse)
──────────────────────────────────────────────────────────────────────
KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None, base_fun=None)
    .fit(X, X_dot=None, *, dt=None, opt='LBFGS', lr=1.0, batch=-1,
          lamb=0.0, rollout_weight=0.0, rollout_loss_fn=None, val_frac=0.15, test_frac=0.15)
    .predict(X)              → X_dot_pred
    .rollout(x0, T, dt, integrator='rk4')  → trajectory (T, n)
    .get_formula(var_names, round_places=3, simplify=False)  → list[sympy.Expr]
    .score_formula(formulas, X, y_true, var_names)           → list[float]  (R²)
    .get_A()                 → ndarray (n, m)

For discrete maps: pass X=current_state, X_dot=next_state.
For periodic phases: rollout_loss_fn=angle_mse.
For large datasets:  opt='Adam', lr=2e-3, batch=4096.

fit_kan(model_, dataset, opt, steps, lr, batch, rollout_weight, rollout_horizon,
        dynamics_fn, integrator, rollout_loss_fn, update_grid, stop_grid_update_step)
    Low-level trainer.  dataset needs train_input/train_label/test_input/test_label.
    For rollout loss also: train_traj (Nw,T,n), train_t (T,), test_traj, test_t.

make_windows(traj, window)  → (T-w+1, w, n) overlapping windows for rollout training.
angle_mse(pred, true)       → MSE with wrapped angle differences (use for Kuramoto).
wrap_pi_torch(x)            → wrap angles to (-π, π].
""",
        "symbolic": """\
SYMBOLIC  (from kandy import auto_symbolic_with_costs, score_formula, ...)
───────────────────────────────────────────────────────────────────────────
auto_symbolic_with_costs(model_, preferred_idx, preferred_lib, other_lib,
                          weight_simple=0.8, r2_threshold=0.90, verbose=1)
    Must run a forward pass with model_.save_act=True first.
    preferred_idx: set of input feature indices to treat as "physics-informed" (cheap cost).
    Libraries: POLY_LIB_CHEAP, POLY_LIB, TRIG_LIB_CHEAP, TRIG_LIB.

score_formula(formulas, theta, y_true, var_names)  → list[float]  (R² per output)
formulas_to_latex(formulas, lhs_names)             → LaTeX align* string
substitute_params(formulas, {'sigma': 10.0, ...})  → simplified Expr list
make_symbolic_lib({'name': (torch_fn, sympy_fn, cost), ...})
""",
        "kanelift": """\
KANELift — EXPERIMENTAL KAN Autoencoder Koopman Lift
─────────────────────────────────────────────────────
Architecture:
  encoder KAN  :  x_t   ──[n, latent_dim]──►  z_t
  propagator K :  z_t   ──Linear(no bias)──►  z_{t+1}
  decoder KAN  :  z_{t+1} ──[latent_dim, n]──►  x_{t+1}

Training loss = gamma_predx * MSE(x_{t+1}_pred, x_{t+1})
              + alpha_latent * MSE(K·enc(x_t), enc(x_{t+1}))
              + beta_recon   * MSE(dec(enc(x_t)), x_t)

Usage:
  from kandy import KANELift, KANDy
  lift = KANELift(latent_dim=8)          # single-layer encoder by default
  lift.train_koopman(
      traj,                              # (T, n) or (B, T, n)
      dt=0.01,
      epochs=100, lr=5e-4, batch_size=512,
      alpha_latent=1.0, beta_recon=1.0, gamma_predx=1.0,
  )
  # Now use as a drop-in lift for KANDy:
  model = KANDy(lift=lift, grid=5, k=3)
  model.fit(X_state, X_next)

  # Symbolic encoder formulas:
  # First populate activations with a forward pass:
  import torch
  lift.encoder_.save_act = True
  with torch.no_grad():
      lift.encoder_(torch.tensor(X_state[:512].astype('float32')))
  formulas = lift.get_formula(var_names=['x', 'y', 'z'])

Research questions for KANE lift:
  1. Does it converge?  Monitor train_history_['val_total'].
  2. What latent_dim is sufficient for Lorenz?  (Try 6, 8, 10.)
  3. After symbolic extraction, are the encoder formulas interpretable?
  4. Can rollout with KANDy(lift=KANELift, ...) match PolynomialLift accuracy?
  5. Can the K_ matrix eigenvalues reveal the Koopman spectrum?
""",
        "numerics": """\
NUMERICS  (from kandy import solve_burgers, solve_viscous_burgers, cfl_dt, ...)
────────────────────────────────────────────────────────────────────────────────
cfl_dt(u, dx, cfl=0.8)  → stable dt from CFL condition
solve_burgers(u0, n_steps, dt, domain_length=None, scheme='rusanov',
              limiter='minmod', time_stepper='tvdrk2', save_every=1)
solve_viscous_burgers(u0, n_steps, dt, nu, ...)  → IMEX: explicit conv + spectral diff
solve_scalar(u0, dx, n_steps, dt, flux_fn, speed_fn, ...)
fv_rhs(u, dx, flux_fn, speed_fn, scheme, limiter)
Schemes: 'rusanov', 'roe', 'hllc'.  Limiters: 'minmod', 'van_leer', 'superbee'.
""",
        "plotting": """\
PLOTTING  (from kandy.plotting import ...)
──────────────────────────────────────────
plot_attractor_overlay(true_traj, pred_traj, dim_x=0, dim_y=1,
                       labels, colors, title, save)
plot_loss_curves(results_dict, title, save)
plot_all_edges(model_, X, input_names, output_names, title, save)
plot_trajectory_error(true_traj, pred_traj, dt, title, save)
use_pub_style()   # publication-quality matplotlib defaults
All save functions write both .png (300 dpi) and .pdf.
""",
    }

    def _run(self, topic: str = "all") -> str:
        if topic == "all":
            return "\n\n".join(self._DOCS.values())
        key = topic.lower().strip()
        if key in self._DOCS:
            return self._DOCS[key]
        return (
            f"Unknown topic '{topic}'.  "
            f"Choose from: {', '.join(self._DOCS)}, or 'all'."
        )
