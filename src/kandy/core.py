"""kandy.core — KANDy: KAN-based dynamical system identification.

Algorithm
---------
    x_dot = A * Psi(phi(x))

where:
    phi  : Koopman lift  R^n -> R^m  (user-designed; encodes ALL cross-terms)
    Psi  : separable spline map — a SINGLE-LAYER KAN with width=[m, n]
    A    : linear mixing matrix extracted from KAN output weights

Critical constraint: the KAN is ALWAYS single-layer (width=[m, n]).  Deep
KANs (width=[m, h, n]) cannot represent bilinear terms like x*y from raw
inputs; cross-terms must be pre-encoded in phi.  Depth is therefore never
exposed as a user parameter.

Typical usage
-------------
>>> from kandy import KANDy, PolynomialLift
>>> lift = PolynomialLift(degree=2)          # phi: R^3 -> R^9 for Lorenz
>>> model = KANDy(lift=lift, grid=5, k=3)
>>> model.fit(X_traj, X_dot)
>>> formulas = model.get_formula()           # list of SymPy expressions
>>> X_pred   = model.rollout(x0, T=1000, dt=0.005)
"""
from __future__ import annotations

import random
import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch
from kan import KAN
import sympy as sp

from .lifts import Lift
from .training import fit_kan
from .symbolic import score_formula as _score_formula


class KANDy:
    """KAN-based dynamical system identification.

    Parameters
    ----------
    lift : Lift or callable
        Koopman lift phi: R^n -> R^m.  Must encode ALL cross-interaction
        terms present in the target system's RHS (e.g., x*y and x*z for
        Lorenz).  Use ``PolynomialLift(degree=2)`` for polynomial systems
        or ``CustomLift(fn, output_dim=m)`` for hand-crafted lifts.
    grid : int
        Number of grid points per spline segment (default 5).
    k : int
        Spline degree — 3 gives cubic splines (default 3).
    steps : int
        Number of LBFGS optimisation steps (default 500).
    seed : int
        Random seed for full reproducibility (default 42).
    device : str or None
        ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect, default).
    base_fun : callable or None
        Base activation for each KAN edge (default: SiLU from PyKAN).
        Use ``lambda x: torch.exp(-x**2)`` for RBF-style activations.

    Attributes
    ----------
    model_ : KAN
        Fitted PyKAN model (available after ``fit``).
    lift_dim_ : int
        Dimension m of lifted space (available after ``fit``).
    state_dim_ : int
        Dimension n of state space / derivative space (available after ``fit``).
    """

    def __init__(
        self,
        lift: Union[Lift, Callable],
        grid: int = 5,
        k: int = 3,
        steps: int = 500,
        seed: int = 42,
        device: Optional[str] = None,
        base_fun=None,
    ):
        self.lift = lift
        self.grid = grid
        self.k = k
        self.steps = steps
        self.seed = seed
        self.base_fun = base_fun

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_: Optional[KAN] = None
        self.lift_dim_: Optional[int] = None
        self.state_dim_: Optional[int] = None
        self._train_input: Optional[torch.Tensor] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        X_dot: Optional[np.ndarray] = None,
        *,
        dt: Optional[float] = None,
        t: Optional[np.ndarray] = None,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        lamb: float = 0.0,
        rollout_weight: float = 0.0,
        rollout_horizon: Optional[int] = None,
        rollout_loss_fn: Optional[Callable] = None,
        opt: str = "LBFGS",
        lr: float = 1.0,
        batch: int = -1,
        stop_grid_update_step: int = 50,
        fit_steps: Optional[int] = None,
        patience: int = 10,
        verbose: bool = True,
    ) -> "KANDy":
        """Fit the KANDy model.

        Parameters
        ----------
        X : np.ndarray, shape (N, n)
            State trajectory.  Rows are time snapshots.
        X_dot : np.ndarray, shape (N, n), optional
            Time derivatives.  If omitted ``dt`` must be given and forward
            differences are used.
        dt : float, optional
            Uniform time step — used for forward-difference derivative
            estimation (when X_dot is omitted) and for building trajectory
            data for rollout loss.
        t : np.ndarray, shape (N,), optional
            Time values for each snapshot.  If not provided but ``dt`` is,
            a uniform grid is constructed automatically.
        val_frac, test_frac : float
            Chronological train / val / test split fractions.
        lamb : float
            Sparsity regularisation weight (L1 + entropy, default 0).
        rollout_weight : float
            Weight on the integrated trajectory loss (default 0 = derivative
            supervision only).
        rollout_horizon : int, optional
            Number of integration steps per trajectory segment for the rollout
            loss.  None = use full trajectory length minus 1.
        opt : str
            Optimiser: ``'LBFGS'`` (default) or ``'Adam'``.
            Use ``'Adam'`` for large datasets or when training discrete maps
            with many parameters (e.g. Holling, Ikeda).
        lr : float
            Learning rate (default 1.0 for LBFGS; use ~1e-3 for Adam).
        batch : int
            Mini-batch size (-1 = full batch, default).
        stop_grid_update_step : int
            Step at which KAN grid updates stop (default 50).
        rollout_loss_fn : callable, optional
            Loss function used for the rollout loss only.  If ``None``
            (default), falls back to MSE — the same as the derivative loss.
            Pass ``kandy.angle_mse`` for systems with periodic phase variables
            (e.g., Kuramoto), or any ``(pred, true) -> scalar`` callable.
        fit_steps : int, optional
            Override ``self.steps`` for this call.
        patience : int
            Early stopping patience (default 10). Set to 0 to disable.
        verbose : bool
            Print training progress.

        Returns
        -------
        self
        """
        # PyKAN's .to(device) does not reliably move internal spline grids
        # to CUDA, causing device-mismatch errors in B_batch. Force CPU.
        self.device = torch.device("cpu")

        self._set_seeds()
        steps = fit_steps if fit_steps is not None else self.steps

        # Derivative estimation via forward differences if X_dot not provided
        if X_dot is None:
            if dt is None:
                raise ValueError("Provide X_dot or dt (for forward-difference estimation).")
            X_dot = (X[1:] - X[:-1]) / dt
            X = X[:-1]

        X = np.asarray(X, dtype=np.float64)
        X_dot = np.asarray(X_dot, dtype=np.float64)

        if X.shape[0] != X_dot.shape[0]:
            raise ValueError(
                f"X and X_dot must have the same number of rows; "
                f"got {X.shape[0]} and {X_dot.shape[0]}."
            )

        # Apply Koopman lift
        if hasattr(self.lift, "fit"):
            self.lift.fit(X)
        theta = self.lift(X)                        # (N, m)
        if theta.ndim == 1:
            theta = theta[:, None]

        m = theta.shape[1]                          # lift dimension
        n = X_dot.shape[1] if X_dot.ndim > 1 else 1  # state / derivative dim

        if verbose:
            print(f"[KANDy] Lift: R^{X.shape[1]} -> R^{m}  |  KAN width=[{m}, {n}]")

        # Chronological train / val / test split (no shuffling)
        N = len(theta)
        n1 = int(N * (1 - val_frac - test_frac))
        n2 = int(N * (1 - test_frac))

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        dataset = {
            "train_input": _t(theta[:n1]),
            "train_label": _t(X_dot[:n1]),
            "val_input":   _t(theta[n1:n2]),
            "val_label":   _t(X_dot[n1:n2]),
            "test_input":  _t(theta[n2:]),
            "test_label":  _t(X_dot[n2:]),
        }

        # Pack trajectory data for rollout loss
        if rollout_weight > 0.0:
            if t is None and dt is not None:
                t = np.arange(N) * dt
            if t is not None:
                train_traj = torch.tensor(
                    X[:n1][None, :, :], dtype=torch.float32, device=self.device
                )
                train_t = torch.tensor(
                    t[:n1], dtype=torch.float32, device=self.device
                )
                dataset["train_traj"] = train_traj
                dataset["train_t"] = train_t

        if verbose:
            print(
                f"[KANDy] Split — train: {n1}, val: {n2-n1}, test: {N-n2}  "
                f"(total {N} points)"
            )

        # Build single-layer KAN  (depth fixed at 1 — never expose as param)
        kan_kwargs: dict = dict(
            width=[m, n],
            grid=self.grid,
            k=self.k,
            seed=self.seed,
        )
        if self.base_fun is not None:
            kan_kwargs["base_fun"] = self.base_fun

        self.model_ = KAN(**kan_kwargs)

        # Build dynamics_fn that applies the lift before the KAN
        # (trajectories are in raw state space but the KAN expects lifted features)
        _lift = self.lift
        _model = self.model_

        def _dynamics_fn(state: torch.Tensor) -> torch.Tensor:
            # Use torch_fn if available (preserves gradient graph)
            if hasattr(_lift, "torch_fn") and _lift.torch_fn is not None:
                lifted_t = _lift.torch_fn(state)
            else:
                state_np = state.detach().cpu().numpy()
                lifted = _lift(state_np)
                lifted_t = torch.tensor(
                    lifted, dtype=torch.float32, device=state.device
                )
            return _model(lifted_t)

        self.train_results_ = fit_kan(
            self.model_,
            dataset,
            opt=opt,
            lr=lr,
            batch=batch,
            steps=steps,
            lamb=lamb,
            rollout_weight=rollout_weight,
            rollout_horizon=rollout_horizon,
            rollout_loss_fn=rollout_loss_fn,
            dynamics_fn=_dynamics_fn if rollout_weight > 0.0 else None,
            stop_grid_update_step=stop_grid_update_step,
            patience=patience,
        )

        self._train_input = dataset["train_input"]
        self.lift_dim_ = m
        self.state_dim_ = n
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def get_A(self) -> np.ndarray:
        """Return the linear mixing matrix A of shape (n, m).

        In the KANDy model x_dot = A * Psi(phi(x)), A is the per-edge
        scale factor stored in the KAN's spline layer.  Each entry A[i, j]
        weights spline psi_j applied to lifted feature theta_j for output i.

        Returns
        -------
        A : np.ndarray, shape (n, m)
        """
        self._check_fitted()
        with torch.no_grad():
            # PyKAN stores per-edge spline scales in act_fun[0].scale_sp
            # shape: (n_out * n_in,) → reshape to (n_out, n_in) = (n, m)
            try:
                scale = self.model_.act_fun[0].scale_sp
                A = scale.detach().cpu().numpy().reshape(self.state_dim_, self.lift_dim_)
            except AttributeError:
                warnings.warn(
                    "Could not extract A from model.act_fun[0].scale_sp. "
                    "PyKAN API may have changed.  Returning None.",
                    stacklevel=2,
                )
                A = None
        return A

    def get_formula(
        self,
        var_names: Optional[list[str]] = None,
        round_places: int = 3,
        simplify: bool = False,
        lib: Optional[list] = None,
    ) -> list:
        """Extract symbolic formulas via PyKAN's auto_symbolic.

        Calls ``model.auto_symbolic()`` then ``model.symbolic_formula()``
        and returns one SymPy expression per output dimension.

        Parameters
        ----------
        var_names : list of str, optional
            Names to substitute for PyKAN's internal variable names.
            Defaults to the lift's ``feature_names`` if available, else
            ``['theta_0', 'theta_1', ...]``.
        round_places : int
            Decimal places for rounding numeric constants (default 3).
        simplify : bool
            If True, attempt SymPy simplification pipeline after extraction:
            ``sp.factor`` → ``sp.together`` → ``sp.nsimplify``.  Can be slow
            for large expressions (default False).
        lib : list, optional
            Symbolic function library for ``auto_symbolic``.  Each entry is
            a string name (e.g. ``'x^2'``) or a tuple
            ``(name, torch_fn, sympy_fn, complexity)``.
            If None, uses PyKAN's default library.

        Returns
        -------
        formulas : list of sympy.Expr
            One expression per output dimension, e.g.
            ``[x_dot_expr, y_dot_expr, z_dot_expr]`` for a 3D system.
        """
        self._check_fitted()

        # Populate activations with a forward pass on training data
        self.model_.save_act = True
        with torch.no_grad():
            self.model_(self._train_input)

        auto_sym_kwargs = {}
        if lib is not None:
            auto_sym_kwargs["lib"] = lib
        self.model_.auto_symbolic(**auto_sym_kwargs)
        exprs, inputs = self.model_.symbolic_formula()

        # Build variable substitution map
        if var_names is None:
            if hasattr(self.lift, "feature_names"):
                try:
                    var_names = self.lift.feature_names
                except Exception:
                    var_names = [f"theta_{i}" for i in range(self.lift_dim_)]
            else:
                var_names = [f"theta_{i}" for i in range(self.lift_dim_)]

        sub_map = {
            sp.Symbol(str(inp)): sp.Symbol(name)
            for inp, name in zip(inputs, var_names)
        }

        formulas = []
        for expr_str in exprs:
            sym = sp.sympify(expr_str)
            sym = sym.xreplace(sub_map)
            sym = _round_sympy(sym, round_places)
            sym = sp.expand(sym)
            if simplify:
                sym = _simplify_pipeline(sym)
            formulas.append(sym)

        return formulas

    def score_formula(
        self,
        formulas: list,
        X: np.ndarray,
        y_true: np.ndarray,
        var_names: Optional[list[str]] = None,
    ) -> list[float]:
        """Evaluate symbolic formula accuracy as R² on held-out data.

        Parameters
        ----------
        formulas : list of sympy.Expr
            Symbolic expressions from :meth:`get_formula`.
        X : np.ndarray, shape (N, n)
            State array (will be lifted and, if needed, normalised externally).
        y_true : np.ndarray, shape (N,) or (N, n_out)
            Ground-truth targets.
        var_names : list of str, optional
            Feature names matching the symbols in ``formulas``.
            Defaults to ``['theta_0', 'theta_1', ...]``.

        Returns
        -------
        r2_scores : list of float
            R² per output dimension.
        """
        self._check_fitted()
        theta = self.lift(np.asarray(X, dtype=np.float64))
        if theta.ndim == 1:
            theta = theta[:, None]

        if var_names is None:
            if hasattr(self.lift, "feature_names"):
                try:
                    var_names = self.lift.feature_names
                except Exception:
                    var_names = [f"theta_{i}" for i in range(self.lift_dim_)]
            else:
                var_names = [f"theta_{i}" for i in range(self.lift_dim_)]

        return _score_formula(formulas, theta, y_true, var_names)

    # ------------------------------------------------------------------
    # Prediction / rollout
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict derivatives x_dot = A*Psi(phi(X)) for state array X.

        Parameters
        ----------
        X : np.ndarray, shape (N, n) or (n,)

        Returns
        -------
        X_dot_pred : np.ndarray, shape (N, n) or (n,)
        """
        self._check_fitted()
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        theta = self.lift(X)
        if theta.ndim == 1:
            theta = theta[None, :]
        t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            xdot = self.model_(t).cpu().numpy()
        return xdot.squeeze() if scalar else xdot

    def rollout(
        self,
        x0: np.ndarray,
        T: int,
        dt: float,
        integrator: str = "rk4",
    ) -> np.ndarray:
        """Autoregressive trajectory integration using the learned model.

        Integrates dx/dt = model(phi(x)) starting from x0 for T steps.

        Parameters
        ----------
        x0 : np.ndarray, shape (n,)
            Initial state.
        T : int
            Number of time steps (output has shape (T, n)).
        dt : float
            Time step.
        integrator : str
            ``'rk4'`` (default) or ``'euler'``.

        Returns
        -------
        traj : np.ndarray, shape (T, n)
        """
        self._check_fitted()

        def dynamics(x: np.ndarray) -> np.ndarray:
            return self.predict(x)

        traj = [x0.copy()]
        x = x0.copy().astype(np.float64)

        for _ in range(T - 1):
            if integrator == "rk4":
                k1 = dynamics(x)
                k2 = dynamics(x + 0.5 * dt * k1)
                k3 = dynamics(x + 0.5 * dt * k2)
                k4 = dynamics(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            elif integrator == "euler":
                x = x + dt * dynamics(x)
            else:
                raise ValueError(f"Unknown integrator {integrator!r}. Use 'rk4' or 'euler'.")
            traj.append(x.copy())

        return np.array(traj)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "KANDy model is not fitted.  Call .fit(X, X_dot) first."
            )

    @staticmethod
    def _central_diff(X: np.ndarray, dt: float) -> np.ndarray:
        """Central-difference derivative estimate.  Returns shape (N-2, n)."""
        return (X[2:] - X[:-2]) / (2.0 * dt)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"KANDy(lift={self.lift!r}, grid={self.grid}, k={self.k}, "
            f"steps={self.steps}, status={status})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _round_sympy(expr, places: int = 3):
    """Round all numeric atoms in a SymPy expression to *places* decimals."""
    return expr.xreplace(
        {a: round(float(a), places) for a in expr.atoms(sp.Number)}
    )


def _simplify_pipeline(expr) -> sp.Expr:
    """Apply a cascade of SymPy simplifiers, returning the shortest result."""
    candidates = [expr]
    for fn in (sp.factor, sp.together, lambda e: sp.nsimplify(e, rational=False, tolerance=1e-3)):
        try:
            candidates.append(fn(expr))
        except Exception:
            pass
    return min(candidates, key=lambda e: len(str(e)))
