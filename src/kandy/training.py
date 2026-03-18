"""kandy.training — KANDy training loop with integrated rollout loss.

This module provides ``fit_kan``, a drop-in replacement for PyKAN's built-in
``model.fit()`` that adds a differentiable **integrated trajectory loss**:

    L = L_deriv + rollout_weight * L_rollout + lamb * L_reg

where:
    L_deriv   — MSE between model(phi(x)) and x_dot  (standard supervision)
    L_rollout — MSE between numerically-integrated trajectory and true traj
    L_reg     — PyKAN sparsity regularisation (L1 + entropy)

The rollout loss forces the model to produce trajectories that are globally
consistent with the true dynamics, not just locally accurate at each snapshot.
This is critical for chaotic systems (Lorenz, KS) where small derivative
errors compound exponentially during rollout.

Dataset keys
------------
Required:
    train_input  (N_train, m)   lifted states theta = phi(x)
    train_label  (N_train, n)   derivatives x_dot
    test_input   (N_test,  m)
    test_label   (N_test,  n)

Optional (needed for rollout loss):
    train_traj   (B_train, T, n)  raw state trajectories
    train_t      (T,)             time values for train trajectories
    test_traj    (B_test, T, n)
    test_t       (T,)

val_input / val_label are accepted but treated as test for logging.

Usage
-----
>>> from kandy.training import fit_kan
>>> results = fit_kan(
...     model,
...     dataset,
...     opt="LBFGS",
...     steps=300,
...     rollout_weight=1.0,
...     rollout_horizon=10,
...     dynamics_fn=my_phi_wrapped_fn,   # maps state -> derivative
...     integrator="rk4",
... )
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    import kan.LBFGS as _lbfgs_module
    _LBFGS = _lbfgs_module.LBFGS
except (ImportError, AttributeError):
    try:
        from torch.optim import LBFGS as _LBFGS  # fallback to PyTorch built-in
    except ImportError:
        _LBFGS = None


# ---------------------------------------------------------------------------
# Angle-aware loss utilities
# ---------------------------------------------------------------------------

def wrap_pi_torch(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (−π, π].

    Used as a pre-processing step in the rollout loss for systems with
    periodic phase variables (e.g., Kuramoto oscillators).  Applying this
    before squaring ensures that a phase error of 2π − ε is treated the same
    as ε rather than as a large error.

    Parameters
    ----------
    x : torch.Tensor
        Angle differences, any shape.

    Returns
    -------
    torch.Tensor
        Wrapped differences in (−π, π].
    """
    return (x + torch.pi) % (2.0 * torch.pi) - torch.pi


def angle_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """MSE loss with phase-difference wrapping.

    Equivalent to ``mean((wrap_pi(pred − true))²)``.  Pass as
    ``rollout_loss_fn`` for systems whose state contains angular variables.
    """
    return torch.mean(wrap_pi_torch(pred - true) ** 2)


def order_param_torch(theta: torch.Tensor) -> torch.Tensor:
    """Kuramoto order parameter  r = |⟨e^{iθ}⟩|.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, T, N)
        Phase trajectories for B trajectories, T timesteps, N oscillators.

    Returns
    -------
    r : torch.Tensor, shape (B, T)
        Synchronisation order parameter (real, ∈ [0, 1]).
    """
    z = torch.exp(1j * theta)
    r = torch.abs(torch.mean(z, dim=2))
    return r.real


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def euler_step(
    f: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Single Euler step: s_{t+1} = s_t + dt * f(s_t).

    Parameters
    ----------
    f  : callable (B, n) -> (B, n)  dynamics function
    s  : (B, n) current state
    dt : (B, 1) or scalar time step
    """
    return s + dt * f(s)


def rk4_step(
    f: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Single RK4 step: 4th-order Runge-Kutta.

    Parameters
    ----------
    f  : callable (B, n) -> (B, n)  dynamics function
    s  : (B, n) current state
    dt : (B, 1) or scalar time step
    """
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(
    f: Callable[[torch.Tensor], torch.Tensor],
    s0: torch.Tensor,
    t: torch.Tensor,
    horizon: Optional[int] = None,
    integrator: Literal["rk4", "euler"] = "rk4",
) -> torch.Tensor:
    """Integrate a batch of initial conditions forward in time.

    Parameters
    ----------
    f         : dynamics callable  (B, n) -> (B, n)
    s0        : (B, n) initial states
    t         : (T,) or (B, T) time values
    horizon   : number of steps to integrate (default: T-1)
    integrator: 'rk4' (default) or 'euler'

    Returns
    -------
    pred_traj : (B, H+1, n)  predicted trajectory including s0
    """
    if t.dim() == 1:
        t = t.unsqueeze(0).expand(s0.shape[0], -1)   # (B, T)
    B, T = t.shape
    H = (T - 1) if horizon is None else min(horizon, T - 1)

    step_fn = rk4_step if integrator == "rk4" else euler_step
    if integrator not in ("rk4", "euler"):
        raise ValueError(f"integrator must be 'rk4' or 'euler', got {integrator!r}")

    state = s0
    out = [state]
    for i in range(H):
        dt = (t[:, i + 1] - t[:, i]).unsqueeze(1)   # (B, 1)
        state = step_fn(f, state, dt)
        out.append(state)

    return torch.stack(out, dim=1)   # (B, H+1, n)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def fit_kan(
    model: nn.Module,
    dataset: Dict[str, torch.Tensor],
    *,
    # ---- optimiser ----
    opt: Literal["LBFGS", "Adam"] = "LBFGS",
    steps: int = 100,
    lr: float = 1.0,
    batch: int = -1,
    # ---- derivative supervision ----
    loss_fn: Optional[Callable] = None,
    # ---- rollout loss (may differ from derivative loss) ----
    rollout_loss_fn: Optional[Callable] = None,
    # ---- PyKAN regularisation ----
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    reg_metric: str = "edge_forward_spline_n",
    # ---- grid update (PyKAN) ----
    update_grid: bool = True,
    grid_update_num: int = 10,
    start_grid_update_step: int = -1,
    stop_grid_update_step: int = 50,
    # ---- rollout / integration loss ----
    rollout_weight: float = 0.0,
    rollout_horizon: Optional[int] = None,
    traj_batch: int = -1,
    dynamics_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    integrator: Literal["rk4", "euler"] = "rk4",
    # ---- early stopping ----
    patience: int = 10,
    # ---- misc ----
    singularity_avoiding: bool = False,
    y_th: float = 1000.0,
    log: int = 1,
    metrics: Optional[Sequence[Callable]] = None,
) -> Dict[str, List[float]]:
    """Train a PyKAN model with optional differentiable integrated rollout loss.

    This extends PyKAN's built-in ``model.fit()`` with a second loss term that
    integrates the learned dynamics forward in time and penalises deviation from
    true state trajectories.  This is the core training procedure for KANDy.

    Parameters
    ----------
    model : KAN
        A PyKAN model, typically ``KAN(width=[m, n], grid=g, k=k)``.
        Must already be on the correct device.
    dataset : dict
        Must contain ``train_input``, ``train_label``, ``test_input``,
        ``test_label`` as ``torch.FloatTensor``.
        For rollout loss, also include ``train_traj`` (B, T, n),
        ``train_t`` (T,), and optionally ``test_traj``, ``test_t``.
    opt : str
        Optimiser: ``'LBFGS'`` (default, recommended) or ``'Adam'``.
    steps : int
        Number of optimisation steps.
    lr : float
        Learning rate (default 1.0 for LBFGS; use ~1e-3 for Adam).
    batch : int
        Mini-batch size for derivative supervision (-1 = full batch).
    loss_fn : callable, optional
        Loss function ``(pred, target) -> scalar`` for derivative supervision.
        Default: MSE.
    rollout_loss_fn : callable, optional
        Loss function used exclusively for the integrated rollout loss.
        If ``None`` (default), falls back to ``loss_fn``.
        Use ``kandy.training.angle_mse`` for systems with periodic phase
        variables (e.g., Kuramoto oscillators), or supply any callable of the
        form ``(pred_traj, true_traj) -> scalar``.
    lamb : float
        Regularisation weight.  Requires ``model.save_act = True``.
    lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff : float
        PyKAN regularisation sub-weights.
    reg_metric : str
        PyKAN regularisation metric (default ``'edge_forward_spline_n'``).
    update_grid : bool
        Whether to update KAN grid during training (default True).
    grid_update_num : int
        Number of grid updates within ``[start_grid_update_step,
        stop_grid_update_step]``.
    start_grid_update_step : int
        First step at which grid updates begin (-1 = from step 0).
    stop_grid_update_step : int
        Step at which grid updates stop (default 50).
    rollout_weight : float
        Weight on the integrated trajectory loss (default 0 = off).
        Set to a positive value (e.g., 1.0) to enable.
    rollout_horizon : int, optional
        Number of integration steps per trajectory segment.
        None = use full trajectory length minus 1.
    traj_batch : int
        Trajectory mini-batch size for rollout loss (-1 = all trajectories).
    dynamics_fn : callable, optional
        Function ``f(state: Tensor[B, n]) -> Tensor[B, n]`` returning
        derivatives.  Defaults to ``model(state)`` with the lift applied
        inside the KAN.  **Supply this when the model takes lifted features**
        (theta = phi(x)) but the trajectories are in raw state space — the
        callable should apply phi internally before forwarding through the KAN.
    integrator : str
        ``'rk4'`` (default, 4th-order) or ``'euler'`` (1st-order).
    singularity_avoiding : bool
        Passed to PyKAN's forward to avoid numerical singularities.
    y_th : float
        Threshold for singularity avoidance.
    log : int
        Print progress every ``log`` steps (default 1).
    metrics : sequence of callables, optional
        Extra metrics to log; each called with no args after each step.

    Returns
    -------
    results : dict with keys
        ``train_loss``, ``test_loss``, ``reg``,
        ``rollout_train_loss``, ``rollout_test_loss``
        — each a list of floats, one entry per step.

    Notes
    -----
    - The combined objective is:
          L = MSE(model(theta), xdot) + rollout_weight * MSE(traj_pred, traj_true) + lamb * L_reg
    - Rollout loss requires raw-state trajectories in the dataset;
      derivative supervision only requires snapshot pairs (theta, xdot).
    - Grid updates happen only during the window
      ``[start_grid_update_step, stop_grid_update_step)`` at a frequency
      determined by ``stop_grid_update_step / grid_update_num``.
    """
    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if loss_fn is None:
        loss_fn = lambda x, y: torch.mean((x - y) ** 2)

    _rollout_loss_fn = rollout_loss_fn if rollout_loss_fn is not None else loss_fn

    # ------------------------------------------------------------------
    # Dynamics function for rollout
    # ------------------------------------------------------------------
    if dynamics_fn is None:
        def dynamics_fn(state: torch.Tensor) -> torch.Tensor:
            return model.forward(state,
                                 singularity_avoiding=singularity_avoiding,
                                 y_th=y_th)

    # ------------------------------------------------------------------
    # k-step forward (derivative supervision; k=1 standard)
    # ------------------------------------------------------------------
    def _deriv_forward(inputs: torch.Tensor) -> torch.Tensor:
        return model.forward(inputs,
                             singularity_avoiding=singularity_avoiding,
                             y_th=y_th)

    # ------------------------------------------------------------------
    # PyKAN symbolic / regularisation setup
    # ------------------------------------------------------------------
    if lamb > 0.0 and not model.save_act:
        model.save_act = True

    old_save_act, old_symbolic_enabled = model.disable_symbolic_in_fit(lamb)

    # ------------------------------------------------------------------
    # Grid update frequency
    # ------------------------------------------------------------------
    grid_update_freq = max(1, int(stop_grid_update_step / grid_update_num))

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.get_params(), lr=lr)
    elif opt == "LBFGS":
        if _LBFGS is None:
            raise ImportError(
                "LBFGS not found.  Either PyKAN or PyTorch LBFGS must be available."
            )
        optimizer = _LBFGS(
            model.get_params(),
            lr=lr,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32,
            tolerance_change=1e-32,
        )
    else:
        raise ValueError(f"opt must be 'LBFGS' or 'Adam', got {opt!r}")

    # ------------------------------------------------------------------
    # Dataset bookkeeping
    # ------------------------------------------------------------------
    n_train = dataset["train_input"].shape[0]
    n_test  = dataset["test_input"].shape[0]
    batch_sz      = n_train if (batch == -1 or batch > n_train) else batch
    batch_sz_test = n_test  if (batch == -1 or batch > n_test)  else batch

    has_train_traj = ("train_traj" in dataset) and ("train_t" in dataset)
    has_test_traj  = ("test_traj"  in dataset) and ("test_t"  in dataset)

    if has_train_traj:
        n_traj_train = dataset["train_traj"].shape[0]
        tbatch = n_traj_train if (traj_batch == -1 or traj_batch > n_traj_train) else traj_batch
    else:
        n_traj_train = tbatch = 0

    if has_test_traj:
        n_traj_test = dataset["test_traj"].shape[0]
        tbatch_test = n_traj_test if (traj_batch == -1 or traj_batch > n_traj_test) else traj_batch
    else:
        n_traj_test = tbatch_test = 0

    # ------------------------------------------------------------------
    # Early stopping state
    # ------------------------------------------------------------------
    _best_loss       = float("inf")
    _no_improve      = 0
    _best_state_dict = None   # best model checkpoint for restoration

    # ------------------------------------------------------------------
    # Results accumulator
    # ------------------------------------------------------------------
    results: Dict[str, List] = {
        "train_loss": [],
        "test_loss": [],
        "reg": [],
        "rollout_train_loss": [],
        "rollout_test_loss": [],
    }
    if metrics is not None:
        for m in metrics:
            results[m.__name__] = []

    # ------------------------------------------------------------------
    # Rolling state shared across closure calls
    # ------------------------------------------------------------------
    # We use a mutable container so the closure can update them.
    _state = {
        "train_loss":         torch.tensor(0.0),
        "reg":                torch.tensor(0.0),
        "rollout_train_loss": torch.tensor(0.0),
    }

    # ------------------------------------------------------------------
    # Rollout loss helper
    # ------------------------------------------------------------------
    def _rollout_loss_on_batch(
        traj: torch.Tensor,
        t:    torch.Tensor,
    ) -> torch.Tensor:
        """
        traj : (B, T, n)
        t    : (T,) or (B, T)
        Returns scalar loss between predicted and true trajectories,
        using rollout_loss_fn (which may differ from the derivative loss_fn).
        """
        s0   = traj[:, 0, :]                                              # (B, n)
        pred = integrate_trajectory(dynamics_fn, s0, t,
                                    horizon=rollout_horizon,
                                    integrator=integrator)                # (B, H+1, n)
        true = traj[:, : pred.shape[1], :]                               # align horizons
        return _rollout_loss_fn(pred, true)

    # ------------------------------------------------------------------
    # LBFGS closure
    # ------------------------------------------------------------------
    train_ptr = test_ptr = traj_ptr = traj_test_ptr = 0

    def closure():
        optimizer.zero_grad()

        # derivative supervision
        train_idx = np.arange(train_ptr, train_ptr + batch_sz) % n_train
        pred = _deriv_forward(dataset["train_input"][train_idx])
        tl   = loss_fn(pred, dataset["train_label"][train_idx])
        _state["train_loss"] = tl

        # integrated rollout loss
        if rollout_weight > 0.0 and has_train_traj:
            traj_idx = np.arange(traj_ptr, traj_ptr + tbatch) % n_traj_train
            rl = _rollout_loss_on_batch(
                dataset["train_traj"][traj_idx],
                dataset["train_t"],
            )
        else:
            rl = torch.tensor(0.0, device=pred.device)
        _state["rollout_train_loss"] = rl

        # regularisation
        if model.save_act:
            if reg_metric == "edge_backward":
                model.attribute()
            if reg_metric == "node_backward":
                model.node_attribute()
            reg = model.get_reg(
                reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
            )
        else:
            reg = torch.tensor(0.0, device=pred.device)
        _state["reg"] = reg

        obj = tl + rollout_weight * rl + lamb * reg
        obj.backward()
        return obj

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for step in range(steps):
        # --- pointer bookkeeping ---
        train_ptr_snapshot = train_ptr
        traj_ptr_snapshot  = traj_ptr

        # advance pointers
        train_ptr = (train_ptr + batch_sz)  % n_train
        test_ptr  = (test_ptr  + batch_sz_test) % n_test
        if has_train_traj and rollout_weight > 0.0:
            traj_ptr = (traj_ptr + tbatch) % n_traj_train
        if has_test_traj and rollout_weight > 0.0:
            traj_test_ptr = (traj_test_ptr + tbatch_test) % n_traj_test

        # current index arrays (for the closure and Adam path)
        train_idx = np.arange(train_ptr_snapshot,
                               train_ptr_snapshot + batch_sz) % n_train

        # --- grid update (PyKAN) ---
        in_grid_window = (
            update_grid
            and step % grid_update_freq == 0
            and (start_grid_update_step < 0 or step >= start_grid_update_step)
            and step < stop_grid_update_step
        )
        if in_grid_window:
            model.update_grid(dataset["train_input"][train_idx])

        # --- optimiser step ---
        if opt == "LBFGS":
            # closure uses train_ptr_snapshot implicitly via the outer train_ptr
            # reset it so the closure sees the right batch
            train_ptr = (train_ptr_snapshot + batch_sz) % n_train
            traj_ptr  = (traj_ptr_snapshot  + tbatch  ) % n_traj_train if (
                has_train_traj and rollout_weight > 0.0) else traj_ptr
            # revert to snapshot so closure picks the right batch
            # (Python closures capture the variable by reference)
            _closure_train_idx = np.arange(
                train_ptr_snapshot, train_ptr_snapshot + batch_sz) % n_train
            _closure_traj_idx  = (
                np.arange(traj_ptr_snapshot, traj_ptr_snapshot + tbatch) % n_traj_train
                if (has_train_traj and rollout_weight > 0.0) else None
            )

            def _lbfgs_closure():
                optimizer.zero_grad()
                pred = _deriv_forward(dataset["train_input"][_closure_train_idx])
                tl   = loss_fn(pred, dataset["train_label"][_closure_train_idx])
                _state["train_loss"] = tl

                if rollout_weight > 0.0 and has_train_traj:
                    rl = _rollout_loss_on_batch(
                        dataset["train_traj"][_closure_traj_idx],
                        dataset["train_t"],
                    )
                else:
                    rl = torch.tensor(0.0, device=pred.device)
                _state["rollout_train_loss"] = rl

                if model.save_act:
                    if reg_metric == "edge_backward":
                        model.attribute()
                    if reg_metric == "node_backward":
                        model.node_attribute()
                    reg = model.get_reg(
                        reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                    )
                else:
                    reg = torch.tensor(0.0, device=pred.device)
                _state["reg"] = reg

                obj = tl + rollout_weight * rl + lamb * reg
                obj.backward()
                return obj

            optimizer.step(_lbfgs_closure)

        else:  # Adam
            pred = _deriv_forward(dataset["train_input"][train_idx])
            tl   = loss_fn(pred, dataset["train_label"][train_idx])
            _state["train_loss"] = tl

            if rollout_weight > 0.0 and has_train_traj:
                traj_idx_adam = (
                    np.arange(traj_ptr_snapshot, traj_ptr_snapshot + tbatch)
                    % n_traj_train
                )
                rl = _rollout_loss_on_batch(
                    dataset["train_traj"][traj_idx_adam],
                    dataset["train_t"],
                )
            else:
                rl = torch.tensor(0.0, device=pred.device)
            _state["rollout_train_loss"] = rl

            if model.save_act:
                if reg_metric == "edge_backward":
                    model.attribute()
                if reg_metric == "node_backward":
                    model.node_attribute()
                reg = model.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg = torch.tensor(0.0, device=pred.device)
            _state["reg"] = reg

            obj = tl + rollout_weight * rl + lamb * reg
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()

        # --- test evaluation (no grad) ---
        with torch.no_grad():
            test_idx = np.arange(test_ptr, test_ptr + batch_sz_test) % n_test
            test_pred = _deriv_forward(dataset["test_input"][test_idx])
            test_loss = loss_fn(test_pred, dataset["test_label"][test_idx])

            if rollout_weight > 0.0 and has_test_traj:
                traj_test_idx = (
                    np.arange(traj_test_ptr, traj_test_ptr + tbatch_test)
                    % n_traj_test
                )
                rollout_test_loss = _rollout_loss_on_batch(
                    dataset["test_traj"][traj_test_idx],
                    dataset["test_t"],
                )
            else:
                rollout_test_loss = torch.tensor(0.0)

        # --- log ---
        tl_val  = float(_state["train_loss"].detach().cpu())
        rl_val  = float(_state["rollout_train_loss"].detach().cpu())
        reg_val = float(_state["reg"].detach().cpu())
        tst_val = float(test_loss.detach().cpu())
        rlt_val = float(rollout_test_loss.detach().cpu())

        results["train_loss"].append(tl_val)
        results["test_loss"].append(tst_val)
        results["reg"].append(reg_val)
        results["rollout_train_loss"].append(rl_val)
        results["rollout_test_loss"].append(rlt_val)

        if metrics is not None:
            for m in metrics:
                results[m.__name__].append(m().item())

        if step % log == 0:
            print(
                f"step {step:5d} | "
                f"train {tl_val:.6e} | test {tst_val:.6e} | "
                f"roll_train {rl_val:.6e} | roll_test {rlt_val:.6e} | "
                f"reg {reg_val:.6e}"
            )

        # early stopping: stop if train loss hasn't improved for `patience` steps
        if patience > 0:
            total_loss = tl_val + rollout_weight * rl_val
            if total_loss < _best_loss:
                _best_loss       = total_loss
                _no_improve      = 0
                import copy
                _best_state_dict = copy.deepcopy(model.state_dict())
            else:
                _no_improve += 1
            if _no_improve >= patience:
                print(f"[fit_kan] Early stopping at step {step} (no improvement for {patience} steps).")
                if _best_state_dict is not None:
                    model.load_state_dict(_best_state_dict)
                break

    model.log_history("fit_kan")
    model.symbolic_enabled = old_symbolic_enabled
    return results


# ---------------------------------------------------------------------------
# Discrete-map utilities
# ---------------------------------------------------------------------------

def make_windows(
    traj: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Slice a state trajectory into overlapping fixed-length windows.

    Used to build the ``train_traj`` / ``test_traj`` dataset entries for
    discrete-map systems (Hénon, Ikeda, Holling) where the rollout loss
    iterates the learned map rather than integrating a continuous ODE.

    Parameters
    ----------
    traj : torch.Tensor, shape (T, n)
        Contiguous state sequence (T time steps, n state dimensions).
    window : int
        Length of each window (number of steps, **inclusive** of initial
        condition).  Should be ``rollout_horizon + 1``.

    Returns
    -------
    windows : torch.Tensor, shape (T - window + 1, window, n)
        All contiguous sub-sequences of length ``window``.

    Examples
    --------
    >>> t = torch.arange(10).float().unsqueeze(1)   # (10, 1)
    >>> w = make_windows(t, window=4)
    >>> w.shape
    torch.Size([7, 4, 1])
    """
    T = traj.shape[0]
    n_windows = T - window + 1
    return torch.stack([traj[i : i + window] for i in range(n_windows)], dim=0)


# ---------------------------------------------------------------------------
# Convenience: numpy-based RK4 for post-fit rollout (no grad needed)
# ---------------------------------------------------------------------------

def rk4_integrate_numpy(
    rhs: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 integrator operating on numpy arrays.

    Useful for evaluating the fitted model after training (e.g., generating
    long attractor trajectories for publication figures).

    Parameters
    ----------
    rhs    : callable (n,) -> (n,)  right-hand side f(y)
    y0     : (n,) initial condition
    t_span : (t_start, t_end)
    dt     : time step

    Returns
    -------
    ts : (T,) time values
    ys : (T, n) trajectory
    """
    t = t_span[0]
    y = np.array(y0, dtype=np.float64)
    ts, ys = [t], [y.copy()]
    while t < t_span[1] - 1e-12:
        h  = min(dt, t_span[1] - t)
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y  = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.array(ys)
