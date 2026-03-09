"""kandy.plotting — Publication-quality visualisation for KANDy models.

All plot functions follow a consistent pattern:
  - Accept an optional ``ax`` / ``fig`` argument so they compose into
    larger figure layouts.
  - Return the ``(fig, ax)`` or ``(fig, axes)`` they drew on.
  - Accept ``save`` (path prefix) to write PNG + PDF at 300 dpi.
  - Never call ``plt.show()`` — the caller decides when to display.

Modules in use:
  - Edge activations : ``get_edge_activation``, ``plot_edge``, ``plot_all_edges``
  - Curve fitting    : ``fit_linear``, ``fit_sine``, ``fit_sech2``,
                       ``fit_sech2_tanh``, ``fit_polynomial``
  - Training         : ``plot_loss_curves``
  - Trajectory       : ``plot_attractor_overlay``, ``plot_trajectory_error``
  - Architecture     : ``plot_kan_architecture``

Typical usage
-------------
>>> from kandy.plotting import get_edge_activation, plot_edge, plot_all_edges
>>> model.save_act = True
>>> model(dataset["train_input"])
>>> x, y = get_edge_activation(model, l=0, i=2, j=0)
>>> fig, ax = plot_edge(x, y, fits=["linear", "sine"])
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Publication rc defaults (applied lazily so the user can override them)
# ---------------------------------------------------------------------------
_PUB_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "axes.linewidth":     0.7,
    "lines.linewidth":    1.5,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi":         150,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}


def use_pub_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update(_PUB_RC)


def _save(fig: plt.Figure, save: Optional[str]) -> None:
    """Save *fig* as ``{save}.png`` and ``{save}.pdf`` if *save* is not None."""
    if save is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
        fig.savefig(f"{save}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save}.pdf", dpi=300, bbox_inches="tight")


# ===========================================================================
# 1.  Edge activation extraction
# ===========================================================================

import torch


@torch.no_grad()
def get_edge_activation(
    model,
    l: int,
    i: int,
    j: int,
    X: Optional["torch.Tensor"] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the learned spline activation for edge (l, i, j).

    The KAN spline on edge (l, i, j) maps the i-th input of layer l to a
    contribution towards the j-th output of layer l+1.  This function
    returns the (input, output) scatter for that spline, sorted by input.

    Parameters
    ----------
    model : KAN
        A PyKAN model.  Must have ``model.save_act = True`` and a forward
        pass must have been run to populate ``model.acts`` and
        ``model.spline_postacts``.
    l : int
        Layer index (0-based).
    i : int
        Input feature index (column of layer l's input).
    j : int
        Output index (which output of layer l+1 this edge feeds into).
    X : torch.Tensor, optional
        If provided, run a fresh forward pass with ``save_act=True`` first.

    Returns
    -------
    x : np.ndarray, shape (N,)   sorted input values to the spline
    y : np.ndarray, shape (N,)   corresponding spline output values
    """
    if X is not None:
        model.save_act = True
        model(X)

    if not hasattr(model, "acts") or model.acts is None:
        raise RuntimeError(
            "model.acts is empty.  Run a forward pass with model.save_act=True first, "
            "or pass X to get_edge_activation()."
        )

    rank = torch.argsort(model.acts[l][:, i]).cpu().numpy()
    x = model.acts[l][:, i][rank].cpu().detach().numpy()
    y = model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy()
    return x, y


def get_all_edge_activations(
    model,
    X: Optional["torch.Tensor"] = None,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Extract activations for every edge in the model.

    Returns
    -------
    activations : dict  mapping (l, i, j) -> (x, y)
    """
    if X is not None:
        model.save_act = True
        with torch.no_grad():
            model(X)

    result = {}
    for l in range(len(model.width_in) - 1):
        for i in range(int(model.width_in[l])):
            for j in range(int(model.width_out[l + 1])):
                result[(l, i, j)] = get_edge_activation(model, l, i, j)
    return result


# ===========================================================================
# 2.  Curve fitting utilities
# ===========================================================================

def fit_linear(
    x: np.ndarray, y: np.ndarray
) -> dict:
    """Fit y = m*x + b and return parameters + R²."""
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"name": "linear", "params": (m, b), "y_hat": y_hat, "r2": r2,
            "label": f"linear: {m:.3f}x + {b:.3f}  ($R^2$={r2:.3f})"}


def fit_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int = 2
) -> dict:
    """Fit y = p(x) with a polynomial of given degree and return R²."""
    coeffs = np.polyfit(x, y, degree)
    y_hat = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    term_strs = []
    for k, c in enumerate(coeffs):
        power = degree - k
        if power == 0:
            term_strs.append(f"{c:.3f}")
        elif power == 1:
            term_strs.append(f"{c:.3f}x")
        else:
            term_strs.append(f"{c:.3f}x^{power}")
    label = "poly: " + " + ".join(term_strs) + f"  ($R^2$={r2:.3f})"
    return {"name": "polynomial", "degree": degree, "params": coeffs,
            "y_hat": y_hat, "r2": r2, "label": label}


def fit_sine(
    x: np.ndarray, y: np.ndarray
) -> dict:
    """Fit y = A*sin(w*x + phi) + c and return parameters + R²."""
    def _model(x, A, w, phi, c):
        return A * np.sin(w * x + phi) + c

    p0 = [np.ptp(y) / 2, 1.0, 0.0, np.mean(y)]
    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=20_000)
    except RuntimeError:
        params = np.array(p0)

    y_hat = _model(x, *params)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    A, w, phi, c = params
    label = f"sine: {A:.3f}·sin({w:.3f}x+{phi:.3f})+{c:.3f}  ($R^2$={r2:.3f})"
    return {"name": "sine", "params": params, "y_hat": y_hat, "r2": r2, "label": label}


def fit_sech2(
    x: np.ndarray, y: np.ndarray
) -> dict:
    """Fit y = a - A*sech²((x-x0)/ell)  (even, dip-shaped shock profile).

    This shape arises naturally in edge activations for the inviscid Burgers
    equation and other PDE shock solutions.
    """
    def _model(x, a, A, x0, ell):
        z = (x - x0) / np.maximum(np.abs(ell), 1e-12)
        return a - A * (1.0 / np.cosh(z)) ** 2

    a0  = np.median(y)
    A0  = a0 - np.min(y)
    x0_0 = x[np.argmin(y)]
    ell0 = np.std(x) / 5.0 or 1.0
    p0 = [a0, A0, x0_0, ell0]
    bounds = ([-np.inf, -np.inf, x.min() - 1e-6, 1e-6],
              [ np.inf,  np.inf, x.max() + 1e-6, np.inf])
    p0 = [np.clip(p, lo + 1e-8, hi - 1e-8) for p, lo, hi in zip(p0, bounds[0], bounds[1])]

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, bounds=bounds, maxfev=200_000)
    except (RuntimeError, ValueError):
        params = np.array(p0)

    y_hat = _model(x, *params)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    a, A, x0, ell = params
    label = f"sech²: {a:.3f} − {A:.3f}·sech²((x−{x0:.3f})/{ell:.3f})  ($R^2$={r2:.3f})"
    return {"name": "sech2", "params": params, "y_hat": y_hat, "r2": r2, "label": label,
            "_model": _model}


def fit_sech2_tanh(
    x: np.ndarray, y: np.ndarray
) -> dict:
    """Fit y = c + K*sech²((x-x0)/ell)*tanh((x-x0)/ell)  (odd shock shape).

    This is the derivative of the sech² profile and corresponds to the
    u*u_x term in Burgers' equation edge activations.
    """
    def _model(x, c, K, x0, ell):
        z = (x - x0) / np.maximum(np.abs(ell), 1e-12)
        return c + K * (1.0 / np.cosh(z)) ** 2 * np.tanh(z)

    # Initial guesses: odd function, so median ≈ c, amplitude from peak
    c0 = np.median(y)
    # Estimate x0 as zero-crossing of (y - c0)
    y_centered = y - c0
    zero_cross = x[np.argmin(np.abs(y_centered))]
    x0_0 = zero_cross
    ell0 = np.std(x) / 8.0 or 1.0
    K0 = np.sign(np.corrcoef(x - x0_0, y_centered)[0, 1] + 1e-12) * np.ptp(y)
    p0 = [c0, K0, x0_0, ell0]
    bounds = ([-np.inf, -np.inf, x.min() - 1e-6, 1e-6],
              [ np.inf,  np.inf, x.max() + 1e-6, np.inf])
    # Clamp initial guess to be inside bounds
    p0 = [np.clip(p, lo + 1e-8, hi - 1e-8) for p, lo, hi in zip(p0, bounds[0], bounds[1])]

    try:
        params, _ = curve_fit(_model, x, y, p0=p0, bounds=bounds, maxfev=200_000)
    except (RuntimeError, ValueError):
        params = np.array(p0)

    y_hat = _model(x, *params)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    c, K, x0, ell = params
    label = f"sech²·tanh: {c:.3f} + {K:.3f}·sech²·tanh  ($R^2$={r2:.3f})"
    return {"name": "sech2_tanh", "params": params, "y_hat": y_hat, "r2": r2,
            "label": label, "_model": _model}


_FIT_REGISTRY = {
    "linear":     fit_linear,
    "polynomial": fit_polynomial,
    "sine":       fit_sine,
    "sech2":      fit_sech2,
    "sech2_tanh": fit_sech2_tanh,
}


# ===========================================================================
# 3.  Single-edge plot
# ===========================================================================

def plot_edge(
    x: np.ndarray,
    y: np.ndarray,
    fits: Sequence[str] = (),
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = r"$\theta_i$ (spline input)",
    ylabel: str = r"$\psi_i(\theta_i)$ (spline output)",
    scatter_kw: Optional[dict] = None,
    poly_degree: int = 2,
    x_grid_n: int = 500,
    save: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single edge activation with optional curve fits.

    Parameters
    ----------
    x, y : np.ndarray
        Sorted edge activation data from ``get_edge_activation``.
    fits : sequence of str
        Any combination of ``'linear'``, ``'polynomial'``, ``'sine'``,
        ``'sech2'``, ``'sech2_tanh'``.  Each fit is computed and overlaid.
    ax : Axes, optional
        Axes to draw on.  A new figure is created if None.
    title : str
        Axes title (e.g. ``'Edge (0, 2, 1)'``).
    xlabel, ylabel : str
        Axis labels (LaTeX strings accepted).
    scatter_kw : dict, optional
        Passed to ``ax.scatter``.  Default: ``{s: 12, alpha: 0.6}``.
    poly_degree : int
        Degree for polynomial fit (used only when ``'polynomial'`` in fits).
    x_grid_n : int
        Number of points on smooth x-grid for fit curves.
    save : str, optional
        Path prefix (without extension).  Saves PNG + PDF.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    skw = {"s": 12, "alpha": 0.6, "color": "#1f77b4", "rasterized": True,
           "label": "activation", "zorder": 1}
    if scatter_kw:
        skw.update(scatter_kw)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    ax.scatter(xs, ys, **skw)

    x_grid = np.linspace(xs.min(), xs.max(), x_grid_n)

    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for k, fit_name in enumerate(fits):
        if fit_name not in _FIT_REGISTRY:
            warnings.warn(f"Unknown fit type {fit_name!r}. Skipping.")
            continue
        fit_fn = _FIT_REGISTRY[fit_name]
        kwargs = {}
        if fit_name == "polynomial":
            kwargs["degree"] = poly_degree
        result = fit_fn(xs, ys, **kwargs)

        # Evaluate on smooth grid
        if fit_name == "linear":
            m, b = result["params"]
            y_smooth = m * x_grid + b
        elif fit_name == "polynomial":
            y_smooth = np.polyval(result["params"], x_grid)
        elif fit_name in ("sine", "sech2", "sech2_tanh"):
            y_smooth = result["_model"](x_grid, *result["params"]) if "_model" in result else (
                result["y_hat"]  # fallback: use training-point evaluations
            )
        else:
            y_smooth = np.interp(x_grid, xs, result["y_hat"])

        ax.plot(x_grid, y_smooth, color=colors[k % len(colors)],
                linewidth=2.0, label=result["label"], zorder=k + 2)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if fits:
        ax.legend(fontsize=7.5, framealpha=0.85)

    _save(fig, save)
    return fig, ax


# ===========================================================================
# 4.  All-edges grid
# ===========================================================================

def plot_all_edges(
    model,
    X: Optional["torch.Tensor"] = None,
    *,
    fits: Sequence[str] = (),
    in_var_names: Optional[Sequence[str]] = None,
    out_var_names: Optional[Sequence[str]] = None,
    figsize_per_panel: tuple[float, float] = (3.0, 2.5),
    poly_degree: int = 2,
    save: Optional[str] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot all edge activations in a grid (inputs × outputs).

    One panel per edge (l=0, i, j) for a single-layer KAN.  For multi-layer
    models only layer 0 is shown (pass the layer index explicitly to
    ``plot_edge`` for deeper layers).

    Parameters
    ----------
    model : KAN
        Trained PyKAN model.
    X : torch.Tensor, optional
        Run a forward pass first to populate activations.
    fits : sequence of str
        Fit types to overlay on each panel (see ``plot_edge``).
    in_var_names : list of str, optional
        Names for the input (lifted feature) dimensions.
    out_var_names : list of str, optional
        Names for the output (derivative) dimensions.
    figsize_per_panel : (width, height)
        Size of each individual panel in inches.
    poly_degree : int
        Polynomial degree for ``'polynomial'`` fits.
    save : str, optional
        Path prefix for PNG + PDF output.

    Returns
    -------
    fig, axes : Figure and (n_in, n_out) array of Axes
    """
    if X is not None:
        model.save_act = True
        with torch.no_grad():
            model(X)

    l = 0
    n_in  = int(model.width_in[l])
    n_out = int(model.width_out[l + 1])

    if in_var_names is None:
        in_var_names = [rf"$\theta_{{{i}}}$" for i in range(n_in)]
    if out_var_names is None:
        out_var_names = [rf"$\dot{{x}}_{{{j}}}$" for j in range(n_out)]

    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(
        n_out, n_in,
        figsize=(pw * n_in, ph * n_out),
        squeeze=False,
    )

    for j in range(n_out):
        for i in range(n_in):
            ax = axes[j, i]
            try:
                x, y = get_edge_activation(model, l, i, j)
                plot_edge(
                    x, y,
                    fits=fits,
                    ax=ax,
                    title=f"edge ({l},{i},{j})",
                    xlabel=in_var_names[i] if i < len(in_var_names) else f"θ_{i}",
                    ylabel=in_var_names[i] + "→" + out_var_names[j]
                           if j == 0 else "",
                    poly_degree=poly_degree,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="red")
                ax.set_title(f"edge ({l},{i},{j})", fontsize=9)

        # y-axis label for each row
        axes[j, 0].set_ylabel(out_var_names[j] if j < len(out_var_names)
                               else f"output {j}")

    fig.tight_layout(pad=0.5)
    _save(fig, save)
    return fig, axes


# ===========================================================================
# 5.  KAN architecture figure (wraps model.plot)
# ===========================================================================

def plot_kan_architecture(
    model,
    X: "torch.Tensor",
    *,
    beta: float = 3.0,
    in_var_names: Optional[Sequence[str]] = None,
    out_var_names: Optional[Sequence[str]] = None,
    save: Optional[str] = None,
) -> plt.Figure:
    """Render PyKAN's built-in architecture diagram and save it.

    Runs a forward pass with ``save_act=True``, calls ``model.plot()``,
    and saves PNG + PDF.

    Parameters
    ----------
    model : KAN
    X : torch.Tensor
        Input tensor used to populate activations (e.g., ``dataset['train_input']``).
    beta : float
        Passed to ``model.plot(beta=...)``; controls the activation display scale.
    in_var_names : list of str, optional
        LaTeX strings for input nodes.
    out_var_names : list of str, optional
        LaTeX strings for output nodes.
    save : str, optional
        Path prefix (no extension).

    Returns
    -------
    fig : Figure
    """
    model.save_act = True
    with torch.no_grad():
        model(X)

    plot_kw: dict = {"beta": beta}
    if in_var_names is not None:
        plot_kw["in_vars"] = in_var_names
    if out_var_names is not None:
        plot_kw["out_vars"] = out_var_names

    model.plot(**plot_kw)
    fig = plt.gcf()
    _save(fig, save)
    return fig


# ===========================================================================
# 6.  Training loss curves
# ===========================================================================

def plot_loss_curves(
    results: dict,
    *,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    show_rollout: bool = True,
    save: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training / test loss and rollout loss from ``fit_kan`` results.

    Parameters
    ----------
    results : dict
        Return value of ``fit_kan()``.  Expected keys:
        ``train_loss``, ``test_loss``, ``rollout_train_loss``,
        ``rollout_test_loss``, ``reg``.
    ax : Axes, optional
    log_scale : bool
        Use log y-axis (default True).
    show_rollout : bool
        Overlay rollout loss curves (default True).
    save : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
    else:
        fig = ax.get_figure()

    steps = np.arange(len(results["train_loss"]))

    ax.plot(steps, results["train_loss"], color="#1f77b4", lw=1.5, label="train")
    ax.plot(steps, results["test_loss"],  color="#1f77b4", lw=1.5, ls="--",
            alpha=0.7, label="test")

    if show_rollout and "rollout_train_loss" in results:
        rl_train = np.array(results["rollout_train_loss"])
        rl_test  = np.array(results["rollout_test_loss"])
        if rl_train.max() > 0:
            ax.plot(steps, rl_train, color="#d62728", lw=1.5,
                    label="rollout train")
            ax.plot(steps, rl_test,  color="#d62728", lw=1.5, ls="--",
                    alpha=0.7, label="rollout test")

    if "reg" in results:
        reg = np.array(results["reg"])
        if reg.max() > 0:
            ax.plot(steps, reg, color="#7f7f7f", lw=1.0, ls=":",
                    label="regularisation")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.5, which="both")

    _save(fig, save)
    return fig, ax


# ===========================================================================
# 7.  Attractor / trajectory overlay
# ===========================================================================

def plot_attractor_overlay(
    true_traj: np.ndarray,
    *other_trajs: np.ndarray,
    dim_x: int = 0,
    dim_y: int = 2,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    drop: int = 0,
    lw: float = 1.0,
    alpha_true: float = 0.35,
    save: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Publication-quality attractor comparison.

    Plots the true trajectory as a light grey background and overlays one or
    more model trajectories in distinct colours.

    Parameters
    ----------
    true_traj : np.ndarray, shape (T, n)
        The ground-truth trajectory used as the background reference.
    *other_trajs : np.ndarray, shape (T, n)
        Additional trajectories to overlay (e.g., KAN rollout, symbolic).
    dim_x, dim_y : int
        Which state dimensions to use as x- and y-axis.
    labels : list of str, optional
        Legend labels for [true, other_1, other_2, ...].
    colors : list of str, optional
        Colours for [true, other_1, other_2, ...].
    ax : Axes, optional
    xlabel, ylabel : str, optional
        Axis labels (defaults to ``x_{dim_x}`` etc.).
    xlim, ylim : tuple, optional
    drop : int
        Drop the first ``drop`` points (removes transient).
    lw : float
        Line width for the overlay trajectories.
    alpha_true : float
        Alpha for the background true trajectory.
    save : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.3))
    else:
        fig = ax.get_figure()

    _labels = labels or (["True system"] +
                         [f"Model {k+1}" for k in range(len(other_trajs))])
    _colors = colors or (["#B8B8B8"] + ["#111111", "#555555",
                                          "#d62728", "#1f77b4"][: len(other_trajs)])

    # Background: true attractor (light grey, rasterised for file size)
    t = true_traj[drop:]
    ax.plot(t[:, dim_x], t[:, dim_y],
            color=_colors[0], lw=0.5, alpha=alpha_true,
            zorder=1, label=_labels[0], rasterized=True)

    line_styles = ["-", (0, (5, 2.5)), (0, (3, 1, 1, 1)), "--"]
    for k, traj in enumerate(other_trajs):
        tr = traj[drop:]
        ax.plot(tr[:, dim_x], tr[:, dim_y],
                color=_colors[k + 1],
                lw=lw,
                ls=line_styles[k % len(line_styles)],
                alpha=0.92,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=k + 2,
                label=_labels[k + 1] if k + 1 < len(_labels) else f"traj {k+1}")

    ax.set_xlabel(xlabel or f"$x_{{{dim_x}}}$")
    ax.set_ylabel(ylabel or f"$x_{{{dim_y}}}$")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=True, framealpha=0.92, edgecolor="#888888",
              fancybox=False, borderpad=0.4)

    fig.tight_layout(pad=0.4)
    _save(fig, save)
    return fig, ax


# ===========================================================================
# 8.  Trajectory error vs time
# ===========================================================================

def plot_trajectory_error(
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    t: Optional[np.ndarray] = None,
    *,
    lyapunov_time: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    label: str = "KANDy",
    color: str = "#1f77b4",
    save: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pointwise RMSE between true and predicted trajectories over time.

    Parameters
    ----------
    true_traj, pred_traj : np.ndarray, shape (T, n)
    t : np.ndarray, shape (T,), optional
        Time values.  If None, integer steps are used.
    lyapunov_time : float, optional
        If provided, a vertical dashed line is drawn at this time, and the
        x-axis is labelled in units of Lyapunov time τ_L.
    ax : Axes, optional
    log_scale : bool
        Log y-axis (default True).
    label : str
        Legend label for the error curve.
    color : str
    save : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
    else:
        fig = ax.get_figure()

    rmse = np.sqrt(np.mean((true_traj - pred_traj) ** 2, axis=-1))

    if t is None:
        t = np.arange(len(rmse), dtype=float)

    if lyapunov_time is not None:
        t_plot = t / lyapunov_time
        ax.set_xlabel(r"time / $\tau_L$")
        ax.axvline(x=1.0, color="#888888", lw=0.8, ls="--", alpha=0.7,
                   label=r"$\tau_L$")
    else:
        t_plot = t
        ax.set_xlabel("time")

    ax.plot(t_plot[: len(rmse)], rmse, color=color, lw=1.5, label=label)

    if log_scale:
        ax.set_yscale("log")

    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.5, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, save)
    return fig, ax
