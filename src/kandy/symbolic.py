"""kandy.symbolic — Complexity-aware symbolic extraction for KANDy.

PyKAN's ``auto_symbolic()`` assigns every edge the same symbolic function
library and the same complexity cost.  For physics-informed feature libraries
(e.g. Holling functional-response terms or Ikeda trig rotations) this is
suboptimal: a known-physics feature like ``N/(1+ahN)`` should be preferred over
a generic polynomial when both fit equally well.

This module provides two complementary tools:

1. **Pre-built symbolic libraries** with configurable complexity costs.
2. **``auto_symbolic_with_costs``** — a drop-in replacement for
   ``model.auto_symbolic()`` that assigns a *different* library (and therefore
   different complexity costs) to edges depending on whether their input
   feature is in a user-specified "preferred" set.

The rule is simple:
- Edges whose input feature index ``i`` is in ``preferred_idx`` use
  ``preferred_lib`` (cheap — the solver prefers these).
- All other edges use ``other_lib`` (expensive — the solver avoids these
  unless the fit is clearly better).

Usage
-----
>>> from kandy.symbolic import auto_symbolic_with_costs, make_symbolic_lib
>>> import torch, sympy as sp
>>>
>>> # Physics-preferred extraction for Holling Type II
>>> physics_idx = {6, 7, 8, 9, 10, 11, 12}   # indices of Holling features
>>> auto_symbolic_with_costs(
...     model,
...     preferred_idx=physics_idx,
...     weight_simple=0.7,
...     r2_threshold=0.90,
... )
>>>
>>> # Complexity-aware extraction for Ikeda map (all features are "simple")
>>> auto_symbolic_with_costs(
...     model,
...     preferred_idx=set(range(4)),   # all 4 features are physics-informed
...     weight_simple=0.1,
...     r2_threshold=0.80,
... )
"""
from __future__ import annotations

from typing import Callable, Set

import sympy as sp
import torch

__all__ = [
    "make_symbolic_lib",
    "POLY_LIB",
    "TRIG_LIB",
    "POLY_LIB_CHEAP",
    "TRIG_LIB_CHEAP",
    "auto_symbolic_with_costs",
    "score_formula",
    "formulas_to_latex",
    "substitute_params",
]


# ---------------------------------------------------------------------------
# Library builder
# ---------------------------------------------------------------------------

def make_symbolic_lib(
    entries: dict[str, tuple[Callable, Callable, int]],
) -> dict:
    """Build a PyKAN-compatible symbolic library dict.

    Parameters
    ----------
    entries : dict
        Mapping from name to ``(torch_fn, sympy_fn, complexity_cost)``.
        ``complexity_cost`` is a non-negative integer; lower = more preferred.

    Returns
    -------
    lib : dict
        In the format expected by ``model.fix_symbolic`` and
        ``model.suggest_symbolic``:
        ``{name: (torch_fn, sympy_fn, cost, bound_fn)}``.

    Examples
    --------
    >>> import torch, sympy as sp
    >>> lib = make_symbolic_lib({
    ...     'x':   (lambda x: x,          lambda x: x,          1),
    ...     'x^2': (lambda x: x**2,       lambda x: x**2,       2),
    ...     'sin': (torch.sin,            sp.sin,                3),
    ...     '0':   (lambda x: x * 0,      lambda x: x * 0,      0),
    ... })
    """
    return {
        name: (
            torch_fn,
            sympy_fn,
            cost,
            lambda x, y_th, _fn=torch_fn: ((), _fn(x)),
        )
        for name, (torch_fn, sympy_fn, cost) in entries.items()
    }


# ---------------------------------------------------------------------------
# Pre-built libraries
# ---------------------------------------------------------------------------

#: Polynomial library (x, x², x³, 0) with **low** complexity costs.
#: Use for physics-informed or expected features.
POLY_LIB_CHEAP: dict = make_symbolic_lib({
    "x":   (lambda x: x,           lambda x: x,           1),
    "x^2": (lambda x: x ** 2,      lambda x: x ** 2,      2),
    "x^3": (lambda x: x ** 3,      lambda x: x ** 3,      3),
    "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
})

#: Polynomial library with **high** complexity costs.
#: Use for generic / unexpected features to discourage their selection.
POLY_LIB: dict = make_symbolic_lib({
    "x":   (lambda x: x,           lambda x: x,           3),
    "x^2": (lambda x: x ** 2,      lambda x: x ** 2,      4),
    "x^3": (lambda x: x ** 3,      lambda x: x ** 3,      5),
    "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
})

#: Trig + polynomial library with **low** costs.
#: Suitable when the system naturally contains sin/cos terms (e.g. Ikeda).
TRIG_LIB_CHEAP: dict = make_symbolic_lib({
    "x":   (lambda x: x,           lambda x: x,           1),
    "x^2": (lambda x: x ** 2,      lambda x: x ** 2,      2),
    "x^3": (lambda x: x ** 3,      lambda x: x ** 3,      3),
    "sin": (torch.sin,             sp.sin,                2),
    "cos": (torch.cos,             sp.cos,                2),
    "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
})

#: Trig + polynomial library with **high** costs.
#: Use for complex / unexpected trig features.
TRIG_LIB: dict = make_symbolic_lib({
    "x":   (lambda x: x,           lambda x: x,           3),
    "x^2": (lambda x: x ** 2,      lambda x: x ** 2,      4),
    "x^3": (lambda x: x ** 3,      lambda x: x ** 3,      4),
    "sin": (torch.sin,             sp.sin,                4),
    "cos": (torch.cos,             sp.cos,                4),
    "0":   (lambda x: x * 0,       lambda x: x * 0,       0),
})


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def auto_symbolic_with_costs(
    model,
    preferred_idx: Set[int],
    *,
    preferred_lib: dict | None = None,
    other_lib: dict | None = None,
    a_range: tuple[float, float] = (-10.0, 10.0),
    b_range: tuple[float, float] = (-10.0, 10.0),
    weight_simple: float = 0.8,
    r2_threshold: float = 0.90,
    verbose: int = 1,
) -> None:
    """Complexity-aware symbolic fitting for KANDy models.

    Iterates over every edge (l, i, j) in the KAN and fits the best symbolic
    function from a library whose complexity costs depend on whether the edge's
    input feature index ``i`` is in ``preferred_idx``.

    Edges with ``i ∈ preferred_idx`` use ``preferred_lib`` (cheap costs) —
    the solver prefers these when fit quality is comparable.  All other edges
    use ``other_lib`` (expensive costs) — the solver avoids these.

    Parameters
    ----------
    model : KAN
        A fitted PyKAN model with ``model.save_act = True`` and activations
        populated by a recent forward pass.
    preferred_idx : set of int
        Input feature indices (0-based) that should be preferred.
        For Holling Type II: indices of the functional-response features.
        For Ikeda: all 4 trig-product features (preferred over alternatives).
    preferred_lib : dict, optional
        PyKAN symbolic library with **low** complexity costs for preferred edges.
        Defaults to :data:`POLY_LIB_CHEAP`.
    other_lib : dict, optional
        PyKAN symbolic library with **high** complexity costs for other edges.
        Defaults to :data:`POLY_LIB`.
    a_range : tuple
        Search range for the linear ``a`` coefficient in ``y = a·f(x) + b``.
    b_range : tuple
        Search range for the linear ``b`` bias.
    weight_simple : float
        Trade-off between fit quality (R²) and complexity cost.  Lower values
        favour simpler functions even with slightly worse R².
    r2_threshold : float
        Minimum R² required to fix an edge symbolically.  Edges below this
        threshold are set to zero.
    verbose : int
        0 = silent, 1 = one line per edge (default), 2 = full PyKAN output.

    Notes
    -----
    This function modifies the model in-place by calling ``fix_symbolic`` on
    every edge.  Call ``model.unfix_symbolic_all()`` to reset before retrying.

    Examples
    --------
    Holling Type II — prefer the functional-response features:

    >>> physics_idx = {
    ...     feat_names.index("denom"),
    ...     feat_names.index("invden"),
    ...     feat_names.index("fN"),
    ...     feat_names.index("logistic"),
    ...     feat_names.index("pred"),
    ...     feat_names.index("gainP"),
    ...     feat_names.index("deathP"),
    ... }
    >>> auto_symbolic_with_costs(model, preferred_idx=physics_idx,
    ...                          weight_simple=0.7, r2_threshold=0.90)

    Ikeda map — all 4 trig features are physics-informed:

    >>> auto_symbolic_with_costs(model, preferred_idx=set(range(4)),
    ...                          preferred_lib=TRIG_LIB_CHEAP,
    ...                          other_lib=TRIG_LIB,
    ...                          weight_simple=0.1, r2_threshold=0.80)
    """
    if preferred_lib is None:
        preferred_lib = POLY_LIB_CHEAP
    if other_lib is None:
        other_lib = POLY_LIB

    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                is_preferred = i in preferred_idx
                lib = preferred_lib if is_preferred else other_lib

                name, fun, r2, c = model.suggest_symbolic(
                    l, i, j,
                    a_range=a_range,
                    b_range=b_range,
                    lib=lib,
                    verbose=False,
                    weight_simple=weight_simple,
                )

                if r2 >= r2_threshold:
                    model.fix_symbolic(
                        l, i, j, name,
                        verbose=verbose > 1,
                        log_history=False,
                    )
                    if verbose >= 1:
                        tag = "PHYS" if is_preferred else "GEN"
                        print(
                            f"  fix ({l},{i},{j}) [{tag}] → {name}  "
                            f"R²={r2:.4f}  cost={c}"
                        )
                else:
                    model.fix_symbolic(
                        l, i, j, "0",
                        verbose=verbose > 1,
                        log_history=False,
                    )
                    if verbose >= 1:
                        tag = "PHYS" if is_preferred else "GEN"
                        print(
                            f"  zero ({l},{i},{j}) [{tag}]  "
                            f"best={name}  R²={r2:.4f} < {r2_threshold}"
                        )

    model.log_history("auto_symbolic_with_costs")


# ---------------------------------------------------------------------------
# Formula scoring
# ---------------------------------------------------------------------------

def score_formula(
    formulas: list,
    theta: "np.ndarray",
    y_true: "np.ndarray",
    var_names: list[str],
) -> list[float]:
    """Compute R² of each symbolic formula against held-out data.

    Lambdifies each SymPy expression in ``formulas``, evaluates it on the
    feature matrix ``theta``, and returns the coefficient of determination R².

    Parameters
    ----------
    formulas : list of sympy.Expr
        One expression per output dimension (from :meth:`KANDy.get_formula`).
    theta : np.ndarray, shape (N, m)
        Feature matrix (lifted coordinates), same normalisation as during fit.
    y_true : np.ndarray, shape (N,) or (N, n_out)
        Ground-truth targets (derivatives or next states).
    var_names : list of str
        Variable names matching the columns of ``theta``, in order.
        These must match the symbol names used in ``formulas``.

    Returns
    -------
    r2_scores : list of float
        R² for each output dimension.  Values near 1.0 indicate an accurate
        symbolic recovery.  Negative values indicate the formula is worse than
        a constant predictor.

    Examples
    --------
    >>> formulas = model.get_formula()
    >>> r2 = score_formula(formulas, Theta_test_n, Y_test, FEATURE_NAMES)
    >>> print([f"{v:.4f}" for v in r2])
    """
    import numpy as np

    syms = [sp.Symbol(n) for n in var_names]
    y = np.asarray(y_true)
    if y.ndim == 1:
        y = y[:, None]

    r2_scores = []
    for col, expr in enumerate(formulas):
        try:
            fn = sp.lambdify(syms, expr, modules="numpy")
            y_pred = fn(*[theta[:, i] for i in range(len(syms))])
            y_pred = np.asarray(y_pred, dtype=float)
            if y_pred.ndim == 0:
                y_pred = np.full(len(theta), float(y_pred))
            y_col = y[:, col] if col < y.shape[1] else y[:, 0]
            ss_res = np.sum((y_col - y_pred) ** 2)
            ss_tot = np.sum((y_col - y_col.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-14)
        except Exception:
            r2 = float("nan")
        r2_scores.append(float(r2))

    return r2_scores


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def formulas_to_latex(
    formulas: list,
    lhs_names: list[str] | None = None,
    *,
    environment: str = "align*",
) -> str:
    """Convert a list of SymPy expressions to a LaTeX equation block.

    Parameters
    ----------
    formulas : list of sympy.Expr
        Symbolic expressions, one per output dimension.
    lhs_names : list of str, optional
        Left-hand-side names (e.g. ``['\\dot{x}', '\\dot{y}', '\\dot{z}']``).
        Defaults to ``['f_0', 'f_1', ...]``.
    environment : str
        LaTeX math environment (default ``'align*'``).

    Returns
    -------
    latex_str : str
        Complete LaTeX block, ready to paste into a paper.

    Examples
    --------
    >>> tex = formulas_to_latex(formulas, [r'\\dot{x}', r'\\dot{y}', r'\\dot{z}'])
    >>> print(tex)
    \\begin{align*}
      \\dot{x} &= ...  \\\\
      \\dot{y} &= ...  \\\\
      \\dot{z} &= ...
    \\end{align*}
    """
    if lhs_names is None:
        lhs_names = [f"f_{i}" for i in range(len(formulas))]

    lines = []
    for i, (lhs, expr) in enumerate(zip(lhs_names, formulas)):
        rhs = sp.latex(expr)
        sep = r" \\" if i < len(formulas) - 1 else ""
        lines.append(f"  {lhs} &= {rhs}{sep}")

    body = "\n".join(lines)
    return f"\\begin{{{environment}}}\n{body}\n\\end{{{environment}}}"


# ---------------------------------------------------------------------------
# Parameter substitution
# ---------------------------------------------------------------------------

def substitute_params(
    formulas: list,
    params: dict,
) -> list:
    """Substitute known parameter values into symbolic formulas.

    Parameters
    ----------
    formulas : list of sympy.Expr
        Symbolic expressions (output of :meth:`KANDy.get_formula`).
    params : dict
        Mapping from symbol name (str) or SymPy symbol to numeric value.
        Example: ``{'r': 1.0, 'K': 50.0, 'a': 1.2}``.

    Returns
    -------
    substituted : list of sympy.Expr
        Formulas with parameters replaced by their numeric values and
        simplified via :func:`sympy.nsimplify` where possible.

    Examples
    --------
    >>> sub = substitute_params(formulas, {'r': R_TRUE, 'K': K_TRUE})
    >>> print(sub[0])
    """
    sub_map = {}
    for k, v in params.items():
        sym = sp.Symbol(k) if isinstance(k, str) else k
        sub_map[sym] = sp.nsimplify(v, rational=False, tolerance=1e-4)

    return [sp.simplify(expr.subs(sub_map)) for expr in formulas]

def robust_auto_symbolic(
    model,
    *,
    lib,
    a_range=(-10, 10),
    b_range=(-10, 10),
    r2_threshold=0.90,
    weight_simple=0.90,
    topk_edges=64,
    max_total_complexity=None,
    set_others_to_zero=True,
    weight=0.05,
):
    candidates = []

    # --- collect best symbolic candidate per edge ---
    for l in range(len(model.width_in) - 1):
        for i in range(int(model.width_in[l])):
            for j in range(int(model.width_out[l + 1])):
                name, _, r2, c = model.suggest_symbolic(
                    l, i, j,
                    a_range=a_range,
                    b_range=b_range,
                    lib=lib,
                    verbose=False,
                    weight_simple=weight_simple,
                )

                penalty = (weight_simple / (1e-8 + (1.0 - weight_simple + 1e-8))) * c
                score = r2 - weight * penalty

                candidates.append((l, i, j, str(name), float(r2), float(c), float(score)))

    # --- filter + rank ---
    eligible = [d for d in candidates if d[4] >= r2_threshold and d[3] != "0"]
    eligible.sort(key=lambda d: d[6], reverse=True)

    kept = []
    if max_total_complexity is None:
        kept = eligible[:topk_edges]
    else:
        total_c = 0.0
        for d in eligible:
            if len(kept) >= topk_edges:
                break
            if total_c + d[5] <= max_total_complexity:
                kept.append(d)
                total_c += d[5]

    kept_set = {(l, i, j) for (l, i, j, *_ ) in kept}

    # --- apply fixes ---
    for l, i, j, name, *_ in candidates:
        if (l, i, j) in kept_set:
            model.fix_symbolic(l, i, j, name, verbose=False, log_history=False)
        elif set_others_to_zero:
            model.fix_symbolic(l, i, j, "0", verbose=False, log_history=False)

    model.log_history("robust_auto_symbolic")
    return model
