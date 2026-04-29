"""kandy — KANDy: Kolmogorov-Arnold Networks for Dynamics.

Algorithm:  x_dot = A * Psi(phi(x))
  phi  — Koopman lift (encodes all cross-terms)
  Psi  — separable spline map (single-layer KAN, width=[m, n])
  A    — linear mixing matrix from KAN output weights

Quick start
-----------
>>> from kandy import KANDy, PolynomialLift
>>> lift = PolynomialLift(degree=2)
>>> model = KANDy(lift=lift, grid=5, k=3)
>>> model.fit(X, X_dot)
>>> formulas = model.get_formula()
>>> traj = model.rollout(x0, T=500, dt=0.01)
"""
from .core import KANDy
from .lifts import (
    CustomLift,
    DMDLift,
    FourierLift,
    Lift,
    PolynomialLift,
    RadialBasisLift,
)
from .numerics import (
    burgers_flux,
    burgers_speed,
    burgers_roe_speed,
    cfl_dt,
    minmod,
    van_leer,
    superbee,
    muscl_reconstruct,
    spectral_derivative,
    rusanov_flux,
    roe_flux,
    hllc_flux,
    fv_rhs,
    tvdrk2_step,
    tvdrk3_step,
    solve_burgers,
    solve_viscous_burgers,
    solve_scalar,
)
from .training import (
    euler_step,
    fit_kan,
    integrate_trajectory,
    make_windows,
    rk4_integrate_numpy,
    rk4_step,
    wrap_pi_torch,
    angle_mse,
    order_param_torch,
)
from .symbolic import (
    make_symbolic_lib,
    POLY_LIB,
    POLY_LIB_CHEAP,
    TRIG_LIB,
    TRIG_LIB_CHEAP,
    auto_symbolic_with_costs,
    robust_auto_symbolic,
    score_formula,
    formulas_to_latex,
    substitute_params,
)
from .plotting import (
    get_edge_activation,
    get_all_edge_activations,
    plot_edge,
    plot_all_edges,
    plot_kan_architecture,
    plot_loss_curves,
    plot_attractor_overlay,
    plot_trajectory_error,
    fit_linear,
    fit_polynomial,
    fit_sine,
    use_pub_style,
)

__all__ = [
    # model
    "KANDy",
    # lifts
    "Lift",
    "PolynomialLift",
    "CustomLift",
    "FourierLift",
    "RadialBasisLift",
    "DMDLift",
    # numerics
    "burgers_flux",
    "burgers_speed",
    "burgers_roe_speed",
    "cfl_dt",
    "minmod",
    "van_leer",
    "superbee",
    "muscl_reconstruct",
    "spectral_derivative",
    "rusanov_flux",
    "roe_flux",
    "hllc_flux",
    "fv_rhs",
    "tvdrk2_step",
    "tvdrk3_step",
    "solve_burgers",
    "solve_viscous_burgers",
    "solve_scalar",
    # training
    "fit_kan",
    "rk4_step",
    "euler_step",
    "integrate_trajectory",
    "make_windows",
    "rk4_integrate_numpy",
    "wrap_pi_torch",
    "angle_mse",
    "order_param_torch",
    # symbolic
    "make_symbolic_lib",
    "POLY_LIB",
    "POLY_LIB_CHEAP",
    "TRIG_LIB",
    "TRIG_LIB_CHEAP",
    "auto_symbolic_with_costs",
    "score_formula",
    "formulas_to_latex",
    "substitute_params",
    # plotting
    "get_edge_activation",
    "get_all_edge_activations",
    "plot_edge",
    "plot_all_edges",
    "plot_kan_architecture",
    "plot_loss_curves",
    "plot_attractor_overlay",
    "plot_trajectory_error",
    "fit_linear",
    "fit_polynomial",
    "fit_sine",
    "use_pub_style",
]
