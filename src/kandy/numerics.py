"""kandy.numerics — Finite-volume numerics for 1D conservation laws.

Provides spatial flux schemes and time integrators for the scalar conservation
law on a periodic domain [0, L]:

    u_t + ∂_x F(u) = 0

Intended use: generate training data for KANDy experiments (Burgers, KS, etc.).

Flux schemes
------------
- ``rusanov``  — Local Lax–Friedrichs (most diffusive, always stable)
- ``roe``      — Roe upwind with Harten–Hyman entropy fix (less diffusive)
- ``hllc``     — HLL two-wave solver (HLLC = HLL for scalar laws)

All three schemes use MUSCL reconstruction with a choice of slope limiter to
achieve second-order spatial accuracy away from discontinuities.

Time integrators
----------------
- ``tvdrk2``  — Heun's method (TVD, 2nd order)
- ``tvdrk3``  — Shu–Osher 3-stage scheme (TVD, 3rd order)

Convenience solvers
-------------------
- ``solve_burgers``  — integrate inviscid Burgers u_t + (u²/2)_x = 0
- ``solve_scalar``   — integrate any scalar conservation law

Example
-------
>>> from kandy.numerics import solve_burgers
>>> import numpy as np
>>> x = np.linspace(0, 2*np.pi, 128, endpoint=False)
>>> u0 = np.sin(x)
>>> U = solve_burgers(u0, n_steps=2000, dt=0.005, scheme='roe')
>>> print(U.shape)   # (2000, 128)
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

__all__ = [
    # limiters
    "minmod",
    "van_leer",
    "superbee",
    # MUSCL reconstruction
    "muscl_reconstruct",
    # spectral derivatives
    "spectral_derivative",
    # numerical fluxes
    "rusanov_flux",
    "roe_flux",
    "hllc_flux",
    # RHS builder
    "fv_rhs",
    # time steppers
    "tvdrk2_step",
    "tvdrk3_step",
    # convenience solvers
    "solve_burgers",
    "solve_viscous_burgers",
    "solve_scalar",
    # utilities
    "cfl_dt",
    # built-in flux / speed functions
    "burgers_flux",
    "burgers_speed",
    "burgers_roe_speed",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FluxFn  = Callable[[np.ndarray], np.ndarray]   # F(u) -> array same shape
SpeedFn = Callable[[np.ndarray], np.ndarray]   # max |F'(u)| -> array


# ---------------------------------------------------------------------------
# Built-in flux / speed functions for inviscid Burgers
# ---------------------------------------------------------------------------

def burgers_flux(u: np.ndarray) -> np.ndarray:
    """Physical flux for inviscid Burgers: F(u) = u²/2."""
    return 0.5 * u ** 2


def burgers_speed(u: np.ndarray) -> np.ndarray:
    """Maximum wave speed for Burgers: |F'(u)| = |u|."""
    return np.abs(u)


def burgers_roe_speed(u_L: np.ndarray, u_R: np.ndarray) -> np.ndarray:
    """Roe-averaged wave speed for Burgers: ā = (u_L + u_R) / 2.

    For Burgers F(u) = u²/2 the Roe average is the arithmetic mean, which is
    exact (not approximate).
    """
    return (u_L + u_R) / 2.0


# ---------------------------------------------------------------------------
# Slope limiters
# ---------------------------------------------------------------------------

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter: most compressive of |a|, |b| with matching sign."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def van_leer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """van Leer (harmonic mean) limiter — less diffusive than minmod."""
    ab = a * b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            ab > 0,
            2.0 * ab / (a + b),
            0.0,
        )
    return result


def superbee(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Superbee limiter — least diffusive TVD limiter."""
    s1 = minmod(a, 2.0 * b)
    s2 = minmod(2.0 * a, b)
    return np.where(np.abs(s1) >= np.abs(s2), s1, s2)


_LIMITERS: dict[str, Callable] = {
    "minmod":   minmod,
    "van_leer": van_leer,
    "superbee": superbee,
}


# ---------------------------------------------------------------------------
# MUSCL reconstruction
# ---------------------------------------------------------------------------

def muscl_reconstruct(
    u: np.ndarray,
    dx: float,
    limiter: str = "minmod",
) -> tuple[np.ndarray, np.ndarray]:
    """Second-order MUSCL reconstruction on a uniform periodic grid.

    Returns left and right states (u_L, u_R) at each cell interface i+½.
    The interface at index i lies between cells i and i+1 (periodic wrapping).

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Cell-averaged values.
    dx : float
        Cell width.
    limiter : str
        Slope limiter: ``'minmod'``, ``'van_leer'``, or ``'superbee'``.

    Returns
    -------
    u_L : np.ndarray, shape (N,)
        Left state at interface i+½  (right edge of cell i).
    u_R : np.ndarray, shape (N,)
        Right state at interface i+½  (left edge of cell i+1).
    """
    lim = _LIMITERS[limiter]

    du_fwd = np.roll(u, -1) - u          # u_{i+1} - u_i
    du_bwd = u - np.roll(u, 1)           # u_i - u_{i-1}
    slope  = lim(du_fwd, du_bwd) / dx    # limited slope σ_i  (units: [u]/[x])

    u_L = u + 0.5 * dx * slope           # right edge of cell i
    u_R = np.roll(u - 0.5 * dx * slope, -1)  # left edge of cell i+1

    return u_L, u_R


# ---------------------------------------------------------------------------
# Spectral (Fourier) derivative
# ---------------------------------------------------------------------------

def spectral_derivative(
    u: np.ndarray,
    domain_length: float = 2.0 * np.pi,
    order: int = 1,
    filter_order: int = 0,
    filter_cutoff: float = 0.667,
) -> np.ndarray:
    """Compute the spatial derivative of u on a uniform periodic grid via FFT.

    For a periodic domain [0, L) with N equally spaced points, this computes
    the exact derivative of the trigonometric interpolant:

        d^n u / dx^n  =  IFFT( (i k)^n  FFT(u) )

    where k is the angular wavenumber.  The Nyquist mode is zeroed for
    odd-order derivatives to avoid aliasing artefacts.

    An optional exponential filter suppresses Gibbs oscillations near shocks:

        σ(η) = exp(-α η^p)     where η = |k| / k_max

    with ``p = filter_order`` and α chosen so σ(1) = machine epsilon.
    Set ``filter_order=0`` (default) for no filtering.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Function values on a uniform periodic grid.
    domain_length : float
        Physical domain length L (default 2π).
    order : int
        Derivative order (default 1).
    filter_order : int
        Exponential filter order p (default 0 = no filter).
        Typical values: 8–16 for mild smoothing, 4–6 for strong smoothing.
    filter_cutoff : float
        Fraction of modes to keep unfiltered (default 2/3).
        Modes with |k|/k_max < filter_cutoff are untouched;
        modes above are smoothly damped.

    Returns
    -------
    du : np.ndarray, shape (N,)
        The ``order``-th derivative evaluated at the grid points.
    """
    N = len(u)
    dx = domain_length / N
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)   # angular wavenumber
    u_hat = np.fft.fft(u)
    deriv_hat = u_hat * (1j * k) ** order

    # Zero Nyquist mode for odd-order derivatives (antisymmetric)
    if order % 2 == 1:
        deriv_hat[N // 2] = 0

    # Optional exponential filter
    if filter_order > 0:
        k_max = np.max(np.abs(k))
        eta = np.abs(k) / (k_max + 1e-30)
        # Apply filter only above the cutoff fraction
        alpha = -np.log(np.finfo(float).eps)  # ~36.04
        sigma = np.where(
            eta <= filter_cutoff,
            1.0,
            np.exp(-alpha * ((eta - filter_cutoff) / (1.0 - filter_cutoff)) ** filter_order),
        )
        deriv_hat *= sigma

    return np.real(np.fft.ifft(deriv_hat))


# ---------------------------------------------------------------------------
# Numerical flux schemes
# ---------------------------------------------------------------------------

def rusanov_flux(
    u_L: np.ndarray,
    u_R: np.ndarray,
    flux_fn: FluxFn,
    speed_fn: SpeedFn,
) -> np.ndarray:
    """Rusanov (Local Lax–Friedrichs) numerical flux at each interface.

    F̂_{i+½} = ½(F(u_L) + F(u_R)) − ½ α (u_R − u_L)

    where α = max(speed_fn(u_L), speed_fn(u_R)) is the local wave speed bound.

    Parameters
    ----------
    u_L, u_R : np.ndarray, shape (N,)
        Reconstructed left/right states at each interface.
    flux_fn : callable
        Physical flux  F(u).
    speed_fn : callable
        Local wave-speed bound  |F'(u)| for a single state.

    Returns
    -------
    F_hat : np.ndarray, shape (N,)
        Numerical flux at each interface i+½.
    """
    F_L = flux_fn(u_L)
    F_R = flux_fn(u_R)
    alpha = np.maximum(speed_fn(u_L), speed_fn(u_R))
    return 0.5 * (F_L + F_R) - 0.5 * alpha * (u_R - u_L)


def roe_flux(
    u_L: np.ndarray,
    u_R: np.ndarray,
    flux_fn: FluxFn,
    roe_speed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    entropy_fix: bool = True,
    entropy_eps: float = 0.1,
) -> np.ndarray:
    """Roe upwind numerical flux with optional Harten–Hyman entropy fix.

    F̂_{i+½} = ½(F(u_L) + F(u_R)) − ½ |ā| (u_R − u_L)

    where ā = roe_speed_fn(u_L, u_R) is the Roe-averaged wave speed.

    The entropy fix prevents unphysical expansion shocks at sonic points
    (where the characteristic speed changes sign) by replacing |ā| with a
    smooth positive function when |ā| < ε.

    Parameters
    ----------
    u_L, u_R : np.ndarray, shape (N,)
        Reconstructed left/right states at each interface.
    flux_fn : callable
        Physical flux  F(u).
    roe_speed_fn : callable
        Roe-averaged wave speed  ā(u_L, u_R).
    entropy_fix : bool
        Apply Harten–Hyman entropy fix (default True).
    entropy_eps : float
        Width of the entropy fix region in wave-speed space.

    Returns
    -------
    F_hat : np.ndarray, shape (N,)
        Numerical flux at each interface i+½.
    """
    F_L = flux_fn(u_L)
    F_R = flux_fn(u_R)
    a   = roe_speed_fn(u_L, u_R)

    if entropy_fix:
        # Harten–Hyman: replace |a| with a smoothed absolute value near a=0
        abs_a = np.where(
            np.abs(a) >= entropy_eps,
            np.abs(a),
            0.5 * (a ** 2 / entropy_eps + entropy_eps),
        )
    else:
        abs_a = np.abs(a)

    return 0.5 * (F_L + F_R) - 0.5 * abs_a * (u_R - u_L)


def hllc_flux(
    u_L: np.ndarray,
    u_R: np.ndarray,
    flux_fn: FluxFn,
    speed_fn: SpeedFn,
) -> np.ndarray:
    """HLL (Harten–Lax–van Leer) two-wave numerical flux.

    For scalar conservation laws the contact wave degenerates, so HLLC
    reduces exactly to the two-wave HLL solver.

    Wave-speed estimates:
        S_L = min(F'(u_L), F'(u_R))   (leftmost wave)
        S_R = max(F'(u_L), F'(u_R))   (rightmost wave)

    where F'(u) is approximated by ``speed_fn`` with sign information
    recovered from the state (F' = u for Burgers).

    Parameters
    ----------
    u_L, u_R : np.ndarray, shape (N,)
        Reconstructed left/right states at each interface.
    flux_fn : callable
        Physical flux  F(u).
    speed_fn : callable
        |F'(u)| for a single state.  Sign is inferred from the state sign
        (appropriate for convex fluxes where F''(u) > 0, e.g. Burgers).

    Returns
    -------
    F_hat : np.ndarray, shape (N,)
        Numerical flux at each interface i+½.

    Notes
    -----
    For non-convex fluxes supply a ``speed_fn`` that returns signed speeds,
    or construct wave speed estimates externally and call the HLL formula
    directly.
    """
    F_L = flux_fn(u_L)
    F_R = flux_fn(u_R)

    # Signed wave speeds: for convex F, characteristic speed = F'(u);
    # for Burgers F'(u) = u.  We recover sign from the state directly.
    # For a general flux the caller should override with a signed speed_fn.
    a_L = np.sign(u_L) * speed_fn(u_L)   # signed F'(u_L)
    a_R = np.sign(u_R) * speed_fn(u_R)   # signed F'(u_R)

    S_L = np.minimum(a_L, a_R)
    S_R = np.maximum(a_L, a_R)

    # HLL flux (three regions)
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (u_R - u_L)) / (S_R - S_L + 1e-14)

    F_hat = np.where(S_L >= 0, F_L,
            np.where(S_R <= 0, F_R,
                     F_hll))
    return F_hat


# ---------------------------------------------------------------------------
# RHS builder
# ---------------------------------------------------------------------------

def fv_rhs(
    u: np.ndarray,
    dx: float,
    flux_fn: FluxFn,
    speed_fn: SpeedFn,
    roe_speed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    scheme: Literal["rusanov", "roe", "hllc"] = "rusanov",
    limiter: str = "minmod",
) -> np.ndarray:
    """Compute the finite-volume RHS  −(F̂_{i+½} − F̂_{i−½}) / Δx.

    Combines MUSCL reconstruction and a numerical flux scheme to give the
    semi-discrete update for u_t = RHS(u).

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Current cell-averaged state.
    dx : float
        Uniform cell width.
    flux_fn : callable
        Physical flux F(u).
    speed_fn : callable
        Wave-speed bound |F'(u)|.
    roe_speed_fn : callable, optional
        Roe-averaged speed ā(u_L, u_R).  Required when ``scheme='roe'``.
    scheme : str
        ``'rusanov'``, ``'roe'``, or ``'hllc'``.
    limiter : str
        Slope limiter for MUSCL reconstruction.

    Returns
    -------
    rhs : np.ndarray, shape (N,)
        Semi-discrete RHS (same units as u_t).
    """
    u_L, u_R = muscl_reconstruct(u, dx, limiter=limiter)

    if scheme == "rusanov":
        F_face = rusanov_flux(u_L, u_R, flux_fn, speed_fn)
    elif scheme == "roe":
        if roe_speed_fn is None:
            raise ValueError("roe_speed_fn is required for scheme='roe'.")
        F_face = roe_flux(u_L, u_R, flux_fn, roe_speed_fn)
    elif scheme == "hllc":
        F_face = hllc_flux(u_L, u_R, flux_fn, speed_fn)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose 'rusanov', 'roe', or 'hllc'.")

    # Divergence: ∂_x F ≈ (F_{i+½} - F_{i-½}) / dx
    return -(F_face - np.roll(F_face, 1)) / dx


# ---------------------------------------------------------------------------
# Time integrators
# ---------------------------------------------------------------------------

def tvdrk2_step(
    u: np.ndarray,
    rhs: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """One step of TVD Runge–Kutta 2 (Heun / SSP-RK2).

    u* = u + dt * L(u)
    u^{n+1} = ½ u + ½ (u* + dt * L(u*))

    Second-order accurate, TVD under the CFL condition dt ≤ Δx / max|F'|.
    """
    L1 = rhs(u)
    u1 = u + dt * L1
    return 0.5 * (u + u1 + dt * rhs(u1))


def tvdrk3_step(
    u: np.ndarray,
    rhs: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """One step of TVD Runge–Kutta 3 (Shu–Osher / SSP-RK3).

    u^(1) = u + dt * L(u)
    u^(2) = ¾ u + ¼ (u^(1) + dt * L(u^(1)))
    u^{n+1} = ⅓ u + ⅔ (u^(2) + dt * L(u^(2)))

    Third-order accurate, TVD under dt ≤ Δx / max|F'|.
    """
    L1 = rhs(u)
    u1 = u + dt * L1
    L2 = rhs(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L2)
    L3 = rhs(u2)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L3)


_STEPPERS: dict[str, Callable] = {
    "tvdrk2": tvdrk2_step,
    "tvdrk3": tvdrk3_step,
}


# ---------------------------------------------------------------------------
# Convenience solvers
# ---------------------------------------------------------------------------

def solve_scalar(
    u0: np.ndarray,
    dx: float,
    n_steps: int,
    dt: float,
    flux_fn: FluxFn,
    speed_fn: SpeedFn,
    roe_speed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    scheme: Literal["rusanov", "roe", "hllc"] = "rusanov",
    limiter: str = "minmod",
    time_stepper: Literal["tvdrk2", "tvdrk3"] = "tvdrk2",
    save_every: int = 1,
) -> np.ndarray:
    """Integrate a scalar conservation law u_t + ∂_x F(u) = 0.

    Parameters
    ----------
    u0 : np.ndarray, shape (N,)
        Initial condition (cell averages).
    dx : float
        Uniform cell width.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    flux_fn : callable
        Physical flux F(u).
    speed_fn : callable
        Wave-speed bound |F'(u)|.
    roe_speed_fn : callable, optional
        Required for scheme='roe'.
    scheme : str
        Flux scheme: ``'rusanov'``, ``'roe'``, or ``'hllc'``.
    limiter : str
        Slope limiter: ``'minmod'``, ``'van_leer'``, or ``'superbee'``.
    time_stepper : str
        Time integration: ``'tvdrk2'`` or ``'tvdrk3'``.
    save_every : int
        Save a snapshot every ``save_every`` steps (default 1 = every step).

    Returns
    -------
    U : np.ndarray, shape (n_saved, N)
        Trajectory snapshots.
    """
    stepper = _STEPPERS[time_stepper]

    def rhs(u):
        return fv_rhs(u, dx, flux_fn, speed_fn,
                      roe_speed_fn=roe_speed_fn,
                      scheme=scheme, limiter=limiter)

    u = u0.copy().astype(np.float64)
    snapshots = [u.copy()]

    for step in range(1, n_steps):
        u = stepper(u, rhs, dt)
        if step % save_every == 0:
            snapshots.append(u.copy())

    return np.array(snapshots)


def solve_burgers(
    u0: np.ndarray,
    n_steps: int,
    dt: float,
    domain_length: float | None = None,
    scheme: Literal["rusanov", "roe", "hllc"] = "rusanov",
    limiter: str = "minmod",
    time_stepper: Literal["tvdrk2", "tvdrk3"] = "tvdrk2",
    save_every: int = 1,
) -> np.ndarray:
    """Integrate the inviscid Burgers equation u_t + (u²/2)_x = 0.

    A convenience wrapper around :func:`solve_scalar` with the Burgers
    flux and wave-speed functions pre-configured.

    Parameters
    ----------
    u0 : np.ndarray, shape (N,)
        Initial condition on a uniform periodic grid.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.  A stable CFL for Burgers is dt ≤ dx / max|u|.
    domain_length : float, optional
        Physical domain length L.  Inferred as ``N * dx`` where
        ``dx = L / N``.  Defaults to ``2π`` (standard Burgers domain).
    scheme : str
        Flux scheme: ``'rusanov'`` (default), ``'roe'``, or ``'hllc'``.
    limiter : str
        Slope limiter for MUSCL: ``'minmod'``, ``'van_leer'``, ``'superbee'``.
    time_stepper : str
        ``'tvdrk2'`` (default) or ``'tvdrk3'``.
    save_every : int
        Snapshot frequency (default 1 = every step).

    Returns
    -------
    U : np.ndarray, shape (n_saved, N)
        Trajectory snapshots (including initial condition as row 0).

    Examples
    --------
    >>> import numpy as np
    >>> from kandy.numerics import solve_burgers
    >>> x = np.linspace(0, 2*np.pi, 128, endpoint=False)
    >>> u0 = np.sin(x)
    >>> U_rus = solve_burgers(u0, n_steps=2000, dt=0.005, scheme='rusanov')
    >>> U_roe = solve_burgers(u0, n_steps=2000, dt=0.005, scheme='roe')
    >>> U_hll = solve_burgers(u0, n_steps=2000, dt=0.005, scheme='hllc')
    """
    N = len(u0)
    if domain_length is None:
        domain_length = 2.0 * np.pi
    dx = domain_length / N

    return solve_scalar(
        u0=u0,
        dx=dx,
        n_steps=n_steps,
        dt=dt,
        flux_fn=burgers_flux,
        speed_fn=burgers_speed,
        roe_speed_fn=burgers_roe_speed,
        scheme=scheme,
        limiter=limiter,
        time_stepper=time_stepper,
        save_every=save_every,
    )


# ---------------------------------------------------------------------------
# CFL utility
# ---------------------------------------------------------------------------

def cfl_dt(
    u: np.ndarray,
    dx: float,
    cfl: float = 0.8,
    min_speed: float = 1e-8,
) -> float:
    """Compute a stable time step from the CFL condition.

    dt = cfl * dx / max|u|

    Parameters
    ----------
    u : np.ndarray
        Current state (any shape; the global maximum is used).
    dx : float
        Uniform cell width.
    cfl : float
        CFL number (default 0.8; use ≤ 0.5 for 2nd-order schemes).
    min_speed : float
        Minimum wave speed to avoid division by zero (default 1e-8).

    Returns
    -------
    dt : float
        Stable time step.
    """
    max_speed = max(float(np.max(np.abs(u))), min_speed)
    return cfl * dx / max_speed


# ---------------------------------------------------------------------------
# Viscous Burgers solver  (IMEX: explicit convection, implicit diffusion)
# ---------------------------------------------------------------------------

def solve_viscous_burgers(
    u0: np.ndarray,
    n_steps: int,
    dt: float,
    nu: float,
    domain_length: float | None = None,
    scheme: Literal["rusanov", "roe", "hllc"] = "rusanov",
    limiter: str = "minmod",
    time_stepper: Literal["tvdrk2", "tvdrk3"] = "tvdrk2",
    save_every: int = 1,
) -> np.ndarray:
    """Integrate the viscous Burgers equation u_t + (u²/2)_x = ν u_xx.

    Uses an IMEX splitting:
    - **Explicit** (TVD-RK): nonlinear convection  −∂_x(u²/2)
    - **Implicit** (spectral): linear diffusion  ν ∂_xx u

    The implicit step is exact in Fourier space:
        û^{n+1} = û^* / (1 + ν dt k²)

    where û^* is the post-convection state from the explicit stage.
    This allows arbitrarily large ν without stiffness constraints on dt
    (the CFL condition applies only to the convective term).

    Parameters
    ----------
    u0 : np.ndarray, shape (N,)
        Initial condition on a uniform periodic grid.
    n_steps : int
        Number of time steps.
    dt : float
        Time step.  Must satisfy CFL for the convective term:
        dt ≤ dx / max|u|  (use :func:`cfl_dt` to compute).
    nu : float
        Kinematic viscosity (ν ≥ 0).
    domain_length : float, optional
        Physical domain length.  Defaults to 2π.
    scheme : str
        Convective flux scheme: ``'rusanov'``, ``'roe'``, or ``'hllc'``.
    limiter : str
        MUSCL slope limiter: ``'minmod'``, ``'van_leer'``, ``'superbee'``.
    time_stepper : str
        Explicit time integrator: ``'tvdrk2'`` or ``'tvdrk3'``.
    save_every : int
        Snapshot frequency (default 1).

    Returns
    -------
    U : np.ndarray, shape (n_saved, N)
        Trajectory snapshots.

    Examples
    --------
    >>> import numpy as np
    >>> from kandy.numerics import solve_viscous_burgers, cfl_dt
    >>> N = 256; L = 2 * np.pi
    >>> x = np.linspace(0, L, N, endpoint=False)
    >>> u0 = np.sin(x)
    >>> dx = L / N
    >>> dt = cfl_dt(u0, dx, cfl=0.4)
    >>> U = solve_viscous_burgers(u0, n_steps=500, dt=dt, nu=0.01)
    """
    N = len(u0)
    if domain_length is None:
        domain_length = 2.0 * np.pi
    dx = domain_length / N

    # Pre-compute wavenumbers and diffusion factor for spectral implicit step
    k = np.fft.rfftfreq(N, d=domain_length / (2.0 * np.pi * N))  # k_j = 2π j / L
    diffusion_factor = 1.0 / (1.0 + nu * dt * k ** 2)             # (1 + ν dt k²)^{-1}

    stepper = _STEPPERS[time_stepper]

    def convective_rhs(u):
        return fv_rhs(u, dx, burgers_flux, burgers_speed,
                      roe_speed_fn=burgers_roe_speed,
                      scheme=scheme, limiter=limiter)

    def imex_step(u: np.ndarray) -> np.ndarray:
        """One IMEX step: explicit convection then implicit diffusion."""
        u_star = stepper(u, convective_rhs, dt)          # explicit RK stage
        u_hat  = np.fft.rfft(u_star)                     # to Fourier space
        u_hat *= diffusion_factor                          # implicit solve
        return np.fft.irfft(u_hat, n=N)                  # back to physical

    u = u0.copy().astype(np.float64)
    snapshots = [u.copy()]

    for step in range(1, n_steps):
        u = imex_step(u)
        if step % save_every == 0:
            snapshots.append(u.copy())

    return np.array(snapshots)
