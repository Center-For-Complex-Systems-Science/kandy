"""kandy.lifts — Koopman lift functions phi: R^n -> R^m.

The Koopman lift is the most critical design choice in KANDy. It must encode
ALL cross-interaction terms that appear in the target system's RHS. Missing
cross-terms make the algorithm structurally incorrect, not merely inaccurate.

Example for Lorenz (sigma*(y-x), rho*x - x*z - y, x*y - beta*z):
    x*y appears in z_dot  →  must be in phi
    x*z appears in y_dot  →  must be in phi
    Correct: PolynomialLift(degree=2) on R^3 gives the full 9-feature lift.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from itertools import combinations_with_replacement
from typing import Optional

import numpy as np


class Lift(ABC):
    """Abstract base class for Koopman lifts phi: R^n -> R^m.

    All lifts must be callable (apply the lift to state data) and expose
    ``output_dim`` after fitting.  A ``fit`` method is provided for lifts
    that need to inspect the input dimension before transforming data.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply lift to X of shape (N, n) or (n,).  Returns (N, m) or (m,)."""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of lifted feature space m."""
        ...

    def fit(self, X: np.ndarray) -> "Lift":
        """Fit any internal parameters from X (shape (N, n) or (n,)).

        Default implementation is a no-op.  Override when the lift needs to
        determine feature indices or normalisation statistics from data.
        """
        return self

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names for each lifted feature (optional)."""
        return [f"phi_{i}" for i in range(self.output_dim)]


class PolynomialLift(Lift):
    """Full polynomial Koopman lift up to a given degree.

    For degree=2 and n-dimensional input, the features are every monomial
    x_0^a0 * x_1^a1 * ... with a0+a1+...<=degree (and >=1 if include_bias=False).
    Cross-products such as x_0*x_1 are explicitly included — this is mandatory
    for KANDy to represent bilinear dynamics like x*y in the Lorenz system.

    Parameters
    ----------
    degree : int
        Maximum monomial degree (default 2).
    include_bias : bool
        Whether to include the constant feature 1 (default False).

    Example
    -------
    >>> lift = PolynomialLift(degree=2)
    >>> lift.fit(np.zeros((10, 3)))  # 3D input
    >>> lift.output_dim
    9
    >>> lift.feature_names
    ['x_0', 'x_1', 'x_2', 'x_0^2', 'x_0*x_1', 'x_0*x_2', 'x_1^2', 'x_1*x_2', 'x_2^2']
    """

    def __init__(self, degree: int = 2, include_bias: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self._input_dim: Optional[int] = None
        self._feature_combos: Optional[list[tuple[int, ...]]] = None

    def fit(self, X: np.ndarray) -> "PolynomialLift":
        """Determine input dimension and build feature index table."""
        n = X.shape[-1] if X.ndim > 1 else 1
        self._input_dim = n
        combos: list[tuple[int, ...]] = []
        if self.include_bias:
            combos.append(())
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n), d):
                combos.append(combo)
        self._feature_combos = combos
        return self

    @property
    def output_dim(self) -> int:
        if self._feature_combos is None:
            raise RuntimeError("Call lift.fit(X) before accessing output_dim.")
        return len(self._feature_combos)

    @property
    def feature_names(self) -> list[str]:
        if self._feature_combos is None:
            raise RuntimeError("Call lift.fit(X) before accessing feature_names.")
        names = []
        for combo in self._feature_combos:
            if len(combo) == 0:
                names.append("1")
            else:
                c = Counter(combo)
                parts = []
                for idx, power in sorted(c.items()):
                    parts.append(f"x_{idx}" if power == 1 else f"x_{idx}^{power}")
                names.append("*".join(parts))
        return names

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply polynomial lift.

        Parameters
        ----------
        X : np.ndarray, shape (N, n) or (n,)

        Returns
        -------
        theta : np.ndarray, shape (N, m) or (m,)
        """
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        if self._feature_combos is None:
            self.fit(X)
        cols = []
        for combo in self._feature_combos:
            if len(combo) == 0:
                cols.append(np.ones(X.shape[0], dtype=X.dtype))
            else:
                col = np.ones(X.shape[0], dtype=X.dtype)
                for idx in combo:
                    col = col * X[:, idx]
                cols.append(col)
        result = np.column_stack(cols)
        return result[0] if scalar else result


class CustomLift(Lift):
    """Wrap hand-crafted feature functions as a KANDy-compatible lift.

    This is the **primary lift type** for physics-informed KANDy experiments.
    The core advantage of KANDy over SINDy is that the feature library is not
    restricted to polynomials: any precomputed feature — rational functions,
    trigonometric composites, state-dependent denominators — can be included.

    The canonical example is the Ikeda optical-cavity map, where SINDy fails
    but KANDy succeeds because the feature ``q = 1 / (1 + x² + y²)`` is
    precomputed and handed to the KAN.  A polynomial library cannot represent
    ``q``; a neural ODE would learn it opaquely.  KANDy precomputes it
    explicitly and then lets the single-layer KAN learn a separable function
    of it — which turns out to be nearly linear, giving a readable formula.

    Two function signatures are required for a complete lift:

    * ``fn(X: np.ndarray) -> np.ndarray`` — **NumPy version** used by
      ``KANDy.fit()`` to build the training dataset (fast, no gradients).
    * ``torch_fn(X: torch.Tensor) -> torch.Tensor`` — **Torch version** used
      during rollout training (``fit_kan`` with ``rollout_weight > 0``) where
      gradients must flow back through the feature computation.

    If ``torch_fn`` is omitted, rollout-loss training will still work as long
    as ``dynamics_fn`` is provided externally — but attaching ``torch_fn``
    directly to the lift keeps the feature logic in one place.

    Parameters
    ----------
    fn : callable
        NumPy feature function: ``fn(X: ndarray (N, n)) -> ndarray (N, m)``.
    output_dim : int
        Output dimension m.  Must be stated explicitly.
    torch_fn : callable, optional
        PyTorch feature function: ``torch_fn(X: Tensor (B, n)) -> Tensor (B, m)``.
        Required for gradient-compatible rollout training.
    name : str, optional
        Human-readable name shown in repr (default ``'custom'``).

    Attributes
    ----------
    torch_fn : callable or None
        The torch feature function, accessible as ``lift.torch_fn``.  Use
        this inside ``dynamics_fn`` to keep feature logic co-located with
        the lift definition.

    Examples
    --------
    Lorenz system — cross-products are all the lift needs:

    >>> def lorenz_phi_np(X):
    ...     x, y, z = X[:,0], X[:,1], X[:,2]
    ...     return np.column_stack([x, y, z, x*y, x*z, y*z])
    ...
    >>> def lorenz_phi_torch(X):
    ...     x, y, z = X[:,0], X[:,1], X[:,2]
    ...     return torch.stack([x, y, z, x*y, x*z, y*z], dim=1)
    ...
    >>> lift = CustomLift(lorenz_phi_np, output_dim=6, torch_fn=lorenz_phi_torch)

    Ikeda map — rational and trig features that SINDy cannot represent:

    >>> def ikeda_np(X):
    ...     x, y = X[:,0], X[:,1]
    ...     q = 1.0 / (1.0 + x**2 + y**2)   # <-- SINDy cannot fit this
    ...     t = 0.4 - 6.0 * q
    ...     return np.column_stack([0.9*x*np.cos(t), 0.9*y*np.cos(t),
    ...                             0.9*x*np.sin(t), 0.9*y*np.sin(t)])
    ...
    >>> def ikeda_torch(X):
    ...     x, y = X[:,0], X[:,1]
    ...     q = 1.0 / (1.0 + x**2 + y**2)
    ...     t = 0.4 - 6.0 * q
    ...     return torch.stack([0.9*x*torch.cos(t), 0.9*y*torch.cos(t),
    ...                         0.9*x*torch.sin(t), 0.9*y*torch.sin(t)], dim=1)
    ...
    >>> lift = CustomLift(ikeda_np, output_dim=4, torch_fn=ikeda_torch, name="ikeda")
    >>> # Use lift.torch_fn inside dynamics_fn for rollout training:
    >>> def discrete_rhs(state):
    ...     theta_n = (lift.torch_fn(state) - feat_mean_t) / feat_std_t
    ...     return model.model_(theta_n) - state
    """

    def __init__(
        self,
        fn: callable,
        output_dim: int,
        torch_fn: Optional[callable] = None,
        name: str = "custom",
    ):
        self._fn = fn
        self._output_dim = output_dim
        self.torch_fn = torch_fn
        self.name = name

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def __call__(self, X: np.ndarray) -> np.ndarray:
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        result = self._fn(X)
        return result[0] if scalar else result

    def __repr__(self) -> str:
        has_torch = self.torch_fn is not None
        return (
            f"CustomLift(name={self.name!r}, output_dim={self._output_dim}, "
            f"torch_fn={'yes' if has_torch else 'no'})"
        )


class FourierLift(Lift):
    """Lift a periodic spatial field to its leading Fourier coefficients.

    Designed for PDE systems (KS, Burgers, NS) where each snapshot is a
    spatial field u ∈ R^{N_x} on a uniform periodic grid.  The lift extracts
    the DC component and the real/imaginary parts of the first ``n_modes``
    non-trivial Fourier modes.

    Output dimension: ``1 + 2 * n_modes`` (DC + cos + sin coefficients).

    Parameters
    ----------
    n_modes : int
        Number of non-trivial Fourier modes to retain (i.e. wavenumbers
        k = 1, 2, ..., n_modes).  Output dim = 1 + 2*n_modes.

    Examples
    --------
    >>> lift = FourierLift(n_modes=8)
    >>> u = np.random.randn(500, 64)   # 500 snapshots on 64-point grid
    >>> theta = lift(u)
    >>> theta.shape
    (500, 17)
    """

    def __init__(self, n_modes: int):
        self.n_modes = n_modes

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.n_modes

    @property
    def feature_names(self) -> list[str]:
        names = ["DC"]
        for k in range(1, self.n_modes + 1):
            names += [f"Re(û_{k})", f"Im(û_{k})"]
        return names

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply Fourier lift.

        Parameters
        ----------
        X : np.ndarray, shape (N, N_x) or (N_x,)
            Spatial field snapshots.  Each row is one snapshot.

        Returns
        -------
        theta : np.ndarray, shape (N, 1 + 2*n_modes) or (1 + 2*n_modes,)
        """
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        N, Nx = X.shape
        # Full complex FFT; normalise by Nx to get amplitudes
        U_hat = np.fft.rfft(X, axis=1) / Nx   # (N, Nx//2 + 1)
        dc = U_hat[:, 0].real[:, None]          # (N, 1)  — always real
        n = min(self.n_modes, U_hat.shape[1] - 1)
        modes = U_hat[:, 1 : n + 1]             # (N, n)  — complex
        cols = [dc]
        for k in range(n):
            cols.append(modes[:, k].real[:, None])
            cols.append(modes[:, k].imag[:, None])
        # Pad with zeros if n_modes > available modes
        for _ in range(self.n_modes - n):
            cols += [np.zeros((N, 1)), np.zeros((N, 1))]
        result = np.hstack(cols).astype(np.float64)
        return result[0] if scalar else result

    def __repr__(self) -> str:
        return f"FourierLift(n_modes={self.n_modes})"


class RadialBasisLift(Lift):
    """Radial basis function (RBF) Koopman lift.

    Computes Gaussian RBF features:

        phi_i(x) = exp( -||x - c_i||^2 / (2 * sigma^2) )

    where the centres c_i are selected from training data.  This provides
    a smooth, data-adaptive dictionary that approximates Koopman
    eigenfunctions for systems with compact attractors.

    Parameters
    ----------
    n_centers : int
        Number of RBF centres (= output dimension).
    sigma : float or None
        Bandwidth.  If ``None`` (default), auto-selected as the median
        pairwise distance between the chosen centres divided by
        ``sqrt(2 * n_centers)``.
    center_method : str
        How to place centres: ``'random'`` (subsample training data,
        default) or ``'kmeans'`` (k-means clustering via scipy).

    Examples
    --------
    >>> lift = RadialBasisLift(n_centers=50)
    >>> lift.fit(X_train)
    >>> theta = lift(X_test)
    >>> theta.shape[1]
    50
    """

    def __init__(
        self,
        n_centers: int,
        sigma: Optional[float] = None,
        center_method: str = "random",
    ):
        self.n_centers = n_centers
        self.sigma = sigma
        self.center_method = center_method
        self._centers: Optional[np.ndarray] = None
        self._sigma_fit: Optional[float] = None

    def fit(self, X: np.ndarray) -> "RadialBasisLift":
        """Set RBF centres from training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, n)
        """
        if X.ndim == 1:
            X = X[:, None]
        n_c = min(self.n_centers, len(X))

        if self.center_method == "kmeans":
            from scipy.cluster.vq import kmeans
            centres, _ = kmeans(X.astype(np.float64), n_c, seed=0)
        else:  # random
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), size=n_c, replace=False)
            centres = X[idx].astype(np.float64)

        self._centers = centres   # (n_c, n)

        if self.sigma is not None:
            self._sigma_fit = float(self.sigma)
        else:
            # Median pairwise distance heuristic
            diffs = centres[:, None, :] - centres[None, :, :]   # (n_c, n_c, n)
            dists = np.sqrt((diffs ** 2).sum(axis=-1))           # (n_c, n_c)
            upper = dists[np.triu_indices(n_c, k=1)]
            median_dist = float(np.median(upper)) if len(upper) > 0 else 1.0
            self._sigma_fit = max(median_dist / np.sqrt(2.0 * n_c), 1e-6)

        return self

    @property
    def output_dim(self) -> int:
        return self.n_centers

    @property
    def feature_names(self) -> list[str]:
        return [f"rbf_{i}" for i in range(self.n_centers)]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate RBF features.

        Parameters
        ----------
        X : np.ndarray, shape (N, n) or (n,)

        Returns
        -------
        theta : np.ndarray, shape (N, n_centers) or (n_centers,)
        """
        if self._centers is None:
            self.fit(X)
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        # Squared distances: (N, n_c)
        diff = X[:, None, :] - self._centers[None, :, :]    # (N, n_c, d)
        sq_dist = (diff ** 2).sum(axis=-1)                   # (N, n_c)
        result = np.exp(-sq_dist / (2.0 * self._sigma_fit ** 2))
        return result[0] if scalar else result

    def __repr__(self) -> str:
        sigma_str = f"{self._sigma_fit:.4g}" if self._sigma_fit is not None else "auto"
        return (
            f"RadialBasisLift(n_centers={self.n_centers}, "
            f"sigma={sigma_str}, method={self.center_method!r})"
        )


class DMDLift(Lift):
    """Data-driven lift from Extended Dynamic Mode Decomposition (EDMD).

    Computes the leading Koopman eigenfunctions as linear combinations of
    a user-supplied observable dictionary.  Given consecutive state pairs
    (x_k, x_{k+1}) from training data, the EDMD Koopman matrix

        K = argmin ||Psi(X') - K @ Psi(X)||_F

    is estimated and its leading eigenvectors are used as the lift.

    For complex-conjugate eigenvector pairs, both the real and imaginary
    parts are returned as separate features, so output dimension is always
    real-valued.

    Parameters
    ----------
    n_modes : int
        Number of Koopman modes to retain.  Output dimension is at most
        ``2 * n_modes`` (real + imaginary parts of complex eigenvectors),
        or exactly ``n_modes`` if all retained eigenvalues are real.
    dictionary : Lift, optional
        Observable basis applied before computing K.
        Defaults to ``PolynomialLift(degree=2)`` (all monomials up to
        degree 2, including cross-terms).
    sort_by : str
        Eigenvalue selection criterion: ``'magnitude'`` (default — most
        persistent modes for discrete-time), ``'real'`` (largest real part,
        suitable for continuous-time / growth rate selection).

    Notes
    -----
    ``fit`` expects consecutive pairs, so X must be a trajectory array of
    shape (T, n) with T ≥ 2.  The EDMD matrix is estimated from
    (X[:-1], X[1:]) internally.

    Examples
    --------
    >>> from kandy.lifts import DMDLift, PolynomialLift
    >>> lift = DMDLift(n_modes=10, dictionary=PolynomialLift(degree=2))
    >>> lift.fit(X_trajectory)      # (T, n) trajectory
    >>> theta = lift(X_new)         # (N, ≤20) real Koopman features
    """

    def __init__(
        self,
        n_modes: int,
        dictionary: Optional["Lift"] = None,
        sort_by: str = "magnitude",
    ):
        self.n_modes = n_modes
        self.sort_by = sort_by
        self._dictionary = dictionary   # resolved at fit time if None
        self._evecs: Optional[np.ndarray] = None   # (m, n_kept) complex
        self._evals: Optional[np.ndarray] = None   # (n_kept,) complex
        self._output_dim: Optional[int] = None
        self._is_real: Optional[np.ndarray] = None  # (n_kept,) bool

    def fit(self, X: np.ndarray) -> "DMDLift":
        """Compute EDMD Koopman matrix and extract leading eigenfunctions.

        Parameters
        ----------
        X : np.ndarray, shape (T, n)
            State trajectory.  Consecutive pairs (X[:-1], X[1:]) are used.
        """
        if X.ndim == 1:
            X = X[:, None]
        if len(X) < 2:
            raise ValueError("DMDLift.fit requires at least 2 time steps.")

        # Resolve default dictionary
        if self._dictionary is None:
            self._dictionary = PolynomialLift(degree=2)

        X_curr = X[:-1]   # (T-1, n)
        X_next = X[1:]    # (T-1, n)

        # Fit dictionary on current states
        if hasattr(self._dictionary, "fit"):
            self._dictionary.fit(X_curr)

        Psi0 = self._dictionary(X_curr).astype(np.float64)   # (T-1, m)
        Psi1 = self._dictionary(X_next).astype(np.float64)   # (T-1, m)

        T = Psi0.shape[0]

        # EDMD Koopman matrix: K = pinv(Psi0) @ Psi1
        # Use least-squares for numerical stability
        K, _, _, _ = np.linalg.lstsq(Psi0, Psi1, rcond=None)   # (m, m)

        # Eigendecompose K
        evals, evecs = np.linalg.eig(K.T)   # right eigenvectors of K^T = left of K

        # Select n_modes modes by criterion
        if self.sort_by == "real":
            order = np.argsort(evals.real)[::-1]
        else:  # magnitude
            order = np.argsort(np.abs(evals))[::-1]

        n_keep = min(self.n_modes, len(evals))
        idx = order[:n_keep]
        self._evals = evals[idx]           # (n_keep,) complex
        self._evecs = evecs[:, idx]        # (m, n_keep) complex

        # Determine which modes are effectively real (|Im| < tol)
        tol = 1e-8 * np.abs(self._evals).max()
        self._is_real = np.abs(self._evals.imag) < tol

        # Output dim: 1 feature per real mode, 2 per complex mode
        self._output_dim = int(
            self._is_real.sum() + 2 * (~self._is_real).sum()
        )
        return self

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Call lift.fit(X) before accessing output_dim.")
        return self._output_dim

    @property
    def feature_names(self) -> list[str]:
        if self._is_real is None:
            raise RuntimeError("Call lift.fit(X) before accessing feature_names.")
        names = []
        k = 0
        for i, real in enumerate(self._is_real):
            if real:
                names.append(f"phi_{k}")
                k += 1
            else:
                names += [f"Re(phi_{k})", f"Im(phi_{k})"]
                k += 1
        return names

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the EDMD Koopman eigenfunctions.

        Parameters
        ----------
        X : np.ndarray, shape (N, n) or (n,)

        Returns
        -------
        theta : np.ndarray, shape (N, output_dim) or (output_dim,)  [real]
        """
        if self._evecs is None:
            raise RuntimeError("Call lift.fit(X) before applying the lift.")
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]

        Psi = self._dictionary(X).astype(np.float64)   # (N, m)
        # Project: phi_i(x) = evecs[:,i]^H @ Psi(x)
        projs = Psi @ self._evecs.conj()                # (N, n_keep) complex

        cols = []
        for i, real in enumerate(self._is_real):
            if real:
                cols.append(projs[:, i].real[:, None])
            else:
                cols.append(projs[:, i].real[:, None])
                cols.append(projs[:, i].imag[:, None])

        result = np.hstack(cols).astype(np.float64)
        return result[0] if scalar else result

    def __repr__(self) -> str:
        fitted = self._evecs is not None
        return (
            f"DMDLift(n_modes={self.n_modes}, sort_by={self.sort_by!r}, "
            f"fitted={fitted})"
        )
