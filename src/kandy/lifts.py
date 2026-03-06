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


class DelayEmbedding(Lift):
    """Takens delay-coordinate embedding.

    Lifts a trajectory to a delay-coordinate representation.  For input
    dimension n and d delays, each time window [x_t, x_{t-1}, ..., x_{t-d+1}]
    is stacked into a n*d dimensional feature vector.

    Parameters
    ----------
    delays : int
        Number of delay steps (default 3).  Output dim = n * delays.

    Notes
    -----
    ``__call__`` takes a full trajectory of shape (T, n) and returns
    (T - delays + 1, n * delays).  Row i contains
    [x_{i+d-1}, ..., x_i] (most-recent first).
    """

    def __init__(self, delays: int = 3):
        self.delays = delays
        self._input_dim: Optional[int] = None

    def fit(self, X: np.ndarray) -> "DelayEmbedding":
        self._input_dim = X.shape[1] if X.ndim > 1 else 1
        return self

    @property
    def output_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError("Call lift.fit(X) before accessing output_dim.")
        return self._input_dim * self.delays

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply delay embedding to trajectory X of shape (T, n).

        Returns
        -------
        embedded : np.ndarray, shape (T - delays + 1, n * delays)
        """
        if X.ndim == 1:
            X = X[:, None]
        if self._input_dim is None:
            self.fit(X)
        T, n = X.shape
        if T < self.delays:
            raise ValueError(
                f"Trajectory length {T} is shorter than delays={self.delays}."
            )
        cols = [X[self.delays - 1 - lag : T - lag] for lag in range(self.delays)]
        return np.hstack(cols)


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


# ---------------------------------------------------------------------------
# EXPERIMENTAL: KAN Autoencoder Koopman Lift  (KANELift)
# ---------------------------------------------------------------------------

class KANELift(Lift):
    """**EXPERIMENTAL** — KAN-based deep Koopman autoencoder lift.

    .. warning::
        This class is experimental and currently under active development.
        Training is less stable than MLP-based autoencoders.  Results may
        not reproduce across runs even with fixed seeds.  The API may change
        in future versions.

    Overview
    --------
    Learns the Koopman lift φ : ℝⁿ → ℝᵐ (the encoder) and its approximate
    inverse (the decoder) jointly as KAN networks, together with a linear
    Koopman propagator K ∈ ℝᵐˣᵐ:

    ::

        encoder : x_t  ──KAN──►  z_t   ∈ ℝᵐ
        propagator :    z_t  ──K──►   z_{t+1}
        decoder :  z_{t+1} ──KAN──►  x_{t+1}

    Training minimises three losses jointly:

    * **Prediction loss**      MSE(decoder(K·encoder(xₜ)), x_{t+1})
    * **Latent consistency**   MSE(K·encoder(xₜ), encoder(x_{t+1}))
    * **Reconstruction loss**  MSE(decoder(encoder(xₜ)), xₜ)

    After training, ``__call__(X)`` applies only the encoder, making this
    a drop-in replacement for any other lift.  Because the encoder is a KAN,
    symbolic formulas for φ can be extracted via ``get_formula()``, giving
    an interpretable, data-driven Koopman lift rather than a polynomial guess.

    Design philosophy — favouring shallowness
    ------------------------------------------
    Standard KANs should be single-layer for symbolic readability (the bilinear
    obstruction means cross-terms must be pre-encoded, not learned by depth).
    Here the encoder itself *is* the cross-term discovery step, so a single
    hidden layer is permitted.  The default architecture is:

    * Encoder: ``[n, latent_dim]``  (single-layer — no hidden layer)
    * Decoder: ``[latent_dim, n]``  (single-layer)

    Pass ``hidden_dim`` to add exactly one hidden layer:

    * Encoder: ``[n, hidden_dim, latent_dim]``
    * Decoder: ``[latent_dim, hidden_dim, n]``

    More hidden layers are not supported; they defeat the interpretability goal.

    Parameters
    ----------
    latent_dim : int
        Dimension of the Koopman latent space (encoded coordinates m).
    hidden_dim : int or None
        Width of an optional single hidden KAN layer.  Default None = no
        hidden layer (single-layer encoder/decoder).
    grid : int
        KAN spline grid points per segment (default 5).
    k : int
        KAN spline degree (default 3, i.e. cubic).
    base_fun : callable or None
        Base activation for each KAN edge.  Default: RBF ``exp(-x²)``.
        ``'silu'`` uses the PyKAN default SiLU.
    seed : int
        Random seed (default 0).

    Attributes
    ----------
    encoder_ : KAN
        Fitted encoder KAN (available after ``train_koopman``).
    decoder_ : KAN
        Fitted decoder KAN (available after ``train_koopman``).
    K_ : torch.nn.Linear
        Fitted linear Koopman propagator K (no bias).
    train_history_ : dict
        Loss curves from ``train_koopman``.

    Examples
    --------
    >>> from kandy import KANELift, KANDy
    >>> lift = KANELift(latent_dim=8, hidden_dim=None)
    >>> lift.train_koopman(traj, dt=0.01, epochs=50, lr=1e-3)
    >>> model = KANDy(lift=lift, grid=5, k=3)
    >>> model.fit(X, X_dot)
    >>> # Extract symbolic encoder formulas
    >>> enc_formulas = lift.get_formula(var_names=["x", "y", "z"])
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        grid: int = 5,
        k: int = 3,
        base_fun=None,
        seed: int = 0,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        self.seed = seed

        self.encoder_: Optional[object] = None
        self.decoder_: Optional[object] = None
        self.K_: Optional[object] = None
        self.train_history_: Optional[dict] = None
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Call train_koopman() before using the lift.")
        return self._output_dim

    def _make_base_fun(self):
        import torch
        if self.base_fun is None:
            return lambda x: torch.exp(-(x ** 2))   # RBF default
        return self.base_fun

    def _build_kans(self, input_dim: int):
        from kan import KAN
        import torch
        base = self._make_base_fun()

        if self.hidden_dim is None:
            enc_width = [input_dim, self.latent_dim]
            dec_width = [self.latent_dim, input_dim]
        else:
            enc_width = [input_dim, self.hidden_dim, self.latent_dim]
            dec_width = [self.latent_dim, self.hidden_dim, input_dim]

        encoder = KAN(
            width=enc_width, grid=self.grid, k=self.k,
            base_fun=base, seed=self.seed,
        )
        decoder = KAN(
            width=dec_width, grid=self.grid, k=self.k,
            base_fun=base, seed=self.seed,
        )
        K = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        return encoder, decoder, K

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_koopman(
        self,
        traj: "np.ndarray",
        dt: float = 1.0,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 512,
        alpha_latent: float = 1.0,
        beta_recon: float = 1.0,
        gamma_predx: float = 1.0,
        grad_clip: float = 1.0,
        val_frac: float = 0.2,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "KANELift":
        """Train the KAN autoencoder Koopman model on trajectory data.

        Parameters
        ----------
        traj : np.ndarray
            Trajectory array of shape ``(T, n)`` (single trajectory) or
            ``(B, T, n)`` (batch of trajectories).  Consecutive rows are
            treated as (xₜ, x_{t+1}) pairs.
        dt : float
            Time step.  Used only if the Koopman propagator is interpreted as
            a continuous-time generator via matrix exponential (future work).
            Currently stored but not applied — K is trained as a discrete-map
            propagator directly.
        epochs : int
            Number of training epochs (default 50).
        lr : float
            Adam learning rate (default 1e-3).
        batch_size : int
            Mini-batch size (default 512).
        alpha_latent : float
            Weight on the latent consistency loss (default 1.0).
        beta_recon : float
            Weight on the reconstruction loss (default 1.0).
        gamma_predx : float
            Weight on the one-step prediction loss (default 1.0).
        grad_clip : float
            Gradient norm clip value.  Set to None to disable.
        val_frac : float
            Fraction of pairs held out for validation (default 0.2).
        device : str or None
            ``'cpu'``, ``'cuda'``, or None (auto-detect).
        verbose : bool
            Print epoch-level progress.

        Returns
        -------
        self
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Flatten to (N, n) pairs
        traj_np = np.asarray(traj, dtype=np.float32)
        if traj_np.ndim == 3:
            # (B, T, n) → concatenate all pairs
            x_t   = traj_np[:, :-1, :].reshape(-1, traj_np.shape[-1])
            x_tp1 = traj_np[:, 1:,  :].reshape(-1, traj_np.shape[-1])
        else:
            x_t   = traj_np[:-1]
            x_tp1 = traj_np[1:]

        input_dim = x_t.shape[1]
        self._input_dim  = input_dim
        self._output_dim = self.latent_dim

        # Train / val split
        n = len(x_t)
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val

        x_t_tr,   x_tp1_tr   = x_t[:n_train],   x_tp1[:n_train]
        x_t_val,  x_tp1_val  = x_t[n_train:],   x_tp1[n_train:]

        def _ds(a, b):
            return TensorDataset(
                torch.tensor(a, dtype=torch.float32),
                torch.tensor(b, dtype=torch.float32),
            )

        train_dl = DataLoader(_ds(x_t_tr, x_tp1_tr),
                              batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl   = DataLoader(_ds(x_t_val, x_tp1_val),
                              batch_size=batch_size, shuffle=False)

        encoder, decoder, K = self._build_kans(input_dim)
        encoder = encoder.to(dev)
        decoder = decoder.to(dev)
        K       = K.to(dev)

        # Disable save_act during gradient training — only enable for symbolic
        encoder.save_act = False
        decoder.save_act = False

        params = list(encoder.parameters()) + list(decoder.parameters()) + list(K.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        mse = torch.nn.MSELoss()

        history: dict = {
            "train_total": [], "train_predx": [], "train_latent": [], "train_recon": [],
            "val_total":   [], "val_predx":   [], "val_latent":   [], "val_recon":   [],
        }

        def _forward(x_t_b, x_tp1_b):
            z_t   = encoder(x_t_b)                   # encode current
            x_rec = decoder(z_t)                      # reconstruct
            z_tp1_pred = K(z_t)                       # propagate in latent
            x_tp1_pred = decoder(z_tp1_pred)          # decode prediction
            z_tp1_true = encoder(x_tp1_b)             # encode true next
            return z_tp1_pred, z_tp1_true, x_tp1_pred, x_rec

        def _losses(z_tp1_pred, z_tp1_true, x_tp1_pred, x_rec, x_t_b, x_tp1_b):
            loss_predx  = mse(x_tp1_pred, x_tp1_b)
            loss_latent = mse(z_tp1_pred, z_tp1_true)
            loss_recon  = mse(x_rec, x_t_b)
            total = (gamma_predx * loss_predx
                     + alpha_latent * loss_latent
                     + beta_recon   * loss_recon)
            return total, loss_predx, loss_latent, loss_recon

        def _eval():
            encoder.eval(); decoder.eval(); K.eval()
            ts, ps, ls, rs = [], [], [], []
            with torch.no_grad():
                for x_t_b, x_tp1_b in val_dl:
                    x_t_b   = x_t_b.to(dev)
                    x_tp1_b = x_tp1_b.to(dev)
                    out = _forward(x_t_b, x_tp1_b)
                    t, p, l, r = _losses(*out, x_t_b, x_tp1_b)
                    ts.append(t.item()); ps.append(p.item())
                    ls.append(l.item()); rs.append(r.item())
            return (float(np.mean(ts)), float(np.mean(ps)),
                    float(np.mean(ls)), float(np.mean(rs)))

        for ep in range(1, epochs + 1):
            encoder.train(); decoder.train(); K.train()
            ts, ps, ls, rs = [], [], [], []

            for x_t_b, x_tp1_b in train_dl:
                x_t_b   = x_t_b.to(dev)
                x_tp1_b = x_tp1_b.to(dev)

                out = _forward(x_t_b, x_tp1_b)
                total, lp, ll, lr_ = _losses(*out, x_t_b, x_tp1_b)

                opt.zero_grad(set_to_none=True)
                total.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()

                ts.append(total.item()); ps.append(lp.item())
                ls.append(ll.item());    rs.append(lr_.item())

            vt, vp, vl, vr = _eval()

            history["train_total"].append(float(np.mean(ts)))
            history["train_predx"].append(float(np.mean(ps)))
            history["train_latent"].append(float(np.mean(ls)))
            history["train_recon"].append(float(np.mean(rs)))
            history["val_total"].append(vt)
            history["val_predx"].append(vp)
            history["val_latent"].append(vl)
            history["val_recon"].append(vr)

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                print(
                    f"[KANELift] ep {ep:03d}/{epochs} | "
                    f"train {history['train_total'][-1]:.5f} "
                    f"(pred {history['train_predx'][-1]:.5f}, "
                    f"latent {history['train_latent'][-1]:.5f}, "
                    f"recon {history['train_recon'][-1]:.5f}) | "
                    f"val {vt:.5f}"
                )

        self.encoder_      = encoder
        self.decoder_      = decoder
        self.K_            = K
        self.train_history_ = history
        self.dt            = dt
        return self

    # ------------------------------------------------------------------
    # Lift interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KANELift":
        """No-op: fitting is done via :meth:`train_koopman`."""
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply the encoder (the learned Koopman lift) to X.

        Parameters
        ----------
        X : np.ndarray, shape (N, n) or (n,)

        Returns
        -------
        theta : np.ndarray, shape (N, latent_dim) or (latent_dim,)
        """
        import torch
        if self.encoder_ is None:
            raise RuntimeError("Call train_koopman() before applying the lift.")
        scalar = X.ndim == 1
        if scalar:
            X = X[None, :]
        t = torch.tensor(X, dtype=torch.float32)
        self.encoder_.eval()
        self.encoder_.save_act = False
        with torch.no_grad():
            z = self.encoder_(t).cpu().numpy()
        return z[0] if scalar else z

    # ------------------------------------------------------------------
    # Symbolic extraction from the encoder KAN
    # ------------------------------------------------------------------

    def get_formula(
        self,
        var_names: Optional[list] = None,
        round_places: int = 3,
    ) -> list:
        """Extract symbolic formulas for the encoder (the learned lift).

        Calls ``auto_symbolic`` on the encoder KAN and returns one SymPy
        expression per latent dimension — i.e. the symbolic form of each
        Koopman observable φᵢ(x).

        Parameters
        ----------
        var_names : list of str, optional
            Names for input state variables (length n).
            Defaults to ``['x_0', 'x_1', ...]``.
        round_places : int
            Decimal rounding for numeric constants.

        Returns
        -------
        formulas : list of sympy.Expr
            One expression per latent dimension.
        """
        import torch
        import sympy as sp

        if self.encoder_ is None:
            raise RuntimeError("Call train_koopman() before extracting formulas.")

        enc = self.encoder_
        enc.eval()
        enc.save_act = True

        # We need a forward pass to populate activations.
        # Use a small random batch since we may not have training data.
        # Users should call this after passing real data through if possible.
        n = self._input_dim
        with torch.no_grad():
            enc(torch.zeros(1, n))

        enc.auto_symbolic()
        exprs_raw, inputs = enc.symbolic_formula()

        if var_names is None:
            var_names = [f"x_{i}" for i in range(n)]

        sub_map = {
            sp.Symbol(str(inp)): sp.Symbol(name)
            for inp, name in zip(inputs, var_names)
        }

        formulas = []
        for expr in exprs_raw:
            if not isinstance(expr, (list, tuple)):
                expr = [expr]
            for e in expr:
                try:
                    sym = sp.sympify(e)
                    sym = sym.xreplace(sub_map)
                    sym = sym.xreplace(
                        {a: round(float(a), round_places) for a in sym.atoms(sp.Number)}
                    )
                    formulas.append(sp.expand(sym))
                except Exception:
                    pass

        enc.save_act = False
        return formulas

    @property
    def feature_names(self) -> list:
        return [f"phi_{i}" for i in range(self.latent_dim)]

    def __repr__(self) -> str:
        arch = (f"[{self._input_dim}, {self.latent_dim}]"
                if self.hidden_dim is None
                else f"[{self._input_dim}, {self.hidden_dim}, {self.latent_dim}]")
        fitted = self.encoder_ is not None
        return f"KANELift(latent_dim={self.latent_dim}, arch={arch}, fitted={fitted})"
