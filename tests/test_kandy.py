"""Tests for the kandy package.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest

from kandy import KANDy, CustomLift, DelayEmbedding, PolynomialLift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lorenz_data():
    """Tiny synthetic Lorenz-like dataset (no actual integration)."""
    rng = np.random.default_rng(42)
    N, n = 200, 3
    X = rng.standard_normal((N, n)).astype(np.float32)
    # Fake derivatives (just for shape/dtype testing)
    X_dot = rng.standard_normal((N, n)).astype(np.float32)
    return X, X_dot


@pytest.fixture
def henon_series():
    """Small Hénon trajectory."""
    def step(xy, a=1.4, b=0.3):
        x, y = xy
        return np.array([1 - a * x**2 + y, b * x], dtype=np.float32)

    traj = np.zeros((300, 2), dtype=np.float32)
    traj[0] = [0.1, 0.0]
    for i in range(299):
        traj[i + 1] = step(traj[i])
    return traj


# ---------------------------------------------------------------------------
# 1. PolynomialLift
# ---------------------------------------------------------------------------

class TestPolynomialLift:
    def test_output_dim_degree1(self):
        lift = PolynomialLift(degree=1)
        lift.fit(np.zeros((10, 3)))
        assert lift.output_dim == 3

    def test_output_dim_degree2(self):
        lift = PolynomialLift(degree=2)
        lift.fit(np.zeros((10, 3)))
        # degree-2 monomials in R^3: 3 linear + 6 quadratic = 9
        assert lift.output_dim == 9

    def test_output_dim_with_bias(self):
        lift = PolynomialLift(degree=1, include_bias=True)
        lift.fit(np.zeros((10, 2)))
        assert lift.output_dim == 3   # 1 bias + 2 linear

    def test_cross_product_present(self):
        """Cross-terms x_0*x_1, x_0*x_2, x_1*x_2 must be in degree-2 lift."""
        lift = PolynomialLift(degree=2)
        lift.fit(np.zeros((5, 3)))
        names = lift.feature_names
        assert "x_0*x_1" in names
        assert "x_0*x_2" in names
        assert "x_1*x_2" in names

    def test_scalar_input(self):
        lift = PolynomialLift(degree=2)
        lift.fit(np.zeros((5, 2)))
        x = np.array([2.0, 3.0])
        result = lift(x)
        assert result.shape == (lift.output_dim,)
        # x_0*x_1 should equal 6.0
        names = lift.feature_names
        idx = names.index("x_0*x_1")
        assert np.isclose(result[idx], 6.0)

    def test_batch_input(self):
        lift = PolynomialLift(degree=2)
        X = np.random.default_rng(0).standard_normal((50, 3))
        lift.fit(X)
        out = lift(X)
        assert out.shape == (50, 9)

    def test_fit_required_before_output_dim(self):
        lift = PolynomialLift(degree=2)
        with pytest.raises(RuntimeError):
            _ = lift.output_dim


# ---------------------------------------------------------------------------
# 2. DelayEmbedding
# ---------------------------------------------------------------------------

class TestDelayEmbedding:
    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        lift = DelayEmbedding(delays=3)
        lift.fit(X)
        out = lift(X)
        assert out.shape == (50 - 3 + 1, 2 * 3)

    def test_output_dim(self):
        lift = DelayEmbedding(delays=4)
        lift.fit(np.zeros((20, 3)))
        assert lift.output_dim == 12

    def test_short_trajectory_raises(self):
        lift = DelayEmbedding(delays=10)
        lift.fit(np.zeros((5, 2)))
        with pytest.raises(ValueError):
            lift(np.zeros((5, 2)))


# ---------------------------------------------------------------------------
# 3. CustomLift
# ---------------------------------------------------------------------------

class TestCustomLift:
    def test_wraps_callable(self):
        from kandy import CustomLift
        lift = CustomLift(fn=lambda X: X ** 2, output_dim=3)
        X = np.ones((5, 3)) * 2
        out = lift(X)
        assert out.shape == (5, 3)
        assert np.allclose(out, 4.0)

    def test_scalar_input(self):
        from kandy import CustomLift
        lift = CustomLift(fn=lambda X: X, output_dim=2)
        x = np.array([1.0, 2.0])
        out = lift(x)
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# 4. KANDy — API contract (shape / type checks; no actual KAN needed)
# ---------------------------------------------------------------------------

class TestKANDyAPI:
    """These tests verify the API contract without requiring PyKAN.

    If PyKAN is not installed they are skipped; if it is installed the model
    is trained on a tiny synthetic dataset to check end-to-end shapes.
    """

    pykan = pytest.importorskip("kan", reason="PyKAN not installed")

    def test_fit_returns_self(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        result = model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        assert result is model

    def test_predict_shape(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        pred = model.predict(X[:10])
        assert pred.shape == (10, 3)

    def test_predict_scalar_input(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        pred = model.predict(X[0])
        assert pred.shape == (3,)

    def test_rollout_shape(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        traj = model.rollout(X[0], T=20, dt=0.01)
        assert traj.shape == (20, 3)

    def test_rollout_euler(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        traj = model.rollout(X[0], T=10, dt=0.01, integrator="euler")
        assert traj.shape == (10, 3)

    def test_central_diff_fit(self, lorenz_data):
        X, _ = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        # Fit using dt instead of X_dot
        model.fit(X, dt=0.01, val_frac=0.1, test_frac=0.1)
        assert model._is_fitted

    def test_fit_requires_xdot_or_dt(self, lorenz_data):
        X, _ = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, steps=5)
        with pytest.raises(ValueError, match="dt"):
            model.fit(X)

    def test_predict_before_fit_raises(self, lorenz_data):
        X, _ = lorenz_data
        model = KANDy(lift=PolynomialLift(degree=2))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X[0])

    def test_get_A_shape(self, lorenz_data):
        X, X_dot = lorenz_data
        lift = PolynomialLift(degree=2)
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        A = model.get_A()
        if A is not None:
            # A should be (state_dim, lift_dim) = (3, 9)
            assert A.shape == (model.state_dim_, model.lift_dim_)

    def test_custom_lift_integration(self, lorenz_data):
        """End-to-end with a hand-crafted Lorenz lift."""
        X, X_dot = lorenz_data

        def lorenz_phi(states):
            x, y, z = states[:, 0], states[:, 1], states[:, 2]
            return np.column_stack([x, y, z, x**2, x*y, x*z, y**2, y*z, z**2])

        lift = CustomLift(fn=lorenz_phi, output_dim=9, name="lorenz")
        model = KANDy(lift=lift, grid=3, k=3, steps=5, seed=42)
        model.fit(X, X_dot, val_frac=0.1, test_frac=0.1)
        traj = model.rollout(X[0], T=10, dt=0.01)
        assert traj.shape == (10, 3)
