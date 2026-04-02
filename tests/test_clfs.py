import numpy as np
import pytest
from cbfpdg.clfs import ControlLyapunovFunction, QuadraticCLF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_clf():
    """QuadraticCLF with P = I (2x2)."""
    return QuadraticCLF(np.eye(2))


@pytest.fixture
def diagonal_clf():
    """QuadraticCLF with P = diag(2, 3)."""
    return QuadraticCLF(np.diag([2.0, 3.0]))


@pytest.fixture
def simple_dynamics():
    """Harmonic-oscillator-style drift f and identity control g."""
    def f(x):
        return np.array([x[1], -x[0]])

    def g(x):
        return np.eye(2)

    return f, g


# ---------------------------------------------------------------------------
# ControlLyapunovFunction base class
# ---------------------------------------------------------------------------

class TestControlLyapunovFunctionBase:
    def test_V_raises_not_implemented(self):
        clf = ControlLyapunovFunction()
        with pytest.raises(NotImplementedError):
            clf.V(np.array([1.0, 0.0]))

    def test_grad_V_raises_not_implemented(self):
        clf = ControlLyapunovFunction()
        with pytest.raises(NotImplementedError):
            clf.grad_V(np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# QuadraticCLF construction
# ---------------------------------------------------------------------------

class TestQuadraticCLFConstruction:
    def test_valid_symmetric_pd_matrix(self):
        P = np.array([[2.0, 0.5], [0.5, 1.0]])
        clf = QuadraticCLF(P)
        assert clf.P is P

    def test_asymmetric_matrix_raises(self):
        P = np.array([[1.0, 2.0], [3.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            QuadraticCLF(P)

    def test_negative_eigenvalue_raises(self):
        P = np.array([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError, match="positive semidefinite"):
            QuadraticCLF(P)

    def test_singular_matrix_accepted(self):
        # PSD (singular) matrices are allowed; V(x) = 0 in the null-space direction
        P = np.array([[1.0, 0.0], [0.0, 0.0]])
        clf = QuadraticCLF(P)
        assert np.isclose(clf.V(np.array([1.0, 99.0])), 1.0)

    def test_identity_matrix_accepted(self):
        clf = QuadraticCLF(np.eye(3))
        assert clf.P.shape == (3, 3)

    def test_larger_pd_matrix(self):
        P = np.diag([1.0, 2.0, 3.0])
        clf = QuadraticCLF(P)
        assert clf.P.shape == (3, 3)


# ---------------------------------------------------------------------------
# QuadraticCLF.V
# ---------------------------------------------------------------------------

class TestQuadraticCLFV:
    def test_origin_is_zero(self, identity_clf):
        assert identity_clf.V(np.zeros(2)) == 0.0

    def test_identity_P_equals_squared_norm(self, identity_clf):
        x = np.array([3.0, 4.0])
        assert np.isclose(identity_clf.V(x), 25.0)

    def test_returns_float(self, identity_clf):
        result = identity_clf.V(np.array([1.0, 1.0]))
        assert isinstance(result, float)

    def test_positive_away_from_origin(self, identity_clf):
        x = np.array([0.1, 0.0])
        assert identity_clf.V(x) > 0.0

    def test_diagonal_P(self, diagonal_clf):
        x = np.array([1.0, 1.0])
        # V = 2*1^2 + 3*1^2 = 5
        assert np.isclose(diagonal_clf.V(x), 5.0)

    def test_symmetry_in_x(self, identity_clf):
        x = np.array([2.0, -3.0])
        assert np.isclose(identity_clf.V(x), identity_clf.V(-x))

    def test_1d_input_treated_as_vector(self):
        clf = QuadraticCLF(np.array([[4.0]]))
        assert np.isclose(clf.V(np.array([2.0])), 16.0)

    def test_scaling(self, identity_clf):
        x = np.array([1.0, 0.0])
        assert np.isclose(identity_clf.V(2 * x), 4 * identity_clf.V(x))

    def test_3d_system(self):
        P = np.diag([1.0, 2.0, 3.0])
        clf = QuadraticCLF(P)
        x = np.array([1.0, 1.0, 1.0])
        assert np.isclose(clf.V(x), 6.0)


# ---------------------------------------------------------------------------
# QuadraticCLF.grad_V
# ---------------------------------------------------------------------------

class TestQuadraticCLFGradV:
    def test_identity_P_gradient(self, identity_clf):
        x = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(identity_clf.grad_V(x), 2 * x)

    def test_gradient_at_origin_is_zero(self, identity_clf):
        np.testing.assert_array_equal(identity_clf.grad_V(np.zeros(2)), np.zeros(2))

    def test_gradient_shape(self, identity_clf):
        x = np.array([1.0, 2.0])
        assert identity_clf.grad_V(x).shape == x.shape

    def test_diagonal_P_gradient(self, diagonal_clf):
        x = np.array([1.0, 1.0])
        # grad = 2 * P @ x = 2 * [2, 3] = [4, 6]
        np.testing.assert_array_almost_equal(diagonal_clf.grad_V(x), np.array([4.0, 6.0]))

    def test_gradient_matches_numerical(self, diagonal_clf):
        """Verify analytic gradient against a finite-difference approximation."""
        x = np.array([1.5, -0.7])
        eps = 1e-6
        numerical_grad = np.array([
            (diagonal_clf.V(x + eps * np.eye(2)[i]) - diagonal_clf.V(x - eps * np.eye(2)[i])) / (2 * eps)
            for i in range(2)
        ])
        np.testing.assert_array_almost_equal(diagonal_clf.grad_V(x), numerical_grad, decimal=5)

    def test_gradient_is_linear_in_x(self, identity_clf):
        x1, x2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        combined = identity_clf.grad_V(x1 + x2)
        summed = identity_clf.grad_V(x1) + identity_clf.grad_V(x2)
        np.testing.assert_array_almost_equal(combined, summed)


# ---------------------------------------------------------------------------
# QuadraticCLF.lie_derivative
# ---------------------------------------------------------------------------

class TestLieDerivative:
    def test_returns_tuple_of_two(self, identity_clf, simple_dynamics):
        f, g = simple_dynamics
        result = identity_clf.lie_derivative(f, g, np.array([1.0, 2.0]))
        assert isinstance(result, tuple) and len(result) == 2

    def test_LfV_harmonic_oscillator(self, identity_clf, simple_dynamics):
        """For V = ||x||^2, f = [x1, -x0]: LfV = grad_V . f = 2x0*x1 + 2x1*(-x0) = 0."""
        f, g = simple_dynamics
        x = np.array([1.0, 2.0])
        LfV, _ = identity_clf.lie_derivative(f, g, x)
        assert np.isclose(LfV, 0.0)

    def test_LgV_identity_g(self, identity_clf, simple_dynamics):
        """For V = ||x||^2, g = I: LgV = grad_V = 2x."""
        f, g = simple_dynamics
        x = np.array([1.0, 2.0])
        _, LgV = identity_clf.lie_derivative(f, g, x)
        np.testing.assert_array_almost_equal(LgV, 2 * x)

    def test_LfV_zero_drift(self, identity_clf):
        """If f(x) = 0 for all x, LfV must be 0."""
        def f_zero(x):
            return np.zeros_like(x)

        def g_id(x):
            return np.eye(len(x))

        x = np.array([3.0, -1.0])
        LfV, _ = identity_clf.lie_derivative(f_zero, g_id, x)
        assert np.isclose(LfV, 0.0)

    def test_LgV_zero_control(self, identity_clf):
        """If g(x) = 0 for all x, LgV must be 0."""
        def f_id(x):
            return x

        def g_zero(x):
            return np.zeros((len(x), len(x)))

        x = np.array([1.0, 2.0])
        _, LgV = identity_clf.lie_derivative(f_id, g_zero, x)
        np.testing.assert_array_almost_equal(LgV, np.zeros(2))

    def test_LfV_with_custom_P_and_dynamics(self, diagonal_clf):
        """Manually verify LfV for P = diag(2,3), f(x) = [x1, -x0]."""
        def f(x):
            return np.array([x[1], -x[0]])

        def g(x):
            return np.eye(2)

        x = np.array([1.0, 1.0])
        # grad_V = 2*P@x = [4, 6], f(x) = [1, -1]
        # LfV = [4,6].[1,-1] = 4 - 6 = -2
        LfV, _ = diagonal_clf.lie_derivative(f, g, x)
        assert np.isclose(LfV, -2.0)

    def test_at_origin_LfV_is_zero(self, identity_clf, simple_dynamics):
        f, g = simple_dynamics
        LfV, _ = identity_clf.lie_derivative(f, g, np.zeros(2))
        assert np.isclose(LfV, 0.0)

    def test_at_origin_LgV_is_zero(self, identity_clf, simple_dynamics):
        f, g = simple_dynamics
        _, LgV = identity_clf.lie_derivative(f, g, np.zeros(2))
        np.testing.assert_array_almost_equal(LgV, np.zeros(2))

    def test_scalar_1d_system(self):
        """Single-dimensional system: V = p*x^2, f = -x, g = 1."""
        clf = QuadraticCLF(np.array([[2.0]]))

        def f(x):
            return -x

        def g(x):
            return np.array([1.0])

        x = np.array([3.0])
        # grad_V = 2*2*3 = 12; LfV = 12 * (-3) = -36; LgV = 12 * 1 = 12
        LfV, LgV = clf.lie_derivative(f, g, x)
        assert np.isclose(LfV, -36.0)
        assert np.isclose(LgV, 12.0)