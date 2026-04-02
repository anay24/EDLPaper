import numpy as np

class ControlLyapunovFunction:
    """
    Base class for Control Lyapunov Functions (CLFs)
    """

    def V(self, x: np.ndarray) -> float:
        """
        Evaluate the CLF at state x
        Must be implemented by derived classes
        """
        raise NotImplementedError

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of the CLF w.r.t state x
        Must be implemented by derived classes
        """
        raise NotImplementedError

    def lie_derivative(self, f: callable, g: callable, x: np.ndarray) -> float:
        """
        Compute the Lie derivative of the CLF along the dynamics: L_f V + L_g V u

        Args:
            f: function f(x) representing the drift dynamics
            g: function g(x) representing the control dynamics
            x: current state (numpy array)

        Returns:
            LfV, LgV components of the Lie derivative
        """
        grad = self.grad_V(x)
        LfV = grad @ f(x)
        LgV = grad @ g(x)
        return LfV, LgV


class QuadraticCLF(ControlLyapunovFunction):
    """
    Classic quadratic CLF: V(x) = (x-c)^T P (x-c) + d^T (x-c) + e
    """

    def __init__(self, P: np.ndarray, c: np.ndarray = None, d: np.ndarray = None, e: float = 0.0):
        self.P = P
        self.c = c if c is not None else np.zeros(P.shape[0])
        self.d = d if d is not None else np.zeros(P.shape[0])
        self.e = e

        if not np.allclose(P, P.T):
            raise ValueError("Matrix P must be symmetric")
        if not np.all(np.linalg.eigvalsh(P) >= -1e-10):
            raise ValueError("Matrix P must be positive semidefinite")

    def V(self, x: np.ndarray) -> float:
        x = np.atleast_1d(x)
        x_shifted = x - self.c
        return float(x_shifted.T @ self.P @ x_shifted) + self.d.T @ x_shifted + self.e

    def grad_V(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        x_shifted = x - self.c
        return 2 * self.P @ x_shifted + self.d
